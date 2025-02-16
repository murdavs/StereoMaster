import os
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import glob
import gc
import cv2
import time
import numpy as np
import shutil
import subprocess
from fire import Fire
from decord import VideoReader, cpu
import torch
import torch.nn.functional as F
from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer
cm = ColorMatcher()
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline, tensor2vid
import imageio_ffmpeg 


#######################################################
# Additional helper functions for local color matching
#######################################################
def find_best_match_region(template_np, search_np):
    """
    Searches for the best match of 'template_np' in 'search_np'
    using cv2.matchTemplate. Returns (top_y, left_x) as location.
    """
    tmpl_gray = cv2.cvtColor(template_np, cv2.COLOR_BGR2GRAY)
    srch_gray = cv2.cvtColor(search_np, cv2.COLOR_BGR2GRAY)
    if tmpl_gray.shape[0] > srch_gray.shape[0] or tmpl_gray.shape[1] > srch_gray.shape[1]:
        return 0, 0

    method = cv2.TM_CCOEFF_NORMED
    res = cv2.matchTemplate(srch_gray, tmpl_gray, method)
    _, _, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    return (top_left[1], top_left[0])  # (row, col)


def local_color_match_masked_region(final_right, left_frame, mask, cm):
    """
    Takes 'final_right' (inpainted area), finds bounding box in 'mask',
    searches a similar region in 'left_frame', and applies color transfer.
    Returns the updated 'final_right'.
    """
    mask_indices = (mask[0] > 0).nonzero(as_tuple=True)
    if len(mask_indices[0]) == 0:
        return final_right

    y_min, y_max = mask_indices[0].min(), mask_indices[0].max()
    x_min, x_max = mask_indices[1].min(), mask_indices[1].max()

    # Crop region from final_right
    region_crop = final_right[:, y_min:y_max+1, x_min:x_max+1].clone()
    region_crop_np = (region_crop.permute(1,2,0).cpu().numpy() * 255.0).clip(0,255).astype("uint8")

    # Convert left_frame to numpy
    left_frame_np = (left_frame.permute(1,2,0).cpu().numpy() * 255.0).clip(0,255).astype("uint8")

    # Find best match in left_frame_np
    top_y, left_x = find_best_match_region(region_crop_np, left_frame_np)
    Hr, Wr, _ = region_crop_np.shape
    left_sub = left_frame_np[top_y:top_y+Hr, left_x:left_x+Wr, :].copy()
    if left_sub.shape[0] != Hr or left_sub.shape[1] != Wr:
        return final_right

    # Local color transfer
    matched_np = cm.transfer(src=region_crop_np, ref=left_sub, method='mkl')
    matched_np = np.clip(matched_np, 0, 255).astype('uint8')

    matched_t = (torch.from_numpy(matched_np).float().permute(2,0,1)/255.0).to(final_right.device)

    # Replace region in final_right
    out_right = final_right.clone()
    out_right[:, y_min:y_max+1, x_min:x_max+1] = matched_t

    return out_right


#######################################################
# 0) GPU Memory Utility (optional)
#######################################################
def report_gpu_mem(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved  = torch.cuda.memory_reserved() / 1024**2
    print(f"[GPU MEM] {prefix} => allocated={allocated:.2f}MB, reserved={reserved:.2f}MB")


#######################################################
# 1) Overlapping tiles: blending functions
#######################################################
def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """
    Horizontally blend two tiles 'a' (left) and 'b' (right)
    over the specified overlap columns.
    """
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(b.device)
    b[:, :, :, :overlap_size] = (1 - weight_b)*a[:, :, :, -overlap_size:] + weight_b*b[:, :, :, :overlap_size]
    return b

def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """
    Vertically blend two tiles 'a' (above) and 'b' (below)
    over the specified overlap rows.
    """
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(b.device)
    b[:, :, :overlap_size, :] = (1 - weight_b)*a[:, :, -overlap_size:, :] + weight_b*b[:, :, :overlap_size, :]
    return b


#######################################################
# 2) Padding for tiling
#######################################################
def pad_for_tiling(frames: torch.Tensor, tile_num: int, tile_overlap=(128, 128)) -> torch.Tensor:
    """
    If tile_num <= 1, no padding is needed.
    Otherwise, zero-pad the frames so each tile can be
    extracted with the required overlap.
    """
    if tile_num <= 1:
        return frames

    T, C, H, W = frames.shape
    ov_y, ov_x = tile_overlap

    tile_size_y = (H + ov_y*(tile_num-1)) // tile_num
    tile_size_x = (W + ov_x*(tile_num-1)) // tile_num

    stride_y = tile_size_y - ov_y
    stride_x = tile_size_x - ov_x

    ideal_h = stride_y*tile_num + ov_y*(tile_num-1)
    ideal_w = stride_y*tile_num + ov_x*(tile_num-1)  # We do NOT modify original logic

    pad_bottom = max(0, ideal_h - H)
    pad_right  = max(0, ideal_w - W)

    if pad_bottom > 0 or pad_right > 0:
        frames = F.pad(frames, (0, pad_right, 0, pad_bottom), mode="constant", value=0.0)
    return frames


#######################################################
# 3) Spatial tiling with overlap => spatial_tiled_process
#######################################################
def spatial_tiled_process(
    cond_frames: torch.Tensor,
    mask_frames: torch.Tensor,
    pipeline_func,
    tile_num: int,
    spatial_n_compress: int=8,
    **kargs
):
    start_t = time.time()

    height = cond_frames.shape[2]
    width  = cond_frames.shape[3]
    tile_overlap = (128, 128)

    tile_size = (
        int((height + tile_overlap[0]*(tile_num-1)) / tile_num),
        int((width  + tile_overlap[1]*(tile_num-1)) / tile_num)
    )
    tile_stride = (
        tile_size[0] - tile_overlap[0],
        tile_size[1] - tile_overlap[1]
    )

    row_tiles = []
    for i in range(tile_num):
        col_tiles = []
        for j in range(tile_num):
            cond_tile = cond_frames[
                :,
                :,
                i*tile_stride[0]: i*tile_stride[0] + tile_size[0],
                j*tile_stride[1]: j*tile_stride[1] + tile_size[1]
            ]
            mask_tile = mask_frames[
                :,
                :,
                i*tile_stride[0]: i*tile_stride[0] + tile_size[0],
                j*tile_stride[1]: j*tile_stride[1] + tile_size[1]
            ]

            t0 = time.time()
            out = pipeline_func(
                frames=cond_tile,
                frames_mask=mask_tile,
                height=cond_tile.shape[2],
                width=cond_tile.shape[3],
                num_frames=len(cond_tile),
                output_type="latent",
                **kargs
            )
            dt_tile = time.time() - t0

            tile_latent = out.frames[0]  # [T, C, H_tile/8, W_tile/8]
            col_tiles.append(tile_latent)
        row_tiles.append(col_tiles)

    latent_stride = (
        tile_stride[0] // spatial_n_compress,
        tile_stride[1] // spatial_n_compress
    )
    latent_overlap = (
        tile_overlap[0] // spatial_n_compress,
        tile_overlap[1] // spatial_n_compress
    )

    # Vertical/horizontal blending
    for i in range(tile_num):
        for j in range(tile_num):
            tile = row_tiles[i][j]
            if i > 0:
                tile = blend_v(row_tiles[i-1][j], tile, latent_overlap[0])
            if j > 0:
                tile = blend_h(row_tiles[i][j-1], tile, latent_overlap[1])
            row_tiles[i][j] = tile

    # Stitch horizontally in each row
    for i in range(tile_num):
        line_ = row_tiles[i]
        for j in range(tile_num):
            tile = line_[j]
            if i < tile_num - 1:
                tile = tile[:, :, :latent_stride[0], :]
            if j < tile_num - 1:
                tile = tile[:, :, :, :latent_stride[1]]
            line_[j] = tile
        row_cat = torch.cat(line_, dim=3)
        row_tiles[i] = row_cat

    final_latent = torch.cat(row_tiles, dim=2)
    dt_spatial = time.time() - start_t

    return final_latent


import OpenEXR
import Imath
import array

def save_exr_sequence_color(frames_np, out_dir, half_float=False):
    """
    Saves a sequence of color frames as EXR files using OpenCV.
    frames_np => shape [T, H, W, 3], float32 [0..1].
    half_float => if True, stores as half-float (16-bit).
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    T, H, W, C = frames_np.shape
    if C != 3:
        print("[ERROR] => frames_np must be 3 channels.")
        return

    for i in range(T):
        frame = frames_np[i]
        exr_flags = []
        if half_float:
            exr_flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]
        else:
            exr_flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]

        out_exr = os.path.join(out_dir, f"frame_{i:04d}.exr")
        success = cv2.imwrite(out_exr, frame, exr_flags)
        if not success:
            print(f"[ERROR] => Could not write EXR => {out_exr}")
        else:
            print(f"[EXR] => Wrote {out_exr}")


def float32_to_half_bytes(frame_2d):
    import Imath, array
    H,W = frame_2d.shape
    arr_ = []
    for val in frame_2d.flatten():
        half_ = Imath.FloatToHalf(val)
        arr_.append(half_)
    return array.array('H', arr_).tobytes()


def srgb_to_acescg(frames_np):
    frames_lin = srgb_to_linear(frames_np)
    SRGB_TO_ACES = np.array([
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777]
    ], dtype=np.float32)
    out_aces = np.einsum("...rc,cd->...rd", frames_lin, SRGB_TO_ACES)
    return out_aces

def srgb_to_linear(u):
    mask = (u <= 0.04045)
    u_out = np.empty_like(u)
    u_out[mask]  = (u[mask]/12.92)
    u_out[~mask] = ((u[~mask]+0.055)/1.055)**2.4
    return u_out


#######################################################
# 4) Write final video with FFmpeg
#######################################################
def write_video_ffmpeg(
    frames_list,
    fps: float,
    output_path: str,
    codec: str = "libx264",
    crf: int=0,
    preset: str="veryslow"
):
    """
    Encodes 'frames_list' ([H,W,3] np.uint8) into mp4.
    """
    if not frames_list:
        print("[WARN] => No frames => skipping =>", output_path)
        return

    frames_np = np.stack(frames_list, axis=0)
    if frames_np.dtype != np.uint8:
        frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)

    T, H, W, C = frames_np.shape
    if C != 3:
        print("[WARN] => frames must be 3 channels => skipping =>", output_path)
        return

    import imageio_ffmpeg
    cmd_params = [
        "-crf", str(crf),
        "-preset", preset
    ]
    print(f"[INFO] => write_video_ffmpeg => {output_path}, fps={fps}, shape={frames_np.shape}")
    writer = imageio_ffmpeg.write_frames(
        output_path,
        (W, H),
        fps=fps,
        codec=codec,
        output_params=cmd_params
    )
    writer.send(None)
    for i in range(T):
        frame_c = np.ascontiguousarray(frames_np[i])
        writer.send(frame_c)
    writer.close()


#######################################################
# 5) process_single_video_in_chunks
#######################################################
GLOBAL_COLOR_REF = None

def process_single_video_in_chunks(
    pipeline,
    input_video_path,
    save_dir,
    frames_chunk=23,
    overlap=3,
    tile_num=2,
    color_match=True,
    threshold_mask=0.005,
    sbs_mode="FSBS",
    encoder="x264",
    origin_mode="2x2",
    left_video_path=None,
    mask_video_path=None,
    warp_video_path=None,
    orig_video_path=None,
    num_inference_steps=15,
    inpaint_output_mode="mp4",
    color_space="sRGB",
    fsbs_double_height=True,
    right_only_1080p=False,
    downscale_inpainting=True,
    dilation_mask=2,
    blur_mask=69,
    use_color_ref=False,
    partial_ref_frames=0
):
    """
    Generates the inpainted right view in chunks. Writes output
    as mp4 or EXR depending on inpaint_output_mode.

    If color_space="ACEScg", we convert sRGB->ACEScg before writing EXR.

    fsbs_double_height (bool):
      If True, for FSBS we double final height.

    right_only_1080p (bool):
      If True, we output ONLY the right eye, but scaled to the
      same dimension that would be used for HSBS or FSBS, 
      rather than forcibly 1080 in height.

    downscale_inpainting (bool):
      If True, warp+mask are downscaled half before pipeline call.

    dilation_mask (int):
      Number of dilation iterations for the mask.

    blur_mask (int):
      Size of the Gaussian blur kernel (must be an odd number).

    use_color_ref (bool):
      If True, tries to apply color reference from the previous chunk's
      final frame to the new chunk's frames.

    partial_ref_frames (int):
      If > 0 and use_color_ref=True, only apply color reference to
      the first N frames of the chunk. The rest frames remain unchanged.
    """

    global GLOBAL_COLOR_REF

    print(f"\n[INFO] => process_single_video_in_chunks => {input_video_path}, origin_mode={origin_mode}")
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_video_path))[0] + "_inpainting_results"

    out_temp_path = os.path.join(save_dir, f"{base_name}_temp.mp4")
    final_path = os.path.join(save_dir, f"{base_name}_{sbs_mode}_{encoder}.mp4")

    start_init = time.time()

    # ---------------------------------------
    # 1) Read frames (2x2 or triple)
    # ---------------------------------------
    if origin_mode == "2x2":
        vr = VideoReader(input_video_path, ctx=cpu(0))
        num_frames = len(vr)
        fps = vr.get_avg_fps()
        if num_frames == 0:
            print("[WARN] => No frames =>", input_video_path)
            return None

        first_fr = vr[0].asnumpy().astype(np.float32)/255.0
        H_in, W_in, _ = first_fr.shape
        half_h_in = H_in // 2
        half_w_in = W_in // 2
        if half_h_in < 2 or half_w_in < 2:
            print("[WARN] => Too small => skipping =>", input_video_path)
            return None

        half_h_aligned = (half_h_in // 128) * 128
        half_w_aligned = (half_w_in // 128) * 128

        left_orig_list = []
        mask_orig_list = []
        warp_orig_list = []

        for i in range(num_frames):
            fr = vr[i].asnumpy().astype(np.float32)/255.0
            top_left     = fr[:half_h_in, :half_w_in, :]
            bottom_left  = fr[half_h_in:, :half_w_in, :]
            bottom_right = fr[half_h_in:, half_w_in:, :]

            left_aligned = top_left[:half_h_aligned, :half_w_aligned, :]
            mask_aligned = bottom_left[:half_h_aligned, :half_w_aligned, :]
            warp_aligned = bottom_right[:half_h_aligned, :half_w_aligned, :]

            left_orig_list.append(left_aligned)
            mask_orig_list.append(mask_aligned)
            warp_orig_list.append(warp_aligned)

        total_frames = num_frames
        read_mode = "2x2"
        fps = max(fps, 1.0)

    else:  # triple
        vr_left = VideoReader(left_video_path, ctx=cpu(0))
        vr_mask = VideoReader(mask_video_path, ctx=cpu(0))
        vr_warp = VideoReader(warp_video_path, ctx=cpu(0))
        nL = len(vr_left)
        nM = len(vr_mask)
        nW = len(vr_warp)
        nF = min(nL, nM, nW)
        if nF == 0:
            print("[WARN] => triple => some video has 0 frames => skipping.")
            return None

        fps = vr_left.get_avg_fps()
        lf0 = vr_left[0].asnumpy().astype(np.float32)/255.0
        H_in, W_in, _ = lf0.shape

        half_h_aligned = (H_in // 128) * 128
        half_w_aligned = (W_in // 128) * 128

        left_orig_list = []
        mask_orig_list = []
        warp_orig_list = []
        for i in range(nF):
            lf = vr_left[i].asnumpy().astype(np.float32)/255.0
            mk = vr_mask[i].asnumpy().astype(np.float32)/255.0
            wp = vr_warp[i].asnumpy().astype(np.float32)/255.0

            lf_aligned = lf[:half_h_aligned, :half_w_aligned, :]
            mk_aligned = mk[:half_h_aligned, :half_w_aligned, :]
            wp_aligned = wp[:half_h_aligned, :half_w_aligned, :]

            left_orig_list.append(lf_aligned)
            mask_orig_list.append(mk_aligned)
            warp_orig_list.append(wp_aligned)

        total_frames = nF
        read_mode = "triple"
        fps = max(fps, 1.0)

    print(f"[INFO] => total_frames={total_frames}, fps={fps}, chunk={frames_chunk}, overlap={overlap}, tile_num={tile_num}")
    if total_frames < overlap:
        overlap = 0
    if total_frames < frames_chunk:
        frames_chunk = total_frames
    stride = max(1, frames_chunk - overlap)

    # We'll store final_w/final_h just for info
    final_w, final_h = None, None
    if origin_mode == "2x2":
        half_h_final = H_in // 2
        half_w_final = W_in // 2
        if sbs_mode.upper() == "FSBS":
            if fsbs_double_height:
                final_w = half_w_final * 2
                final_h = half_h_final * 2
            else:
                final_w = half_w_final * 2
                final_h = half_h_final
        else:
            final_w = half_w_final
            final_h = half_h_final
    else:
        if sbs_mode.upper() == "FSBS":
            final_w = W_in * 2
            final_h = H_in * 2 if fsbs_double_height else H_in
        else:
            final_w = W_in
            final_h = H_in

    print(f"[INFO] => final SBS resolution = {final_w}x{final_h}")
    final_frames = []

    generated_prev = None
    generated_prev_small = None

    dt_init = time.time() - start_init
    report_gpu_mem("after reading")

    start_idx = 0
    chunk_index = 0

    while start_idx < total_frames:
        chunk_start_time = time.time()
        end = min(start_idx + frames_chunk, total_frames)
        csize = end - start_idx
        if csize <= 0:
            break

        print(f"\n[CHUNK] => {start_idx}..{end} (size={csize}) chunk_index={chunk_index}")

        warp_chunk_np = warp_orig_list[start_idx:end]
        mask_chunk_np = mask_orig_list[start_idx:end]
        left_chunk_np = left_orig_list[start_idx:end]

        warp_t = torch.from_numpy(np.stack(warp_chunk_np, axis=0)).permute(0,3,1,2).float().cuda()
        mask_t = torch.from_numpy(np.stack(mask_chunk_np, axis=0)).permute(0,3,1,2).float().cuda()
        left_t = torch.from_numpy(np.stack(left_chunk_np, axis=0)).permute(0,3,1,2).float().cuda()

        mask_t = mask_t.mean(dim=1, keepdim=True)
        mask_np = mask_t.squeeze(1).cpu().numpy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        for i in range(mask_np.shape[0]):
            mask_uint8 = (mask_np[i] * 255).astype(np.uint8)
            mask_dil = cv2.dilate(mask_uint8, kernel, iterations=dilation_mask)
            mask_blurred = cv2.GaussianBlur(mask_dil, (blur_mask, blur_mask), 0)
            mask_np[i] = mask_blurred.astype(np.float32) / 255.0

        mask_t = torch.from_numpy(mask_np).unsqueeze(1).to(mask_t.device).float()

        # PARTIAL color reference => only apply to first partial_ref_frames if desired
        if use_color_ref and GLOBAL_COLOR_REF is not None and partial_ref_frames > 0:
            # We'll do color transfer from GLOBAL_COLOR_REF to the first 'partial_ref_frames' frames
            print(f"[INFO] => Using partial color reference on first {partial_ref_frames} frames...")

            global_ref_np = (GLOBAL_COLOR_REF.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype("uint8")
            warp_np = warp_t.permute(0,2,3,1).cpu().numpy()

            # Only up to the smaller of partial_ref_frames or the chunk size
            limit_frames = min(partial_ref_frames, warp_np.shape[0])

            for i_ in range(limit_frames):
                src_frame = (warp_np[i_]*255).clip(0,255).astype("uint8")
                matched_ = cm.transfer(src=src_frame, ref=global_ref_np, method='mkl')
                matched_ = np.clip(matched_, 0, 255).astype("uint8")
                warp_np[i_] = matched_.astype(np.float32)/255.0

            warp_t = torch.from_numpy(warp_np).permute(0,3,1,2).float().to(warp_t.device)
        elif use_color_ref and GLOBAL_COLOR_REF is not None:
            # If partial_ref_frames=0 => apply color to all frames
            print("[INFO] => Using color reference from previous chunk for all frames...")
            global_ref_np = (GLOBAL_COLOR_REF.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype("uint8")
            warp_np = warp_t.permute(0,2,3,1).cpu().numpy()
            for i_ in range(warp_np.shape[0]):
                src_frame = (warp_np[i_]*255).clip(0,255).astype("uint8")
                matched_ = cm.transfer(src=src_frame, ref=global_ref_np, method='mkl')
                matched_ = np.clip(matched_, 0, 255).astype("uint8")
                warp_np[i_] = matched_.astype(np.float32)/255.0
            warp_t = torch.from_numpy(warp_np).permute(0,3,1,2).float().to(warp_t.device)

        if downscale_inpainting:
            warp_t_small = F.interpolate(warp_t, scale_factor=0.5, mode="bilinear", align_corners=False)
            mask_t_small = F.interpolate(mask_t, scale_factor=0.5, mode="bilinear", align_corners=False)

            B_small, C_small, H_small, W_small = warp_t_small.shape
            H_small_aligned = (H_small // 8) * 8
            W_small_aligned = (W_small // 8) * 8
            if H_small_aligned < H_small or W_small_aligned < W_small:
                warp_t_small = warp_t_small[:, :, :H_small_aligned, :W_small_aligned]
                mask_t_small = mask_t_small[:, :, :H_small_aligned, :W_small_aligned]

            if generated_prev_small is not None and overlap > 0 and start_idx != 0:
                b1 = generated_prev_small.shape[0]
                b2 = warp_t_small.shape[0]
                ov_act = min(overlap, b1, b2)
                if ov_act > 0:
                    print(f"[CHUNK] => Overlap={ov_act} frames from previous chunk (downscaled)")
                    warp_t_small[:ov_act] = generated_prev_small[-ov_act:]

            report_gpu_mem("before spatial_tiled_process (downscaled)")
            t_spatial_start = time.time()

            with torch.no_grad():
                lat_ = spatial_tiled_process(
                    warp_t_small,
                    mask_t_small,
                    pipeline,
                    tile_num=tile_num,
                    spatial_n_compress=8,
                    min_guidance_scale=1.01,
                    max_guidance_scale=1.01,
                    decode_chunk_size=8,
                    fps=7,
                    motion_bucket_id=127,
                    noise_aug_strength=0.0,
                    num_inference_steps=num_inference_steps
                )

            dt_spatial = time.time() - t_spatial_start

            lat_ = lat_.unsqueeze(0)
            pipeline.vae.to(torch.float16)
            t_decode_start = time.time()
            dec = pipeline.decode_latents(lat_, num_frames=lat_.shape[1], decode_chunk_size=1)
            dec_frames = tensor2vid(dec, pipeline.image_processor, output_type="pil")[0]
            dt_decode = time.time() - t_decode_start

            out_right_list_small = []
            for pf in dec_frames:
                arr = np.array(pf, dtype=np.uint8)
                t_ = torch.from_numpy(arr).permute(2,0,1).float()/255.0
                out_right_list_small.append(t_.cuda())
            right_chunk_small = torch.stack(out_right_list_small, dim=0)

            del lat_, dec, dec_frames, out_right_list_small
            torch.cuda.empty_cache()
            report_gpu_mem("after decoding (downscaled)")
            gc.collect()

            if start_idx != 0 and overlap > 0 and right_chunk_small.shape[0] > overlap:
                right_chunk_small = right_chunk_small[overlap:]

            Tfin = min(
                right_chunk_small.shape[0],
                warp_t.shape[0],
                mask_t.shape[0],
                left_t.shape[0]
            )
            right_chunk_small = right_chunk_small[:Tfin]
            warp_t = warp_t[-Tfin:]
            mask_t = mask_t[-Tfin:]
            left_t = left_t[-Tfin:]

            print(f"[CHUNK] => Tfinal={Tfin}, upscaling inpainted + blending ...")

            if overlap > 0 and right_chunk_small.shape[0] >= overlap:
                generated_prev_small = right_chunk_small[-overlap:].clone()
            else:
                generated_prev_small = right_chunk_small.clone()

            right_chunk_up = []
            for i2 in range(Tfin):
                rc_small = right_chunk_small[i2].unsqueeze(0)
                H_orig = warp_t[i2].shape[1]
                W_orig = warp_t[i2].shape[2]

                rc_up = F.interpolate(
                    rc_small,
                    size=(H_orig, W_orig),
                    mode="bilinear",
                    align_corners=False
                )
                right_chunk_up.append(rc_up.squeeze(0))
            right_chunk = torch.stack(right_chunk_up, dim=0)

        else:
            if generated_prev is not None and overlap > 0 and start_idx != 0:
                b1 = generated_prev.shape[0]
                b2 = warp_t.shape[0]
                ov_act = min(overlap, b1, b2)
                if ov_act > 0:
                    print(f"[CHUNK] => Overlap={ov_act} frames from previous chunk")
                    warp_t[:ov_act] = generated_prev[-ov_act:]

            report_gpu_mem("before spatial_tiled_process")
            t_spatial_start = time.time()

            with torch.no_grad():
                lat_ = spatial_tiled_process(
                    warp_t,
                    mask_t,
                    pipeline,
                    tile_num=tile_num,
                    spatial_n_compress=8,
                    min_guidance_scale=1.01,
                    max_guidance_scale=1.01,
                    decode_chunk_size=8,
                    fps=7,
                    motion_bucket_id=127,
                    noise_aug_strength=0.0,
                    num_inference_steps=num_inference_steps
                )

            dt_spatial = time.time() - t_spatial_start

            lat_ = lat_.unsqueeze(0)
            pipeline.vae.to(torch.float16)
            t_decode_start = time.time()
            dec = pipeline.decode_latents(lat_, num_frames=lat_.shape[1], decode_chunk_size=1)
            dec_frames = tensor2vid(dec, pipeline.image_processor, output_type="pil")[0]
            dt_decode = time.time() - t_decode_start

            out_right_list = []
            for pf in dec_frames:
                arr = np.array(pf, dtype=np.uint8)
                t_ = torch.from_numpy(arr).permute(2,0,1).float()/255.0
                out_right_list.append(t_.cuda())
            right_chunk = torch.stack(out_right_list, dim=0)

            del lat_, dec, dec_frames, out_right_list
            torch.cuda.empty_cache()
            report_gpu_mem("after decoding")
            gc.collect()

            if start_idx != 0 and overlap > 0 and right_chunk.shape[0] > overlap:
                right_chunk = right_chunk[overlap:]

            Tfin = min(
                right_chunk.shape[0],
                warp_t.shape[0],
                mask_t.shape[0],
                left_t.shape[0]
            )
            right_chunk = right_chunk[:Tfin]
            warp_t = warp_t[-Tfin:]
            mask_t = mask_t[-Tfin:]
            left_t = left_t[-Tfin:]

            print(f"[CHUNK] => Tfinal={Tfin}, blending + recoloring both views...")

            if overlap > 0 and right_chunk.shape[0] >= overlap:
                generated_prev = right_chunk[-overlap:].clone()
            else:
                generated_prev = right_chunk.clone()

        for i2 in range(Tfin):
            inpainted = right_chunk[i2]
            original = warp_t[i2]
            mask_val = mask_t[i2][0].clamp(0, 1)
            
            alpha_linear = (mask_val - threshold_mask) / (1.0 - threshold_mask)
            alpha_linear = alpha_linear.clamp(0, 1)
            alpha_linear = alpha_linear.pow(0.5)
            alpha = alpha_linear.unsqueeze(0).repeat(3, 1, 1)
            
            inpainted_np = (inpainted.permute(1,2,0).cpu().numpy()*255.0).clip(0,255).astype("uint8")
            original_np  = (original.permute(1,2,0).cpu().numpy()*255.0).clip(0,255).astype("uint8")
            orig_recolored_np = cm.transfer(src=original_np, ref=inpainted_np, method='mkl')
            orig_recolored_np = Normalizer(orig_recolored_np).uint8_norm()
            orig_recolored_t = (
                torch.from_numpy(orig_recolored_np)
                .float()
                .permute(2,0,1)
                .to(original.device) / 255.0
            )
            final_right = inpainted * alpha + orig_recolored_t * (1 - alpha)
            final_right = local_color_match_masked_region(final_right, left_t[i2], mask_t[i2], cm)
            right_chunk[i2] = final_right.clamp(0,1)

        # 3) Upscale right if needed
        up_right = []
        for i2 in range(Tfin):
            rc = right_chunk[i2].unsqueeze(0)
            if origin_mode == "2x2":
                if sbs_mode.upper() == "HSBS":
                    rc_up = F.interpolate(
                        rc,
                        size=(half_h_in, half_w_in // 2),
                        mode="bilinear",
                        align_corners=False
                    )
                elif sbs_mode.upper() == "FSBS":
                    if fsbs_double_height:
                        rc_up = F.interpolate(
                            rc,
                            size=(2 * half_h_in, half_w_in),
                            mode="bilinear",
                            align_corners=False
                        )
                    else:
                        rc_up = F.interpolate(
                            rc,
                            size=(half_h_in, half_w_in),
                            mode="bilinear",
                            align_corners=False
                        )
                else:
                    rc_up = F.interpolate(
                        rc,
                        size=(half_h_in, half_w_in // 2),
                        mode="bilinear",
                        align_corners=False
                    )
            else:
                if sbs_mode.upper() == "HSBS":
                    rc_up = F.interpolate(rc, size=(H_in, W_in//2), mode="bilinear", align_corners=False)
                elif sbs_mode.upper() == "FSBS":
                    if fsbs_double_height:
                        rc_up = F.interpolate(rc, size=(H_in*2, W_in), mode="bilinear", align_corners=False)
                    else:
                        rc_up = F.interpolate(rc, size=(H_in, W_in), mode="bilinear", align_corners=False)
                else:
                    rc_up = F.interpolate(rc, size=(H_in, W_in//2), mode="bilinear", align_corners=False)
            up_right.append(rc_up.squeeze(0))

        up_right_t = torch.stack(up_right, dim=0)

        if right_only_1080p:
            final_right_only_list = []
            for i2 in range(Tfin):
                rc_ = up_right_t[i2].unsqueeze(0).clamp(0,1)
                rc_upsized = F.interpolate(rc_, scale_factor=(1,2), mode="bilinear", align_corners=False)
                rc_cpu = rc_upsized.squeeze(0).detach().cpu()
                rc_uint8 = (rc_cpu*255).byte().permute(1,2,0).numpy()
                final_right_only_list.append(rc_uint8)
            final_frames.extend(final_right_only_list)

        else:
            for i2 in range(Tfin):
                lf_1 = left_t[i2].unsqueeze(0)
                if origin_mode == "2x2":
                    if sbs_mode.upper() == "HSBS":
                        lf_up = F.interpolate(
                            lf_1,
                            size=(half_h_in, half_w_in // 2),
                            mode="bilinear",
                            align_corners=False
                        )
                    elif sbs_mode.upper() == "FSBS":
                        if fsbs_double_height:
                            lf_up = F.interpolate(
                                lf_1,
                                size=(2 * half_h_in, half_w_in),
                                mode="bilinear",
                                align_corners=False
                            )
                        else:
                            lf_up = F.interpolate(
                                lf_1,
                                size=(half_h_in, half_w_in),
                                mode="bilinear",
                                align_corners=False
                            )
                    else:
                        lf_up = F.interpolate(
                            lf_1,
                            size=(half_h_in, half_w_in // 2),
                            mode="bilinear",
                            align_corners=False
                        )
                else:
                    if sbs_mode.upper() == "HSBS":
                        lf_up = F.interpolate(lf_1, size=(H_in, W_in//2), mode="bilinear", align_corners=False)
                    elif sbs_mode.upper() == "FSBS":
                        if fsbs_double_height:
                            lf_up = F.interpolate(lf_1, size=(H_in*2, W_in), mode="bilinear", align_corners=False)
                        else:
                            lf_up = F.interpolate(lf_1, size=(H_in, W_in), mode="bilinear", align_corners=False)
                    else:
                        lf_up = F.interpolate(lf_1, size=(H_in, W_in//2), mode="bilinear", align_corners=False)

                lf_up = lf_up.squeeze(0)
                rf_t = up_right_t[i2]
                sbs = torch.cat([lf_up, rf_t], dim=2)

                sbs_cpu = sbs.detach().cpu().clamp(0,1)
                sbs_uint8 = (sbs_cpu*255).byte().permute(1,2,0).numpy()
                final_frames.append(sbs_uint8)

        # Optionally store the last frame's color reference
        if use_color_ref and len(up_right_t) > 0:
            last_ = up_right_t[-1].clone()
            GLOBAL_COLOR_REF = last_

        del warp_t, mask_t, left_t, right_chunk, up_right_t, up_right
        torch.cuda.empty_cache()
        gc.collect()

        chunk_dt = time.time() - chunk_start_time
        print(f"[CHUNK] => Finished chunk in {chunk_dt:.2f}s")
        report_gpu_mem("after chunk")

        start_idx += stride
        chunk_index += 1

    # ---------------------------------------
    # 3) Write final (MP4 or EXR)
    # ---------------------------------------
    if not final_frames:
        print("[WARN] => No frames => skipping =>", out_temp_path)
        return None

    if inpaint_output_mode == "mp4":
        print(f"[INFO] => Writing MP4 => {out_temp_path}")
        frames_np = np.stack(final_frames, axis=0)
        frames_np = np.ascontiguousarray(frames_np)
        crf = 0 if encoder=="x264" else 0
        cdc = "libx265" if encoder=="x265" else "libx264"
        cmd_params = ["-crf", str(crf), "-preset", "veryslow"]

        writer = imageio_ffmpeg.write_frames(
            out_temp_path,
            (frames_np.shape[2], frames_np.shape[1]),
            fps=fps,
            codec=cdc,
            output_params=cmd_params
        )
        writer.send(None)
        for i in range(frames_np.shape[0]):
            writer.send(frames_np[i])
        writer.close()

        final_frames.clear()

        if not os.path.exists(out_temp_path):
            print("[WARN] => No temp file => no final =>", out_temp_path)
            return None

        final_path = os.path.join(save_dir, f"{base_name}_{sbs_mode}_{encoder}.mp4")
        if orig_video_path and os.path.isfile(orig_video_path):
            print(f"[AUDIO] => Found orig => {orig_video_path}, muxing ...")
            temp_withaudio = os.path.join(save_dir, f"{base_name}_temp_withaudio.mp4")
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            cmd_mux = [
                ffmpeg_path, "-y",
                "-i", out_temp_path,
                "-i", orig_video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                temp_withaudio
            ]
            print("[AUDIO] =>", " ".join(cmd_mux))
            try:
                subprocess.run(cmd_mux, check=True)
                if os.path.exists(temp_withaudio):
                    shutil.move(temp_withaudio, final_path)
                    time.sleep(0.5)
                    os.remove(out_temp_path)
                    print("[AUDIO] => done =>", final_path)
                    return final_path
                else:
                    print("[AUDIO] => fallback => no temp_withaudio => no audio.")
                    shutil.move(out_temp_path, final_path)
                    return final_path
            except subprocess.CalledProcessError as e:
                print("[AUDIO] => error =>", e)
                shutil.move(out_temp_path, final_path)
                return final_path
        else:
            shutil.move(out_temp_path, final_path)
            return final_path

    else:
        print("[INFO] => Writing EXR =>", inpaint_output_mode)
        frames_np = np.stack(final_frames, axis=0).astype(np.float32)/255.0
        final_frames.clear()

        if color_space == "ACEScg":
            frames_np = srgb_to_acescg(frames_np)

        out_exr_dir = os.path.join(save_dir, f"{base_name}_exrseq_{sbs_mode}")
        half_f = (inpaint_output_mode=="exr16")
        save_exr_sequence_color(frames_np, out_exr_dir, half_float=half_f)

        print(f"[OK] => EXR seq => {out_exr_dir}")
        return out_exr_dir


#######################################################
# 6) BATCH => auto-detect 2x2 or triple
#######################################################
def batch_process(
    pre_trained_path="./weights/stable-video-diffusion-img2vid-xt-1-1",
    unet_path="./weights/StereoCrafter",
    input_folder="./output_splatted",
    output_folder="./completed_output",
    frames_chunk=23,
    overlap=3,
    tile_num=2,
    color_match=True,
    concat_final=False,
    single_video=None,
    threshold_mask=0.008,
    sbs_mode="FSBS",
    encoder="x264",
    origin_mode="",
    inpaint_output_mode="mp4",
    color_space="sRGB", 
    left_video_path=None,
    mask_video_path=None,
    warp_video_path=None,
    orig_video=None,
    num_inference_steps=10,
    fsbs_double_height=True,
    right_only_1080p=False,
    downscale_inpainting=True,
    dilation_mask=2,
    blur_mask=69,
    use_color_ref=False,
    partial_ref_frames=0
):
    """
    Main batch process entry point.
      - use_color_ref (bool): If True, pass color reference from chunk to chunk
      - partial_ref_frames (int): If > 0, only apply that color ref to the first N frames
        of each chunk. The rest remain as is, which can help fade transitions.
    """

    print("[INFO] => batch_process => Loading pipeline...")

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=torch.float16
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path,
        subfolder="vae",
        variant="fp16",
        torch_dtype=torch.float16
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        pre_trained_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch.float16
    ).to("cuda")

    os.makedirs(output_folder, exist_ok=True)

    if single_video and os.path.isfile(single_video):
        single_video = os.path.normpath(os.path.abspath(single_video))
        input_videos = [single_video]
        print("[INFO] => single_video =>", single_video)
    else:
        exts = ("*.mp4", "*.mov", "*.avi", "*.mkv")
        vids = []
        for e_ in exts:
            vids.extend(glob.glob(os.path.join(input_folder, e_)))
        input_videos = sorted(vids)
        if not input_videos:
            print("[INFO] => No videos found in =>", input_folder)
            return

    if not origin_mode:
        if single_video:
            base_ = os.path.basename(single_video)
            name_noext, ext_ = os.path.splitext(base_)
            folder_ = os.path.dirname(single_video)
            if "_splatted" in name_noext:
                base_candidate = name_noext.replace("_splatted","")
                found_left = os.path.join(folder_, base_candidate+"_left.mp4")
                found_mask = os.path.join(folder_, base_candidate+"_mask.mp4")
                print("[INFO] => Checking for left:", found_left)
                print("[INFO] => Checking for mask:", found_mask)
                if os.path.isfile(found_left) and os.path.isfile(found_mask):
                    origin_mode = "triple"
                    left_video_path = found_left
                    mask_video_path = found_mask
                    warp_video_path = single_video
                    print("[INFO] => Using triple => warp =>", single_video)
                else:
                    origin_mode = "2x2"
                    print("[INFO] => Using 2x2 => no separate left/mask found")
            else:
                origin_mode = "2x2"
                print("[INFO] => No '_splatted' => using 2x2 by default")
        else:
            origin_mode = "2x2"
            print("[INFO] => No single_video => using 2x2 by default")

    print("[INFO] => origin_mode=", origin_mode)

    processed = []
    if origin_mode == "triple" and single_video:
        outp = process_single_video_in_chunks(
            pipeline,
            single_video,
            output_folder,
            frames_chunk=frames_chunk,
            overlap=overlap,
            tile_num=tile_num,
            color_match=color_match,
            threshold_mask=threshold_mask,
            inpaint_output_mode=inpaint_output_mode,
            sbs_mode=sbs_mode,
            color_space=color_space,
            encoder=encoder,
            origin_mode="triple",
            left_video_path=left_video_path,
            mask_video_path=mask_video_path,
            warp_video_path=warp_video_path,
            orig_video_path=orig_video,
            num_inference_steps=num_inference_steps,
            fsbs_double_height=fsbs_double_height,
            right_only_1080p=right_only_1080p,
            downscale_inpainting=downscale_inpainting,
            dilation_mask=dilation_mask,
            blur_mask=blur_mask,
            use_color_ref=use_color_ref,
            partial_ref_frames=partial_ref_frames
        )
        if outp:
            processed.append(outp)
    elif origin_mode == "triple" and not single_video:
        print("[WARN] => origin_mode=triple => no single_video => not implemented in this example.")
    else:
        for vid in input_videos:
            print(f"\n[INFO] => Processing => {vid}")
            outp = process_single_video_in_chunks(
                pipeline,
                vid,
                save_dir=output_folder,
                frames_chunk=frames_chunk,
                overlap=overlap,
                tile_num=tile_num,
                color_match=color_match,
                threshold_mask=threshold_mask,
                sbs_mode=sbs_mode,
                inpaint_output_mode=inpaint_output_mode,
                color_space=color_space,
                encoder=encoder,
                origin_mode="2x2",
                orig_video_path=orig_video,
                num_inference_steps=num_inference_steps,
                fsbs_double_height=fsbs_double_height,
                right_only_1080p=right_only_1080p,
                downscale_inpainting=downscale_inpainting,
                dilation_mask=dilation_mask,
                blur_mask=blur_mask,
                use_color_ref=use_color_ref,
                partial_ref_frames=partial_ref_frames
            )
            if outp:
                processed.append(outp)
            torch.cuda.empty_cache()
            gc.collect()

    if concat_final and processed:
        merged_name = os.path.join(output_folder, "final_merged.mp4")
        list_txt = os.path.join(output_folder, "mylist.txt")
        with open(list_txt, "w", encoding="utf-8") as ff:
            for p_ in processed:
                ff.write(f"file '{os.path.abspath(p_)}'\n")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            "ffmpeg_exe", "-y", "-f", "concat", "-safe", "0",
            "-i", list_txt,
            "-c", "copy",
            merged_name
        ]
        print("[INFO] => Concatenating =>", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print("[INFO] => Final merged =>", merged_name)

    pipeline.unet.to(torch.float32)
    pipeline.vae.to(torch.float32)
    pipeline.image_encoder.to(torch.float32)
    pipeline.to("cpu")
    del pipeline, unet, vae, image_encoder
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    Fire(batch_process)
