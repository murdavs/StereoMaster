import os
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
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
import OpenEXR
import Imath
import array
import math

def report_gpu_mem(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved  = torch.cuda.memory_reserved() / 1024**2
    print(f"[INFO] GPU usage {prefix} => allocated={allocated:.2f}MB, reserved={reserved:.2f}MB")

def find_best_match_region(template_np, search_np):
    tmpl_gray = cv2.cvtColor(template_np, cv2.COLOR_BGR2GRAY)
    srch_gray = cv2.cvtColor(search_np, cv2.COLOR_BGR2GRAY)
    if tmpl_gray.shape[0] > srch_gray.shape[0] or tmpl_gray.shape[1] > srch_gray.shape[1]:
        print("[WARN] The template is bigger than the search image, returning (0,0)")
        return 0, 0
    method = cv2.TM_CCOEFF_NORMED
    res = cv2.matchTemplate(srch_gray, tmpl_gray, method)
    _, _, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    return (top_left[1], top_left[0])

def local_color_match_masked_region(final_right, left_frame, mask, cm):
    mask_indices = (mask[0] > 0).nonzero(as_tuple=True)
    if len(mask_indices[0]) == 0:
        return final_right
    y_min, y_max = mask_indices[0].min(), mask_indices[0].max()
    x_min, x_max = mask_indices[1].min(), mask_indices[1].max()
    region_crop = final_right[:, y_min:y_max+1, x_min:x_max+1].clone()
    region_crop_np = (region_crop.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype("uint8")
    left_frame_np = (left_frame.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype("uint8")
    top_y, left_x = find_best_match_region(region_crop_np, left_frame_np)
    Hr, Wr, _ = region_crop_np.shape
    left_sub = left_frame_np[top_y:top_y+Hr, left_x:left_x+Wr, :].copy()
    if left_sub.shape[0] != Hr or left_sub.shape[1] != Wr:
        print("[WARN] The found sub-region size does not match, skipping color match")
        return final_right
    matched_np = cm.transfer(src=region_crop_np, ref=left_sub, method='mkl')
    matched_np = np.clip(matched_np, 0, 255).astype("uint8")
    matched_t = (torch.from_numpy(matched_np).float().permute(2,0,1)/255).to(final_right.device)
    out_right = final_right.clone()
    out_right[:, y_min:y_max+1, x_min:x_max+1] = matched_t
    return out_right

def blend_h(a, b, overlap_size):
    print(f"[INFO] Horizontal blending with overlap={overlap_size}")
    weight_b = (torch.arange(overlap_size).view(1,1,1,-1)/overlap_size).to(b.device)
    b[:,:,:, :overlap_size] = (1 - weight_b)*a[:,:,:, -overlap_size:] + weight_b*b[:,:,:, :overlap_size]
    return b

def blend_v(a, b, overlap_size):
    print(f"[INFO] Vertical blending with overlap={overlap_size}")
    weight_b = (torch.arange(overlap_size).view(1,1,-1,1)/overlap_size).to(b.device)
    b[:,:, :overlap_size,:] = (1 - weight_b)*a[:,:, -overlap_size:,:] + weight_b*b[:,:, :overlap_size,:]
    return b

def pad_for_tiling(frames, tile_num, tile_overlap=(128,128)):
    if tile_num <= 1:
        return frames
    T,C,H,W = frames.shape
    ov_y, ov_x = tile_overlap
    tile_size_y = (H + ov_y*(tile_num-1))//tile_num
    tile_size_x = (W + ov_x*(tile_num-1))//tile_num
    stride_y = tile_size_y - ov_y
    stride_x = tile_size_x - ov_x
    ideal_h = stride_y*tile_num + ov_y*(tile_num-1)
    ideal_w = stride_y*tile_num + ov_x*(tile_num-1)
    pad_bottom = max(0, ideal_h - H)
    pad_right = max(0, ideal_w - W)
    if pad_bottom>0 or pad_right>0:
        frames = F.pad(frames,(0,pad_right,0,pad_bottom),mode="constant",value=0)
    return frames

def spatial_tiled_process(cond_frames, mask_frames, pipeline_func, tile_num, spatial_n_compress=8, **kargs):
    print("[INFO] Starting spatial tiled process")
    start_t = time.time()
    height = cond_frames.shape[2]
    width = cond_frames.shape[3]
    tile_overlap = (128,128)
    tile_size = (
        int((height + tile_overlap[0]*(tile_num-1))/tile_num),
        int((width + tile_overlap[1]*(tile_num-1))/tile_num)
    )
    tile_stride = (
        tile_size[0] - tile_overlap[0],
        tile_size[1] - tile_overlap[1]
    )
    row_tiles = []
    for i in range(tile_num):
        col_tiles = []
        for j in range(tile_num):
            cond_tile = cond_frames[:,:,
                i*tile_stride[0]:i*tile_stride[0]+tile_size[0],
                j*tile_stride[1]:j*tile_stride[1]+tile_size[1]
            ]
            mask_tile = mask_frames[:,:,
                i*tile_stride[0]:i*tile_stride[0]+tile_size[0],
                j*tile_stride[1]:j*tile_stride[1]+tile_size[1]
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
            print(f"[INFO] Tile finished in {dt_tile:.2f}s")
            tile_latent = out.frames[0]
            col_tiles.append(tile_latent)
        row_tiles.append(col_tiles)
    latent_stride = (
        tile_stride[0]//spatial_n_compress,
        tile_stride[1]//spatial_n_compress
    )
    latent_overlap = (
        tile_overlap[0]//spatial_n_compress,
        tile_overlap[1]//spatial_n_compress
    )
    for i in range(tile_num):
        for j in range(tile_num):
            tile = row_tiles[i][j]
            if i>0:
                tile = blend_v(row_tiles[i-1][j], tile, latent_overlap[0])
            if j>0:
                tile = blend_h(row_tiles[i][j-1], tile, latent_overlap[1])
            row_tiles[i][j] = tile
    for i in range(tile_num):
        line_ = row_tiles[i]
        for j in range(tile_num):
            tile = line_[j]
            if i<tile_num-1:
                tile = tile[:,:,:latent_stride[0],:]
            if j<tile_num-1:
                tile = tile[:,:,:,:latent_stride[1]]
            line_[j] = tile
        row_cat = torch.cat(line_,dim=3)
        row_tiles[i] = row_cat
    final_latent = torch.cat(row_tiles,dim=2)
    dt_spatial = time.time() - start_t
    print(f"[INFO] Spatial tiled process completed in {dt_spatial:.2f}s")
    return final_latent

def save_exr_sequence_color(frames_np, out_dir, half_float=False):
    print("[INFO] Saving EXR sequence")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    T,H,W,C = frames_np.shape
    if C!=3:
        print("[WARN] Not 3 channels, skipping EXR saving")
        return
    for i in range(T):
        frame = frames_np[i]
        exr_flags = []
        if half_float:
            exr_flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]
        else:
            exr_flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
        out_exr = os.path.join(out_dir,f"frame_{i:04d}.exr")
        success = cv2.imwrite(out_exr, frame, exr_flags)
        if not success:
            print(f"[ERROR] => Could not write EXR => {out_exr}")
        else:
            print(f"[INFO] Wrote {out_exr}")

def float32_to_half_bytes(frame_2d):
    H,W = frame_2d.shape
    arr_ = []
    for val in frame_2d.flatten():
        half_ = Imath.FloatToHalf(val)
        arr_.append(half_)
    return array.array('H', arr_).tobytes()

def srgb_to_acescg(frames_np):
    print("[INFO] Converting sRGB to ACEScg")
    frames_lin = srgb_to_linear(frames_np)
    SRGB_TO_ACES = np.array([
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777]
    ], dtype=np.float32)
    out_aces = np.einsum("...rc,cd->...rd", frames_lin, SRGB_TO_ACES)
    return out_aces

def srgb_to_linear(u):
    mask = (u<=0.04045)
    u_out = np.empty_like(u)
    u_out[mask] = (u[mask]/12.92)
    u_out[~mask] = ((u[~mask]+0.055)/1.055)**2.4
    return u_out

def write_video_ffmpeg(frames_list, fps, output_path, codec="libx264", crf=0, preset="veryslow"):
    if not frames_list:
        print("[WARN] => No frames => skipping =>", output_path)
        return
    print(f"[INFO] Writing video => {output_path}")
    frames_np = np.stack(frames_list, axis=0)
    if frames_np.dtype != np.uint8:
        frames_np = np.clip(frames_np, 0, 255).astype(np.uint8)
    T,H,W,C = frames_np.shape
    if C!=3:
        print("[WARN] => Not 3 channels => no video written")
        return
    import imageio_ffmpeg
    cmd_params = ["-crf", str(crf), "-preset", preset]
    writer = imageio_ffmpeg.write_frames(output_path, (W,H), fps=fps, codec=codec, output_params=cmd_params)
    writer.send(None)
    for i in range(T):
        frame_c = np.ascontiguousarray(frames_np[i])
        writer.send(frame_c)
    writer.close()

GLOBAL_COLOR_REF = None

def remove_padding_tensor(tensor_3d, top_pad, bottom_pad, left_pad, right_pad):
    C, H, W = tensor_3d.shape
    if bottom_pad == 0:
        h_end = H
    else:
        h_end = -bottom_pad
    if right_pad == 0:
        w_end = W
    else:
        w_end = -right_pad
    out = tensor_3d[:, top_pad:h_end, left_pad:w_end]
    return out

def fix_overlap_before_inference(total_frames, frames_chunk, overlap):
    print(f"[INFO] Checking overlap => total_frames={total_frames}, frames_chunk={frames_chunk}, overlap={overlap}")
    if total_frames <= frames_chunk:
        frames_chunk = total_frames
        overlap = 0
        print(f"[WARN] Video has fewer (or equal) frames ({total_frames}) than chunk size ({frames_chunk}). Forcing overlap=0.")
        return overlap
    original_overlap = overlap
    changed = False
    while True:
        stride = frames_chunk - overlap
        if stride < 1:
            overlap = max(0, frames_chunk - 1)
            changed = True
            break
        N = (total_frames - 1) // stride + 1
        c = total_frames - (N - 1) * stride
        if c <= overlap:
            overlap += 1
            changed = True
            if overlap >= frames_chunk:
                overlap = frames_chunk - 1
                break
        else:
            break
    if overlap < 0:
        overlap = 0
    if changed:
        print(f"[WARN] Overlap changed from {original_overlap} to {overlap}.")
    else:
        print(f"[INFO] Overlap remains {overlap}; no change needed.")
    return overlap

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
    partial_ref_frames=0,
    inpaint=True
):
    global GLOBAL_COLOR_REF
    os.makedirs(save_dir,exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_video_path))[0] + "_inpainting_results"
    out_temp_path = os.path.join(save_dir, base_name + "_temp.mp4")
    final_path = os.path.join(save_dir, base_name + "_" + sbs_mode + "_" + encoder + ".mp4")
    if origin_mode=="2x2":
        vr = VideoReader(input_video_path, ctx=cpu(0))
        num_frames = len(vr)
        fps = vr.get_avg_fps()
        if num_frames==0:
            return None
        print(f"[INFO] Number of frames: {num_frames}, Average FPS: {fps}")
        first_fr = vr[0].asnumpy().astype(np.float32)/255
        H_in, W_in, _ = first_fr.shape
        half_h_in = H_in//2
        half_w_in = W_in//2
        pH = math.ceil(half_h_in/128)*128
        pW = math.ceil(half_w_in/128)*128
        extraH = pH - half_h_in
        extraW = pW - half_w_in
        tH = extraH//2
        bH = extraH - tH
        lW = extraW//2
        rW = extraW - lW
        left_orig_list = []
        mask_orig_list = []
        warp_orig_list = []
        for i in range(num_frames):
            fr = vr[i].asnumpy().astype(np.float32)/255
            top_left = fr[:half_h_in, :half_w_in, :]
            bottom_left = fr[half_h_in:, :half_w_in, :]
            bottom_right = fr[half_h_in:, half_w_in:, :]
            top_left_padded = np.pad(top_left, ((tH,bH), (lW,rW), (0,0)), mode='constant', constant_values=0)
            bottom_left_padded = np.pad(bottom_left, ((tH,bH), (lW,rW), (0,0)), mode='constant', constant_values=0)
            bottom_right_padded = np.pad(bottom_right, ((tH,bH), (lW,rW), (0,0)), mode='constant', constant_values=0)
            left_orig_list.append(top_left_padded)
            mask_orig_list.append(bottom_left_padded)
            warp_orig_list.append(bottom_right_padded)
        total_frames = num_frames
        fps = max(fps,1)
    else:
        print(f"[INFO] origin_mode=triple => reading separate videos => left={left_video_path}, mask={mask_video_path}, warp={warp_video_path}")
        vr_left = VideoReader(left_video_path, ctx=cpu(0))
        vr_mask = VideoReader(mask_video_path, ctx=cpu(0))
        vr_warp = VideoReader(warp_video_path, ctx=cpu(0))
        nL = len(vr_left)
        nM = len(vr_mask)
        nW = len(vr_warp)
        nF = min(nL, nM, nW)
        if nF==0:
            return None
        print(f"[INFO] Left frames={nL}, Mask frames={nM}, Warp frames={nW}, using total={nF}")
        fps = vr_left.get_avg_fps()
        lf0 = vr_left[0].asnumpy().astype(np.float32)/255
        H_in, W_in, _ = lf0.shape
        pH = math.ceil(H_in/128)*128
        pW = math.ceil(W_in/128)*128
        exH = pH - H_in
        exW = pW - W_in
        tH = exH//2
        bH = exH - tH
        lW = exW//2
        rW = exW - lW
        left_orig_list = []
        mask_orig_list = []
        warp_orig_list = []
        for i in range(nF):
            lf = vr_left[i].asnumpy().astype(np.float32)/255
            mk = vr_mask[i].asnumpy().astype(np.float32)/255
            wp = vr_warp[i].asnumpy().astype(np.float32)/255
            lf_padded = np.pad(lf, ((tH,bH), (lW,rW), (0,0)), mode='constant', constant_values=0)
            mk_padded = np.pad(mk, ((tH,bH), (lW,rW), (0,0)), mode='constant', constant_values=0)
            wp_padded = np.pad(wp, ((tH,bH), (lW,rW), (0,0)), mode='constant', constant_values=0)
            left_orig_list.append(lf_padded)
            mask_orig_list.append(mk_padded)
            warp_orig_list.append(wp_padded)
        total_frames = nF
        fps = max(fps,1)
    print(f"[INFO] total_frames={total_frames}, fps={fps}, frames_chunk={frames_chunk}, overlap={overlap}, tile_num={tile_num}")
    if not inpaint:
        warp_t = torch.from_numpy(np.stack(warp_orig_list, axis=0)).permute(0,3,1,2).float().cuda()
        left_t = torch.from_numpy(np.stack(left_orig_list, axis=0)).permute(0,3,1,2).float().cuda()
        Tfin = warp_t.shape[0]
        if origin_mode=="2x2":
            first_fr = warp_orig_list[0]
            H_in2, W_in2, _ = first_fr.shape
            half_h_in2 = H_in2//2
            half_w_in2 = W_in2//2
            if sbs_mode.upper()=="HSBS":
                final_w = half_w_in2
                final_h = half_h_in2
            elif sbs_mode.upper()=="FSBS":
                if fsbs_double_height:
                    final_w = half_w_in2*2
                    final_h = half_h_in2*2
                else:
                    final_w = half_w_in2*2
                    final_h = half_h_in2
            else:
                final_w = half_w_in2*2
                final_h = half_h_in2
        else:
            if sbs_mode.upper()=="HSBS":
                final_w = W_in
                final_h = H_in
            elif sbs_mode.upper()=="FSBS":
                if fsbs_double_height:
                    final_w = W_in*2
                    final_h = H_in*2
                else:
                    final_w = W_in*2
                    final_h = H_in
            else:
                final_w = W_in
                final_h = H_in
        final_frames = []
        for i2 in range(Tfin):
            lf_nopad = remove_padding_tensor(left_t[i2], 0, 0, 0, 0).unsqueeze(0).clamp(0,1)
            wr_nopad = remove_padding_tensor(warp_t[i2], 0, 0, 0, 0).unsqueeze(0).clamp(0,1)
            if origin_mode=="2x2":
                hf = lf_nopad.shape[2]
                wf = lf_nopad.shape[3]
                half_h_in3 = hf
                half_w_in3 = wf
                if sbs_mode.upper()=="HSBS":
                    lf_up = F.interpolate(lf_nopad, size=(half_h_in3, half_w_in3//2), align_corners=False, mode="bilinear")
                    wr_up = F.interpolate(wr_nopad, size=(half_h_in3, half_w_in3//2), align_corners=False, mode="bilinear")
                elif sbs_mode.upper()=="FSBS":
                    if fsbs_double_height:
                        lf_up = F.interpolate(lf_nopad, size=(half_h_in3*2, half_w_in3), align_corners=False, mode="bilinear")
                        wr_up = F.interpolate(wr_nopad, size=(half_h_in3*2, half_w_in3), align_corners=False, mode="bilinear")
                    else:
                        lf_up = F.interpolate(lf_nopad, size=(half_h_in3, half_w_in3), align_corners=False, mode="bilinear")
                        wr_up = F.interpolate(wr_nopad, size=(half_h_in3, half_w_in3), align_corners=False, mode="bilinear")
                else:
                    lf_up = F.interpolate(lf_nopad, size=(half_h_in3, half_w_in3), align_corners=False, mode="bilinear")
                    wr_up = F.interpolate(wr_nopad, size=(half_h_in3, half_w_in3), align_corners=False, mode="bilinear")
            else:
                if sbs_mode.upper()=="HSBS":
                    lf_up = F.interpolate(lf_nopad, size=(H_in, W_in//2), align_corners=False, mode="bilinear")
                    wr_up = F.interpolate(wr_nopad, size=(H_in, W_in//2), align_corners=False, mode="bilinear")
                elif sbs_mode.upper()=="FSBS":
                    if fsbs_double_height:
                        lf_up = F.interpolate(lf_nopad, size=(H_in*2, W_in), align_corners=False, mode="bilinear")
                        wr_up = F.interpolate(wr_nopad, size=(H_in*2, W_in), align_corners=False, mode="bilinear")
                    else:
                        lf_up = F.interpolate(lf_nopad, size=(H_in, W_in), align_corners=False, mode="bilinear")
                        wr_up = F.interpolate(wr_nopad, size=(H_in, W_in), align_corners=False, mode="bilinear")
                else:
                    lf_up = F.interpolate(lf_nopad, size=(H_in, W_in//2), align_corners=False, mode="bilinear")
                    wr_up = F.interpolate(wr_nopad, size=(H_in, W_in//2), align_corners=False, mode="bilinear")
            if right_only_1080p:
                rc_upsized = F.interpolate(wr_up, scale_factor=(1,2), align_corners=False, mode="bilinear")
                rc_cpu = rc_upsized.squeeze(0).detach().cpu()
                rc_uint8 = (rc_cpu*255).byte().permute(1,2,0).numpy()
                final_frames.append(rc_uint8)
            else:
                lf_up = lf_up.squeeze(0)
                wr_up = wr_up.squeeze(0)
                sbs = torch.cat([lf_up, wr_up], dim=2)
                sbs_cpu = sbs.detach().cpu().clamp(0,1)
                sbs_uint8 = (sbs_cpu*255).byte().permute(1,2,0).numpy()
                final_frames.append(sbs_uint8)
        frames_np = np.stack(final_frames, axis=0)
        if inpaint_output_mode=="mp4":
            if encoder == "x264":
                cdc = "libx264"
                crf = 0
                cmd_params = [
                    "-crf", str(crf),
                    "-preset", "slow",
                    "-profile:v", "high10",
                    "-colorspace", "bt709",
                    "-color_primaries", "bt709",
                    "-color_trc", "bt709",
                    "-color_range", "tv",
                    "-tag:v", "avc1",
                    "-movflags", "+faststart+write_colr"
                ]
            elif encoder == "x265":
                cdc = "libx265"
                crf = 12
                cmd_params = [
                    "-crf", str(crf),
                    "-preset", "slow",
                    "-x265-params", "high-tier=1:colormatrix=bt709:colorprim=bt709:transfer=bt709:range=limited",
                    "-colorspace", "bt709",
                    "-color_primaries", "bt709",
                    "-color_trc", "bt709",
                    "-color_range", "tv",
                    "-tag:v", "hvc1",
                    "-movflags", "+faststart+write_colr"
                ]
            else:
                cdc = "libx264"
                crf = 0
                cmd_params = [
                    "-crf", str(crf),
                    "-preset", "slow",
                    "-profile:v", "high10",
                    "-colorspace", "bt709",
                    "-color_primaries", "bt709",
                    "-color_trc", "bt709",
                    "-color_range", "tv",
                    "-tag:v", "avc1",
                    "-movflags", "+faststart+write_colr"
                ]
            writer = imageio_ffmpeg.write_frames(
                out_temp_path,
                (frames_np.shape[2], frames_np.shape[1]),
                fps=fps,
                codec=cdc,
                output_params=cmd_params
            )
            writer.send(None)
            for i in range(frames_np.shape[0]):
                frame_c = np.ascontiguousarray(frames_np[i])
                writer.send(frame_c)
            writer.close()
            if not os.path.exists(out_temp_path):
                return None
            final_path = os.path.join(save_dir, base_name + "_" + sbs_mode + "_" + encoder + ".mp4")
            if orig_video_path and os.path.isfile(orig_video_path):
                temp_withaudio = os.path.join(save_dir, base_name + "_temp_withaudio.mp4")
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                cmd_mux = [
                    ffmpeg_path,"-y",
                    "-i", out_temp_path,
                    "-i", orig_video_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest",
                    temp_withaudio
                ]
                try:
                    subprocess.run(cmd_mux, check=True)
                    if os.path.exists(temp_withaudio):
                        shutil.move(temp_withaudio, final_path)
                        time.sleep(0.5)
                        os.remove(out_temp_path)
                        return final_path
                    else:
                        shutil.move(out_temp_path, final_path)
                        return final_path
                except subprocess.CalledProcessError as e:
                    shutil.move(out_temp_path, final_path)
                    return final_path
            else:
                shutil.move(out_temp_path, final_path)
                return final_path
        else:
            frames_np = frames_np.astype(np.float32)/255.0
            if color_space=="ACEScg":
                frames_np = srgb_to_acescg(frames_np)
            out_exr_dir = os.path.join(save_dir, base_name + "_exrseq_" + sbs_mode)
            half_f = (inpaint_output_mode=="exr16")
            save_exr_sequence_color(frames_np, out_exr_dir, half_float=half_f)
            return out_exr_dir
    overlap = fix_overlap_before_inference(total_frames, frames_chunk, overlap)
    stride = max(1, frames_chunk - overlap)
    if origin_mode=="2x2":
        half_h_final = H_in//2
        half_w_final = W_in//2
        if sbs_mode.upper()=="FSBS":
            if fsbs_double_height:
                final_w = half_w_final * 2
                final_h = half_h_final * 2
            else:
                final_w = half_w_final * 2
                final_h = half_h_final
        elif sbs_mode.upper()=="HSBS":
            final_w = half_w_final
            final_h = half_h_final
        else:
            final_w = half_w_final * 2
            final_h = half_h_final
    else:
        if sbs_mode.upper()=="HSBS":
            final_w = W_in
            final_h = H_in
        elif sbs_mode.upper()=="FSBS":
            if fsbs_double_height:
                final_w = W_in * 2
                final_h = H_in * 2
            else:
                final_w = W_in * 2
                final_h = H_in
        else:
            final_w = W_in
            final_h = H_in
    final_frames = []
    generated_prev = None
    generated_prev_small = None
    start_idx = 0
    chunk_index = 0
    while True:
        if start_idx >= total_frames:
            break
        remainder = total_frames - start_idx
        if remainder <= 0:
            break
        end = min(start_idx + frames_chunk, total_frames)
        csize = end - start_idx
        if csize <= 0:
            break
        print(f"[INFO] Processing chunk #{chunk_index}, frames {start_idx} to {end-1}, total in chunk={csize}")
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
            mask_np[i] = mask_blurred.astype(np.float32)/255.0
        mask_t = torch.from_numpy(mask_np).unsqueeze(1).to(mask_t.device).float()
        report_gpu_mem(prefix=f"Chunk #{chunk_index} before inpainting")
        if inpaint:
            if use_color_ref and GLOBAL_COLOR_REF is not None and partial_ref_frames>0:
                global_ref_np = (GLOBAL_COLOR_REF.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype("uint8")
                warp_np = warp_t.permute(0,2,3,1).cpu().numpy()
                limit_frames = min(partial_ref_frames, warp_np.shape[0])
                for i_ in range(limit_frames):
                    src_frame = (warp_np[i_]*255).clip(0,255).astype("uint8")
                    matched_ = cm.transfer(src=src_frame, ref=global_ref_np, method='mkl')
                    matched_ = np.clip(matched_, 0, 255).astype("uint8")
                    warp_np[i_] = matched_.astype(np.float32)/255.0
                warp_t = torch.from_numpy(warp_np).permute(0,3,1,2).float().to(warp_t.device)
                del warp_np
            elif use_color_ref and GLOBAL_COLOR_REF is not None:
                global_ref_np = (GLOBAL_COLOR_REF.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype("uint8")
                warp_np = warp_t.permute(0,2,3,1).cpu().numpy()
                for i_ in range(warp_np.shape[0]):
                    src_frame = (warp_np[i_]*255).clip(0,255).astype("uint8")
                    matched_ = cm.transfer(src=src_frame, ref=global_ref_np, method='mkl')
                    matched_ = np.clip(matched_, 0, 255).astype("uint8")
                    warp_np[i_] = matched_.astype(np.float32)/255.0
                warp_t = torch.from_numpy(warp_np).permute(0,3,1,2).float().to(warp_t.device)
                del warp_np
            if downscale_inpainting:
                warp_t_small = F.interpolate(warp_t, scale_factor=0.5, mode="bilinear", align_corners=False)
                mask_t_small = F.interpolate(mask_t, scale_factor=0.5, mode="bilinear", align_corners=False)
                B_small, C_small, H_small, W_small = warp_t_small.shape
                H_small_aligned = (H_small // 8) * 8
                W_small_aligned = (W_small // 8) * 8
                if H_small_aligned < H_small or W_small_aligned < W_small:
                    warp_t_small = warp_t_small[:, :, :H_small_aligned, :W_small_aligned]
                    mask_t_small = mask_t_small[:, :, :H_small_aligned, :W_small_aligned]
                if generated_prev_small is not None and overlap>0 and start_idx!=0:
                    b1 = generated_prev_small.shape[0]
                    b2 = warp_t_small.shape[0]
                    ov_act = min(overlap, b1, b2)
                    if ov_act>0:
                        warp_t_small[:ov_act] = generated_prev_small[-ov_act:]
                report_gpu_mem(prefix=f"Chunk #{chunk_index} before spatial tiled process (downscaled)")
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
                report_gpu_mem(prefix=f"Chunk #{chunk_index} after decoding (downscaled)")
                if start_idx!=0 and overlap>0 and right_chunk_small.shape[0]>overlap:
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
                if overlap>0 and right_chunk_small.shape[0]>=overlap:
                    generated_prev_small = right_chunk_small[-overlap:].clone()
                else:
                    generated_prev_small = right_chunk_small.clone()
                right_chunk_up = []
                for i2 in range(Tfin):
                    rc_small = right_chunk_small[i2].unsqueeze(0)
                    H_orig = warp_t[i2].shape[1]
                    W_orig = warp_t[i2].shape[2]
                    rc_up = F.interpolate(rc_small, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
                    right_chunk_up.append(rc_up.squeeze(0))
                right_chunk = torch.stack(right_chunk_up, dim=0)
            else:
                if generated_prev is not None and overlap>0 and start_idx!=0:
                    b1 = generated_prev.shape[0]
                    b2 = warp_t.shape[0]
                    ov_act = min(overlap, b1, b2)
                    if ov_act>0:
                        warp_t[:ov_act] = generated_prev[-ov_act:]
                report_gpu_mem(prefix=f"Chunk #{chunk_index} before spatial tiled process")
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
                report_gpu_mem(prefix=f"Chunk #{chunk_index} after decoding")
                if start_idx!=0 and overlap>0 and right_chunk.shape[0]>overlap:
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
                if overlap>0 and right_chunk.shape[0]>=overlap:
                    generated_prev = right_chunk[-overlap:].clone()
                else:
                    generated_prev = right_chunk.clone()
            for i2 in range(Tfin):
                inpainted = right_chunk[i2]
                original = warp_t[i2]
                mask_val = mask_t[i2][0].clamp(0,1)
                temp_right = local_color_match_masked_region(inpainted.clone(), left_t[i2], mask_t[i2], cm)
                final_right = original.clone()
                bin_mask = (mask_val > 0.5)
                final_right[:, bin_mask] = temp_right[:, bin_mask]
                final_right = local_color_match_masked_region(final_right, left_t[i2], mask_t[i2], cm)
                right_chunk[i2] = final_right.clamp(0,1)
            no_pad_left_list = []
            no_pad_right_list = []
            for i2 in range(Tfin):
                left_nopad = remove_padding_tensor(left_t[i2], tH, bH, lW, rW)
                right_nopad = remove_padding_tensor(right_chunk[i2], tH, bH, lW, rW)
                no_pad_left_list.append(left_nopad.unsqueeze(0))
                no_pad_right_list.append(right_nopad.unsqueeze(0))
            left_t_nopad = torch.cat(no_pad_left_list, dim=0)
            right_chunk_nopad = torch.cat(no_pad_right_list, dim=0)
            up_right = []
            for i2 in range(Tfin):
                rc_small = right_chunk_nopad[i2].unsqueeze(0).clamp(0,1)
                if origin_mode=="2x2":
                    if sbs_mode.upper()=="HSBS":
                        rc_up = F.interpolate(rc_small, size=(half_h_in, half_w_in//2), mode="bilinear", align_corners=False)
                    elif sbs_mode.upper()=="FSBS":
                        if fsbs_double_height:
                            rc_up = F.interpolate(rc_small, size=(half_h_in*2, half_w_in), mode="bilinear", align_corners=False)
                        else:
                            rc_up = F.interpolate(rc_small, size=(half_h_in, half_w_in), mode="bilinear", align_corners=False)
                    else:
                        rc_up = F.interpolate(rc_small, size=(half_h_in, half_w_in), mode="bilinear", align_corners=False)
                else:
                    if sbs_mode.upper()=="HSBS":
                        rc_up = F.interpolate(rc_small, size=(H_in, W_in//2), mode="bilinear", align_corners=False)
                    elif sbs_mode.upper()=="FSBS":
                        if fsbs_double_height:
                            rc_up = F.interpolate(rc_small, size=(H_in*2, W_in), mode="bilinear", align_corners=False)
                        else:
                            rc_up = F.interpolate(rc_small, size=(H_in, W_in), mode="bilinear", align_corners=False)
                    else:
                        rc_up = F.interpolate(rc_small, size=(H_in, W_in//2), mode="bilinear", align_corners=False)
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
                    lf_1 = left_t_nopad[i2].unsqueeze(0)
                    if origin_mode=="2x2":
                        if sbs_mode.upper()=="HSBS":
                            lf_up = F.interpolate(lf_1, size=(half_h_in, half_w_in//2), mode="bilinear", align_corners=False)
                        elif sbs_mode.upper()=="FSBS":
                            if fsbs_double_height:
                                lf_up = F.interpolate(lf_1, size=(half_h_in*2, half_w_in), mode="bilinear", align_corners=False)
                            else:
                                lf_up = F.interpolate(lf_1, size=(half_h_in, half_w_in), mode="bilinear", align_corners=False)
                        else:
                            lf_up = F.interpolate(lf_1, size=(half_h_in, half_w_in), mode="bilinear", align_corners=False)
                    else:
                        if sbs_mode.upper()=="HSBS":
                            lf_up = F.interpolate(lf_1, size=(H_in, W_in//2), mode="bilinear", align_corners=False)
                        elif sbs_mode.upper()=="FSBS":
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
            if use_color_ref and len(up_right_t) > 0:
                GLOBAL_COLOR_REF = up_right_t[-1].clone()
        else:
            Tfin = warp_t.shape[0]
            left_t = left_t[:Tfin]
            no_pad_left_list = []
            no_pad_right_list = []
            for i2 in range(Tfin):
                lf_nopad = remove_padding_tensor(left_t[i2], tH, bH, lW, rW)
                wr_nopad = remove_padding_tensor(warp_t[i2], tH, bH, lW, rW)
                no_pad_left_list.append(lf_nopad.unsqueeze(0))
                no_pad_right_list.append(wr_nopad.unsqueeze(0))
            left_t_nopad = torch.cat(no_pad_left_list, dim=0)
            right_nopad = torch.cat(no_pad_right_list, dim=0)
            up_right = []
            for i2 in range(Tfin):
                rc = right_nopad[i2].unsqueeze(0).clamp(0,1)
                if origin_mode=="2x2":
                    if sbs_mode.upper()=="HSBS":
                        rc_up = F.interpolate(rc, size=(half_h_in, half_w_in//2), mode="bilinear", align_corners=False)
                    elif sbs_mode.upper()=="FSBS":
                        if fsbs_double_height:
                            rc_up = F.interpolate(rc, size=(half_h_in*2, half_w_in), mode="bilinear", align_corners=False)
                        else:
                            rc_up = F.interpolate(rc, size=(half_h_in, half_w_in), mode="bilinear", align_corners=False)
                    else:
                        rc_up = F.interpolate(rc, size=(half_h_in, half_w_in), mode="bilinear", align_corners=False)
                else:
                    if sbs_mode.upper()=="HSBS":
                        rc_up = F.interpolate(rc, size=(H_in, W_in//2), mode="bilinear", align_corners=False)
                    elif sbs_mode.upper()=="FSBS":
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
                    lf_1 = left_t_nopad[i2].unsqueeze(0)
                    if origin_mode=="2x2":
                        if sbs_mode.upper()=="HSBS":
                            lf_up = F.interpolate(lf_1, size=(half_h_in, half_w_in//2), mode="bilinear", align_corners=False)
                        elif sbs_mode.upper()=="FSBS":
                            if fsbs_double_height:
                                lf_up = F.interpolate(lf_1, size=(half_h_in*2, half_w_in), mode="bilinear", align_corners=False)
                            else:
                                lf_up = F.interpolate(lf_1, size=(half_h_in, half_w_in), mode="bilinear", align_corners=False)
                        else:
                            lf_up = F.interpolate(lf_1, size=(half_h_in, half_w_in), mode="bilinear", align_corners=False)
                    else:
                        if sbs_mode.upper()=="HSBS":
                            lf_up = F.interpolate(lf_1, size=(H_in, W_in//2), mode="bilinear", align_corners=False)
                        elif sbs_mode.upper()=="FSBS":
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
        torch.cuda.empty_cache()
        gc.collect()
        remaining_frames = total_frames - end
        remaining_chunks = math.ceil(remaining_frames / stride) if stride > 0 else 0
        print(f"[INFO] Completed chunk #{chunk_index}, processed frames {start_idx} to {end-1}. Chunk size={csize}, total processed so far={len(final_frames)}. Remaining frames={remaining_frames}, estimated remaining chunks={remaining_chunks}")
        start_idx += stride
        chunk_index += 1
    if not final_frames:
        return None
    if inpaint_output_mode=="mp4":
        print(f"[INFO] Writing final MP4 => {out_temp_path}")
        frames_np = np.stack(final_frames, axis=0)
        frames_np = np.ascontiguousarray(frames_np)
        if encoder == "x264":
            cdc = "libx264"
            crf = 0
            cmd_params = [
                "-crf", str(crf),
                "-preset", "slow",
                "-profile:v", "high10",
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-color_range", "tv",
                "-tag:v", "avc1",
                "-movflags", "+faststart+write_colr"
            ]
        elif encoder == "x265":
            cdc = "libx265"
            crf = 12
            cmd_params = [
                "-crf", str(crf),
                "-preset", "slow",
                "-x265-params", "high-tier=1:colormatrix=bt709:colorprim=bt709:transfer=bt709:range=limited",
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-color_range", "tv",
                "-tag:v", "hvc1",
                "-movflags", "+faststart+write_colr"
            ]
        else:
            cdc = "libx264"
            crf = 0
            cmd_params = [
                "-crf", str(crf),
                "-preset", "slow",
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-color_range", "tv",
                "-tag:v", "avc1",
                "-movflags", "+faststart+write_colr"
            ]
        writer = imageio_ffmpeg.write_frames(
            out_temp_path,
            (frames_np.shape[2], frames_np.shape[1]),
            fps=fps,
            codec=cdc,
            output_params=cmd_params
        )
        writer.send(None)
        for i in range(frames_np.shape[0]):
            frame_c = np.ascontiguousarray(frames_np[i])
            writer.send(frame_c)
        writer.close()
        final_path = os.path.join(save_dir, base_name + "_" + sbs_mode + "_" + encoder + ".mp4")
        if not os.path.exists(out_temp_path):
            return None
        if orig_video_path and os.path.isfile(orig_video_path):
            print(f"[INFO] Found original video => {orig_video_path}, adding audio track")
            temp_withaudio = os.path.join(save_dir, base_name + "_temp_withaudio.mp4")
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            cmd_mux = [
                ffmpeg_path,"-y",
                "-i", out_temp_path,
                "-i", orig_video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                temp_withaudio
            ]
            print("[INFO]", " ".join(cmd_mux))
            try:
                subprocess.run(cmd_mux, check=True)
                if os.path.exists(temp_withaudio):
                    shutil.move(temp_withaudio, final_path)
                    time.sleep(0.5)
                    os.remove(out_temp_path)
                    print("[INFO] Audio muxed successfully =>", final_path)
                    return final_path
                else:
                    shutil.move(out_temp_path, final_path)
                    return final_path
            except subprocess.CalledProcessError as e:
                shutil.move(out_temp_path, final_path)
                return final_path
        else:
            shutil.move(out_temp_path, final_path)
            return final_path
    else:
        frames_np = np.stack(final_frames, axis=0).astype(np.float32)/255.0
        final_frames.clear()
        if color_space=="ACEScg":
            frames_np = srgb_to_acescg(frames_np)
        out_exr_dir = os.path.join(save_dir, base_name + "_exrseq_" + sbs_mode)
        half_f = (inpaint_output_mode=="exr16")
        save_exr_sequence_color(frames_np, out_exr_dir, half_float=half_f)
        return out_exr_dir

def batch_process(
    pre_trained_path="./weights/stable-video-diffusion-img2vid-xt-1-1",
    unet_path="./weights/StereoCrafter",
    input_folder="./output_splatted",
    output_folder="./completed_output",
    frames_chunk=23,
    overlap=3,
    tile_num=2,
    color_match=True,
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
    partial_ref_frames=0,
    inpaint=True
):
    print("[INFO] Starting batch_process, loading pipeline and models.")
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
    else:
        exts = ("*.mp4", "*.mov", "*.avi", "*.mkv")
        vids = []
        for e_ in exts:
            vids.extend(glob.glob(os.path.join(input_folder, e_)))
        input_videos = sorted(vids)
        if not input_videos:
            return
    if not origin_mode:
        if single_video:
            base_ = os.path.basename(single_video)
            name_noext, ext_ = os.path.splitext(base_)
            folder_ = os.path.dirname(single_video)
            if "_splatted" in name_noext:
                base_candidate = name_noext.replace("_splatted", "")
                found_left = os.path.join(folder_, base_candidate + "_left.mp4")
                found_mask = os.path.join(folder_, base_candidate + "_mask.mp4")
                if os.path.isfile(found_left) and os.path.isfile(found_mask):
                    origin_mode = "triple"
                    left_video_path = found_left
                    mask_video_path = found_mask
                    warp_video_path = single_video
                else:
                    origin_mode = "2x2"
            else:
                origin_mode = "2x2"
        else:
            origin_mode = "2x2"
    processed = []
    if origin_mode=="triple" and single_video:
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
            partial_ref_frames=partial_ref_frames,
            inpaint=inpaint
        )
        if outp:
            processed.append(outp)
    elif origin_mode=="triple" and not single_video:
        pass
    else:
        for vid in input_videos:
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
                partial_ref_frames=partial_ref_frames,
                inpaint=inpaint
            )
            if outp:
                processed.append(outp)
            torch.cuda.empty_cache()
            gc.collect()
    pipeline.unet.to(torch.float32)
    pipeline.vae.to(torch.float32)
    pipeline.image_encoder.to(torch.float32)
    pipeline.to("cpu")
    del pipeline, unet, vae, image_encoder
    torch.cuda.empty_cache()
    gc.collect()
    print("[INFO] batch_process completed.")

if __name__=="__main__":
    Fire(batch_process)
