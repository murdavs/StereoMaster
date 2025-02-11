import os
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
import glob
import gc
import time
import numpy as np
import shutil
import subprocess
from fire import Fire
from decord import VideoReader, cpu
import torch
import torch.nn.functional as F

from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel

from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline, tensor2vid
import imageio_ffmpeg 


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
    If tile_num>1, zero-pad the frames so each tile can be
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
    ideal_w = stride_x*tile_num + ov_x*(tile_num-1)

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
                **kargs  # <-- AQUÍ entran num_inference_steps, etc.
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

    # Mezcla vertical/horizontal
    for i in range(tile_num):
        for j in range(tile_num):
            tile = row_tiles[i][j]
            if i > 0:
                tile = blend_v(row_tiles[i-1][j], tile, latent_overlap[0])
            if j > 0:
                tile = blend_h(row_tiles[i][j-1], tile, latent_overlap[1])
            row_tiles[i][j] = tile

    # Coser horizontal en cada fila
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

    # Finalmente coser filas
    final_latent = torch.cat(row_tiles, dim=2)
    dt_spatial = time.time() - start_t

    return final_latent


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
    Codifica frames_list (list of [H,W,3] np.uint8) en un mp4.
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
def process_single_video_in_chunks(
    pipeline,
    input_video_path,
    save_dir,
    frames_chunk=23,
    overlap=3,
    tile_num=2,
    color_match=True,
    threshold_mask=0.005,
    sbs_mode="HSBS",
    encoder="x264",
    origin_mode="2x2",
    left_video_path=None,
    mask_video_path=None,
    warp_video_path=None,
    orig_video_path=None,
    num_inference_steps=15  # <-- NUEVO
):

    print(f"\n[INFO] => process_single_video_in_chunks => {input_video_path}, origin_mode={origin_mode}")
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_video_path))[0] + "_inpainting_results"
    out_temp_path = os.path.join(save_dir, f"{base_name}_temp.mp4")
    final_path = os.path.join(save_dir, f"{base_name}_{sbs_mode}_{encoder}.mp4")

    start_init = time.time()

    # ---------------------------------------
    # 1) Leer frames (2x2 o triple)
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
        fps = max(fps, 1.0)  # fallback

    else:  # origin_mode=="triple"
        vr_left = VideoReader(left_video_path, ctx=cpu(0))
        vr_mask = VideoReader(mask_video_path, ctx=cpu(0))
        vr_warp = VideoReader(warp_video_path, ctx=cpu(0))
        nL = len(vr_left)
        nM = len(vr_mask)
        nW = len(vr_warp)
        nF = min(nL, nM, nW)
        if nF == 0:
            print("[WARN] => triple => one of the videos has 0 frames => skipping.")
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

    # Determinar tamaño final SBS
    if origin_mode == "2x2":
        half_h_final = H_in//2
        half_w_final = W_in//2
        if sbs_mode.upper() == "FSBS":
            final_w = half_w_final*2
            final_h = half_h_final
        else:
            final_w = half_w_final
            final_h = half_h_final
    else:  # triple
        if sbs_mode.upper() == "FSBS":
            final_w = W_in*2
            final_h = H_in
        else:
            final_w = W_in
            final_h = H_in

    print(f"[INFO] => final SBS resolution = {final_w}x{final_h}")
    final_frames = []
    generated_prev = None

    dt_init = time.time() - start_init
    report_gpu_mem("after reading")

    # ---------------------------------------
    # 2) Process frames in chunks
    # ---------------------------------------
    start_idx = 0
    while start_idx < total_frames:
        chunk_start_time = time.time()
        end = min(start_idx + frames_chunk, total_frames)
        csize = end - start_idx
        if csize <= 0:
            break

        print(f"\n[CHUNK] => {start_idx}..{end} (size={csize})")
        warp_chunk_np = warp_orig_list[start_idx:end]
        mask_chunk_np = mask_orig_list[start_idx:end]
        left_chunk_np = left_orig_list[start_idx:end]

        # Mover a GPU
        warp_t = torch.from_numpy(np.stack(warp_chunk_np, axis=0)).permute(0,3,1,2).float().cuda()
        mask_t = torch.from_numpy(np.stack(mask_chunk_np, axis=0)).permute(0,3,1,2).float().cuda()
        left_t = torch.from_numpy(np.stack(left_chunk_np, axis=0)).permute(0,3,1,2).float().cuda()

        # Máscara en gris
        mask_t = mask_t.mean(dim=1, keepdim=True)

        # Overlap temporal
        if generated_prev is not None and overlap > 0 and start_idx != 0:
            b1 = generated_prev.shape[0]
            b2 = warp_t.shape[0]
            ov_act = min(overlap, b1, b2)
            if ov_act > 0:
                print(f"[CHUNK] => Overlap={ov_act} frames from previous chunk")
                warp_t[:ov_act] = generated_prev[-ov_act:]

        # Si tile_num>1
        if tile_num > 1:
            warp_t = pad_for_tiling(warp_t, tile_num=tile_num, tile_overlap=(128,128))
            mask_t = pad_for_tiling(mask_t, tile_num=tile_num, tile_overlap=(128,128))

        report_gpu_mem("before spatial_tiled_process")
        t_spatial_start = time.time()

        # Llamamos a spatial_tiled_process => num_inference_steps se pasa por kwargs
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
                num_inference_steps=num_inference_steps  # <-- NUEVO
            )

        dt_spatial = time.time() - t_spatial_start

        # lat_ => [T, C, H/8, W/8], decodificar
        lat_ = lat_.unsqueeze(0)  # [1, T, C, H/8, W/8]

        pipeline.vae.to(torch.float16)
        t_decode_start = time.time()
        dec = pipeline.decode_latents(lat_, num_frames=lat_.shape[1], decode_chunk_size=1)
        dec_frames = tensor2vid(dec, pipeline.image_processor, output_type="pil")[0]
        dt_decode = time.time() - t_decode_start

        # Convertir a tensores float[0..1]
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

        # Quitar overlapped frames si no es la primera chunk
        if start_idx != 0 and overlap > 0:
            if right_chunk.shape[0] > overlap:
                right_chunk = right_chunk[overlap:]

        generated_prev = right_chunk

        # Ajustar shapes
        bA = right_chunk.shape[0]
        if warp_t.shape[0] > bA: warp_t = warp_t[-bA:]
        if mask_t.shape[0] > bA: mask_t = mask_t[-bA:]
        if left_t.shape[0] > bA: left_t = left_t[-bA:]
        Tfin = min(right_chunk.shape[0], warp_t.shape[0], mask_t.shape[0], left_t.shape[0])
        right_chunk = right_chunk[:Tfin]
        warp_t      = warp_t[:Tfin]
        mask_t      = mask_t[:Tfin]
        left_t      = left_t[:Tfin]

        print(f"[CHUNK] => Tfinal={Tfin}, alpha-blend + color matching...")
        # 1) alpha-blend
        for i2 in range(Tfin):
            inpainted = right_chunk[i2]
            original  = warp_t[i2]
            alpha     = (mask_t[i2][0] > threshold_mask).float().unsqueeze(0).repeat(3,1,1)
            final_    = inpainted*alpha + original*(1 - alpha)
            right_chunk[i2] = final_

        # 2) color-match
        if color_match:
            for i2 in range(Tfin):
                lf = left_t[i2]
                rf = right_chunk[i2]
                for c in range(3):
                    mean_l = lf[c].mean()
                    std_l  = lf[c].std(unbiased=False) + 1e-6
                    mean_r = rf[c].mean()
                    std_r  = rf[c].std(unbiased=False) + 1e-6
                    rf[c]  = (rf[c] - mean_r)/std_r
                    rf[c]  = rf[c]*std_l + mean_l
                rf.clamp_(0, 1)

        # 3) re-scale
        up_right = []
        for i2 in range(Tfin):
            rc = right_chunk[i2].unsqueeze(0)
            if origin_mode == "2x2":
                rc_up = F.interpolate(rc, size=(H_in//2, W_in//2), mode="bilinear", align_corners=False)
            else:
                rc_up = F.interpolate(rc, size=(H_in, W_in), mode="bilinear", align_corners=False)
            up_right.append(rc_up.squeeze(0))
        up_right_t = torch.stack(up_right, dim=0)

        # 4) Compose SBS
        for i2 in range(Tfin):
            lf_1 = left_t[i2].unsqueeze(0)
            if origin_mode == "2x2":
                lf_up = F.interpolate(lf_1, size=(H_in//2, W_in//2), mode="bilinear", align_corners=False)
            else:
                lf_up = F.interpolate(lf_1, size=(H_in, W_in), mode="bilinear", align_corners=False)
            lf_up = lf_up.squeeze(0)

            rf_t = up_right_t[i2]

            if sbs_mode.upper() == "FSBS":
                sbs = torch.cat([lf_up, rf_t], dim=2)
            else:
                half_w_l = lf_up.shape[2] // 2
                half_w_r = rf_t.shape[2] // 2
                lf_small = F.interpolate(
                    lf_up.unsqueeze(0),
                    size=(lf_up.shape[1], half_w_l),
                    mode="bilinear", align_corners=False
                ).squeeze(0)
                rf_small = F.interpolate(
                    rf_t.unsqueeze(0),
                    size=(rf_t.shape[1], half_w_r),
                    mode="bilinear", align_corners=False
                ).squeeze(0)
                sbs = torch.cat([lf_small, rf_small], dim=2)

            sbs_cpu = sbs.detach().cpu().clamp(0,1)
            sbs_uint8 = (sbs_cpu*255).byte().permute(1,2,0).numpy()
            final_frames.append(sbs_uint8)

        # Limpieza chunk
        del warp_t, mask_t, left_t, right_chunk, up_right_t, up_right
        torch.cuda.empty_cache()
        gc.collect()

        chunk_dt = time.time() - chunk_start_time
        print(f"[CHUNK] => Finished chunk in {chunk_dt:.2f}s")
        report_gpu_mem("after chunk")
        start_idx += stride

    # ---------------------------------------
    # 3) Escribir video SBS final (sin audio)
    # ---------------------------------------
    if not final_frames:
        print("[WARN] => No frames => skipping =>", out_temp_path)
        return None

    print(f"[INFO] => Encoding final video => {out_temp_path}")
    frames_np = np.stack(final_frames, axis=0)
    frames_np = np.ascontiguousarray(frames_np)
    import imageio_ffmpeg
    crf = 0 if encoder=="x264" else 0
    cdc = "libx265" if encoder=="x265" else "libx264"
    cmd_params = [
        "-crf", str(crf),
        "-preset", "veryslow"
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
        writer.send(frames_np[i])
    writer.close()

    final_frames.clear()

    if not os.path.exists(out_temp_path):
        print("[WARN] => No temp file => no final =>", out_temp_path)
        return None
    # ---------------------------------------
    # 4) Mux audio (si orig_video_path existe)
    # ---------------------------------------
    import imageio_ffmpeg  # Asegúrate de que esté instalado: pip install imageio-ffmpeg

    if orig_video_path and os.path.isfile(orig_video_path):
        print(f"[AUDIO] => Found orig video => {orig_video_path}, muxing audio into final SBS...")
        temp_withaudio = os.path.join(save_dir, f"{base_name}_temp_withaudio.mp4")
        
        # Obtenemos la ruta absoluta de ffmpeg que maneja imageio
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        cmd_mux = [
            ffmpeg_path,  # <--- Usamos la ruta completa a ffmpeg
            "-y",
            "-i", out_temp_path,   # SBS sin audio
            "-i", orig_video_path, # audio original
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

            # Esperamos un poco antes de borrar el archivo temporal de vídeo
        
            time.sleep(0.5)

            # Ahora intentamos eliminarlo
            os.remove(out_temp_path)
            print("[AUDIO] => done =>", final_path)
            return final_path
        else:
            print("[AUDIO] => failed => no temp_withaudio => fallback => no audio")
            shutil.move(out_temp_path, final_path)
            return final_path
    except subprocess.CalledProcessError as e:
        print("[AUDIO] => error =>", e)
        shutil.move(out_temp_path, final_path)
        return final_path



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
    sbs_mode="HSBS",
    encoder="x264",
    origin_mode="",
    left_video_path=None,
    mask_video_path=None,
    warp_video_path=None,
    orig_video=None,         # <-- para inyectar audio
    num_inference_steps=10   
):

    print("[INFO] => batch_process => Loading pipeline...")

    # Cargar pipeline
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

    # Seleccionar qué videos procesar
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

    # Auto-detect origin_mode si no se pasa
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
            sbs_mode=sbs_mode,
            encoder=encoder,
            origin_mode="triple",
            left_video_path=left_video_path,
            mask_video_path=mask_video_path,
            warp_video_path=warp_video_path,
            orig_video_path=orig_video,         # <-- pass audio source
            num_inference_steps=num_inference_steps  
        )
        if outp:
            processed.append(outp)
    elif origin_mode == "triple" and not single_video:
        print("[WARN] => origin_mode=triple => no single_video => not implemented in this example.")
    else:
        # 2x2 mode
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
                encoder=encoder,
                origin_mode="2x2",
                orig_video_path=orig_video,         # <-- pass audio source
                num_inference_steps=num_inference_steps 
            )
            if outp:
                processed.append(outp)
            torch.cuda.empty_cache()
            gc.collect()

    # Concat final si se pide
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

    # Cleanup pipeline
    pipeline.unet.to(torch.float32)
    pipeline.vae.to(torch.float32)
    pipeline.image_encoder.to(torch.float32)
    pipeline.to("cpu")
    del pipeline, unet, vae, image_encoder
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    Fire(batch_process)
