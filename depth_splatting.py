import gc
import os
import math
import cv2
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fire import Fire
from decord import VideoReader, cpu
import imageio_ffmpeg
import subprocess
import uuid

################################################
# Try importing DepthCrafter if available
################################################
try:
    from dependency.DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
    from dependency.DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
except ImportError:
    DepthCrafterPipeline = None
    DiffusersUNetSpatioTemporalConditionModelDepthCrafter = None
    print("[WARN] => DepthCrafter not imported => only pre-rendered depth usage")

try:
    from Forward_Warp import forward_warp
except ImportError:
    forward_warp = None
    print("[WARN] => forward_warp.py not found => cannot do splatting")


################################################
# (1) Ensure decord can read the video
################################################
def ensure_clean_video(video_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        if num_frames == 0:
            raise ValueError("Video with 0 frames. Possibly corrupted.")
        return video_path
    except Exception as e:
        print(f"[WARN] => decord could not read '{video_path}' properly: {e}")
        print("[INFO] => Attempting to re-encode with ffmpeg...")

        tmp_name = f"temp_fixed_{uuid.uuid4().hex}.mp4"
        cmd = [
            "ffmpeg_exe", "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryslow",
            "-crf", "0",
            "-c:a", "copy",
            tmp_name
        ]
        print(f"[INFO] => Running: {' '.join(cmd)}")
        ret = subprocess.run(cmd, capture_output=True)
        if ret.returncode != 0:
            print("[ERROR] => ffmpeg re-encode failed. Returning original video path.")
            return video_path

        original_backup = f"{video_path}.backup"
        try:
            os.rename(video_path, original_backup)
            os.rename(tmp_name, video_path)
            os.remove(original_backup)
        except Exception as e2:
            print("[ERROR] => Could not replace the original:", e2)
            if os.path.exists(original_backup):
                os.rename(original_backup, video_path)
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            return video_path

        print("[OK] => The video was re-encoded successfully. decord should read it fine now.")
        return video_path


################################################
# (2) Re-encode with brightness/gamma
################################################
def reencode_brightness_gamma(in_path, out_path, brightness=1.0, gamma=1.0):
    bri_eq = brightness - 1.0
    bri_eq = max(-1.0, min(1.0, bri_eq))
    gamma  = max(0.1,  min(10.0, gamma))

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"[ERROR] => Could not open video: {in_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    idx=0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_float = frame_bgr.astype(np.float32)/255.0
        # Brightness shift
        frame_float = np.clip(frame_float + bri_eq, 0,1)
        # Gamma
        if abs(gamma-1.0) > 1e-4:
            frame_float = np.power(frame_float, gamma)
        frame_float = np.clip(frame_float,0,1)

        frame_u8 = (frame_float*255.0).astype(np.uint8)
        out.write(frame_u8)

        idx+=1
        if idx%50==0:
            print(f"[INFO] => re-encode => {idx}/{total_frames} frames ...")

    cap.release()
    out.release()
    print(f"[INFO] => Re-encoded => {out_path} with brightness={brightness}, gamma={gamma}")


################################################
# (3) Depth colormap & saving grayscale
################################################
def chunked_vis_sequence_depth(depth_array, chunk_size=50):
    import cv2
    import numpy as np
    T,H,W= depth_array.shape
    out=[]
    start=0
    while start< T:
        end = min(start+chunk_size, T)
        chunk= depth_array[start:end]
        chunk_c=[]
        for f_ in chunk:
            f255= (f_*255).astype(np.uint8)
            c_ = cv2.applyColorMap(f255, cv2.COLORMAP_JET)
            chunk_c.append(c_.astype(np.float32)/255.0)
        chunk_c= np.stack(chunk_c, axis=0)
        out.append(chunk_c)
        start= end
    return np.concatenate(out, axis=0)


def save_grayscale_depth_video(depth_data, out_path, fps):
    import cv2
    import numpy as np
    T,H,W= depth_data.shape
    fourcc= cv2.VideoWriter_fourcc(*"mp4v")
    outv= cv2.VideoWriter(out_path, fourcc, fps, (W,H))
    for i in range(T):
        f_= depth_data[i]
        f255= (f_*255).astype(np.uint8)
        f3= cv2.cvtColor(f255, cv2.COLOR_GRAY2BGR)
        outv.write(f3)
    outv.release()
    print(f"[INFO] => Depth saved => {out_path}")


################################################
# (4) Load pre-rendered depth => (vd, dv)
################################################
def load_pre_rendered_depth(depth_video_path, process_length=-1):
    from decord import VideoReader, cpu
    import numpy as np
    import cv2

    vr= VideoReader(depth_video_path, ctx=cpu(0))
    frames= vr[:].asnumpy().astype(np.float32)/255.0
    if process_length!=-1 and process_length< len(frames):
        frames= frames[:process_length]

    vd= frames.mean(axis=-1)  # [T,H,W]
    vd= np.clip(vd, 0,1)

    dv_list=[]
    for f_ in vd:
        c_= cv2.applyColorMap((f_*255).astype(np.uint8), cv2.COLORMAP_JET)
        dv_list.append(c_.astype(np.float32)/255.0)
    dv= np.stack(dv_list, axis=0) 
    return vd, dv


################################################
# (5) DepthCrafter pipeline (optional)
################################################
class DepthCrafterDemo:
    def __init__(self, unet_path, pre_trained_path, cpu_offload="model"):
        if DepthCrafterPipeline is None or DiffusersUNetSpatioTemporalConditionModelDepthCrafter is None:
            raise ValueError("[ERROR] => DepthCrafter not available.")
        print("[INFO] => Loading DepthCrafter UNet =>", unet_path)
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
        )
        print("[INFO] => Loading pipeline =>", pre_trained_path)
        self.pipe= DepthCrafterPipeline.from_pretrained(
            pre_trained_path, unet=unet, torch_dtype=torch.float16, variant="fp16"
        )

        if cpu_offload=="sequential":
            self.pipe.enable_sequential_cpu_offload()
        elif cpu_offload=="model":
            self.pipe.enable_model_cpu_offload()
        elif cpu_offload is None:
            self.pipe.to("cuda")

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        input_video_path,
        output_video_path,
        process_length=-1,
        num_denoising_steps=8,
        guidance_scale=1.2,
        window_size=70,
        overlap=25,
        seed=42,
        track_time=False,
        max_res=1024,
        pad_to_multiple_of=64
    ):
        import torch
        import numpy as np
        import cv2
        from decord import VideoReader, cpu
        import math

        torch.manual_seed(seed)

        vr = VideoReader(input_video_path, ctx=cpu(0))
        orig_len= len(vr)
        oh,ow= vr.get_batch([0]).shape[1:3]
        fps= vr.get_avg_fps()

        frames_all = vr[:].asnumpy().astype(np.float32)/255.0
        if process_length!=-1 and process_length< orig_len:
            frames_all= frames_all[:process_length]
        T= len(frames_all)
        print(f"[INFO] => DepthCrafter => T={T} frames => shape=({oh},{ow}), max_res={max_res}, seed={seed}")

        newh, neww= oh, ow
        if max(oh,ow)> max_res>0:
            scale= max_res / max(oh,ow)
            newh= int(round(oh* scale))
            neww= int(round(ow* scale))
        nH= int(math.ceil(newh/pad_to_multiple_of)* pad_to_multiple_of)
        nW= int(math.ceil(neww/pad_to_multiple_of)* pad_to_multiple_of)
        phT= (nH- newh)//2
        phB= nH- newh- phT
        pwL= (nW- neww)//2
        pwR= nW- neww- pwL

        frames_padded=[]
        for fr_ in frames_all:
            if (newh!= oh) or (neww!= ow):
                fr_= cv2.resize(fr_, (neww,newh), interpolation=cv2.INTER_AREA)
            can= np.zeros((nH,nW,3), dtype=np.float32)
            can[phT:phT+newh, pwL:pwL+neww] = fr_
            frames_padded.append(can)
        frames_padded= np.stack(frames_padded, axis=0)

        effective_window = window_size - overlap
        self._n_chunks_total = max(1, (T + (effective_window-1)) // effective_window)
        self._current_chunk  = 0
        self._num_inference_steps = num_denoising_steps
        
        def global_callback(step: int, timestep: int, latents: torch.FloatTensor):
            chunk_index = self._current_chunk
            n_chunks    = self._n_chunks_total
            total_steps = n_chunks * self._num_inference_steps
            global_step = chunk_index * self._num_inference_steps + step
            pct = 100.0 * global_step / float(total_steps)
            print(f" [PROGRESS] => chunk {chunk_index+1}/{n_chunks}, step {step}/{self._num_inference_steps} => {pct:.1f}%")
            
        self.pipe.callback = global_callback
        self.pipe.callback_steps = 1

        with torch.inference_mode():
            out = self.pipe(
                frames_padded,
                height=nH,
                width=nW,
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time
            ).frames[0]
        depth_1c= out.mean(axis=-1)
        mn,mx= depth_1c.min(), depth_1c.max()
        depth_norm= (depth_1c- mn)/(mx- mn +1e-8)

        depth_unpad= depth_norm[:, phT:phT+newh, pwL:pwL+neww]
        if (newh!= oh) or (neww!= ow):
            import torch.nn.functional as F
            t_ = torch.from_numpy(depth_unpad).unsqueeze(1).float().cuda()
            t_ = F.interpolate(t_, size=(oh,ow), mode="bilinear", align_corners=False)
            depth_final = t_[:,0].cpu().numpy()
        else:
            depth_final= depth_unpad

        depth_vis= chunked_vis_sequence_depth(depth_final)
        if output_video_path:
            fourcc= cv2.VideoWriter_fourcc(*"mp4v")
            outv= cv2.VideoWriter(output_video_path, fourcc, fps, (ow,oh))
            for i in range(T):
                f_= depth_vis[i]
                f255= np.clip(f_*255,0,255).astype(np.uint8)
                frBGR= cv2.cvtColor(f255, cv2.COLOR_RGB2BGR)
                outv.write(frBGR)
            outv.release()
            print(f"[INFO] => DepthCrafter => wrote color depth => {output_video_path}")

        return depth_final, depth_vis, fps


################################################
# (6) Keyframes interpolation
################################################
def interpolate_frame_params(kframes, frame_idx, default_params):
    if not kframes:
        return default_params

    frames_sorted= sorted(kframes.keys())
    if frame_idx <= frames_sorted[0]:
        return {
            p: kframes[frames_sorted[0]].get(p, default_params[p])
            for p in default_params
        }
    if frame_idx >= frames_sorted[-1]:
        return {
            p: kframes[frames_sorted[-1]].get(p, default_params[p])
            for p in default_params
        }

    left_f= frames_sorted[0]
    right_f= frames_sorted[-1]
    for i in range(len(frames_sorted)-1):
        f0= frames_sorted[i]
        f1= frames_sorted[i+1]
        if f0 <= frame_idx <= f1:
            left_f= f0
            right_f= f1
            break

    if left_f== right_f:
        return {
            p: kframes[left_f].get(p, default_params[p])
            for p in default_params
        }

    ratio= (frame_idx-left_f)/ float(right_f-left_f +1e-9)
    result={}
    for p in default_params:
        lv= kframes[left_f].get(p, default_params[p])
        rv= kframes[right_f].get(p, default_params[p])
        if isinstance(lv, (int,float)) and isinstance(rv, (int,float)):
            val_ = lv + (rv-lv)* ratio
        else:
            val_ = lv
        result[p]= val_
    return result


################################################
# (7) Depth Preprocess => dilate + blur
################################################
def apply_depth_preprocess(depth_frame_01, dilate_h=0, dilate_v=0,
                           blur_ksize=3, blur_sigma=2.0):
    import cv2
    import numpy as np

    dh_i = int(round(dilate_h))
    dv_i = int(round(dilate_v))
    if dh_i < 0: dh_i = 0
    if dv_i < 0: dv_i = 0

    if dh_i > 0 or dv_i > 0:
        depth_u8= (depth_frame_01*255).astype(np.uint8)
        kernel= np.ones((dv_i, dh_i), np.uint8)
        depth_u8= cv2.dilate(depth_u8, kernel, iterations=1)
        depth_frame_01= depth_u8.astype(np.float32)/255.0

    bk_i = int(round(blur_ksize))
    if bk_i < 1:
        bk_i = 1
    if bk_i % 2 == 0:
        bk_i += 1

    if bk_i > 1:
        depth_255 = (depth_frame_01*255).astype(np.float32)
        depth_255 = cv2.GaussianBlur(depth_255, (bk_i, bk_i), sigmaX=blur_sigma)
        depth_frame_01 = np.clip(depth_255/255.0, 0,1)

    return depth_frame_01


################################################
# (8) Fast 2-pass distance transform inpaint
################################################
def fast_inpaint_distance_transform(image_bgr, mask_holes):
    H, W, _ = image_bgr.shape
    INF = 999999.0

    col_arr = image_bgr.astype(np.float32).copy()
    dist_arr= np.full((H,W), INF, dtype=np.float32)

    for y in range(H):
        for x in range(W):
            if not mask_holes[y,x]:
                dist_arr[y,x] = 0.0
            else:
                col_arr[y,x,:] = 0.0

    pass1_neighbors = [
        (-1,  0, 1.0),
        ( 0, -1, 1.0),
        (-1, -1, 1.4),
        (-1,  1, 1.4)
    ]
    pass2_neighbors = [
        ( 1,  0, 1.0),
        ( 0,  1, 1.0),
        ( 1,  1, 1.4),
        ( 1, -1, 1.4)
    ]

    # Pass 1 => top-left -> bottom-right
    for y in range(H):
        for x in range(W):
            d0 = dist_arr[y,x]
            c0 = col_arr[y,x]
            if d0 == INF:
                continue
            for (dy,dx,cost) in pass1_neighbors:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    old_d = dist_arr[ny,nx]
                    new_d = d0 + cost
                    if new_d < old_d:
                        dist_arr[ny,nx] = new_d
                        col_arr[ny,nx]  = c0

    # Pass 2 => bottom-right -> top-left
    for y in range(H-1, -1, -1):
        for x in range(W-1, -1, -1):
            d0 = dist_arr[y,x]
            c0 = col_arr[y,x]
            if d0 == INF:
                continue
            for (dy,dx,cost) in pass2_neighbors:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    old_d = dist_arr[ny,nx]
                    new_d = d0 + cost
                    if new_d < old_d:
                        dist_arr[ny,nx] = new_d
                        col_arr[ny,nx]  = c0

    return np.clip(col_arr, 0,255).astype(np.uint8)


################################################
# (9) ForwardWarpStereo => for splatting
################################################
class ForwardWarpStereo(nn.Module):
    def __init__(self, eps=1e-6, occlu_map=True):
        super().__init__()
        self.eps = eps
        self.occlu_map = occlu_map
        if forward_warp is None:
            raise ValueError("[ERROR] => forward_warp.py not found => can't splat.")
        self.fw= forward_warp()
        self.warp_exponent_base = 1.414

    def forward(self, im, disp, convergence=0.0):
        im= im.contiguous()
        disp= disp.contiguous()+ convergence
        wmap= disp- disp.min()
        wmap = (self.warp_exponent_base)**wmap

        flow_x= -disp.squeeze(1)
        flow_y= torch.zeros_like(flow_x)
        flow= torch.stack((flow_x, flow_y), dim=-1)

        res_accum= self.fw(im*wmap, flow)
        mask= self.fw(wmap, flow)
        mask.clamp_(min=self.eps)
        res= res_accum/ mask

        if not self.occlu_map:
            return res, None
        else:
            ones= torch.ones_like(disp)
            occ= self.fw(ones, flow)
            occ.clamp_(0,1)
            occ= 1.0- occ
            return res, occ


################################################
# (10A) create_anaglyph
################################################
def create_anaglyph(left_frame, right_frame):
    out= np.zeros_like(left_frame)
    out[...,0]= left_frame[...,0]
    out[...,1]= right_frame[...,1]
    out[...,2]= right_frame[...,2]
    return out


################################################
# (10A) DepthSplatting2x2
################################################
def DepthSplatting2x2(
    input_video_path,
    output_video_path,
    video_depth,
    depth_vis,
    max_disp,
    process_length,
    batch_size,
    convergence=0.0,
    dilate_h=4,
    dilate_v=1,
    blur_ksize=3,
    blur_sigma=2.0,
    keyframes_params=None,
    enable_interpolation=True,
    warp_exponent_base=1.414
):
    """
    2x2 mode: top-left is the original frame, top-right is the depth color,
    bottom-left is the holes mask, bottom-right is the right-eye view.
    All automatic or fixed cropping logic is removed. No final scaling applied.
    """
    import gc
    import numpy as np
    import torch
    from decord import VideoReader, cpu
    import cv2

    video_path_fixed = ensure_clean_video(input_video_path)
    vr = VideoReader(video_path_fixed, ctx=cpu(0))
    fps = vr.get_avg_fps()
    frames_all = vr[:].asnumpy().astype(np.float32)/255.0
    if process_length != -1 and process_length < len(frames_all):
        frames_all = frames_all[:process_length]
        video_depth = video_depth[:process_length]
        depth_vis = depth_vis[:process_length]
    total_frames = len(frames_all)
    if total_frames == 0:
        print(f"[WARN] => No frames => skipping => {input_video_path}")
        return

    H, W, _ = frames_all[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out_grid = cv2.VideoWriter(output_video_path, fourcc, fps, (W*2, H*2))

    anag_dir = os.path.join(os.path.dirname(output_video_path), "output_anaglyph")
    os.makedirs(anag_dir, exist_ok=True)
    anag_name = os.path.splitext(os.path.basename(output_video_path))[0] + "_anaglyph.mp4"
    anag_path = os.path.join(anag_dir, anag_name)
    out_anag = cv2.VideoWriter(anag_path, fourcc, fps, (W, H))

    print(f"[INFO] => Splat Mode (2x2) => {output_video_path}")
    print(f"[INFO] => Anaglyph => {anag_path} => size=({W}x{H})")

    if keyframes_params is None:
        keyframes_params = {}
    real_kf = {}
    for k, v in (keyframes_params.items()):
        try:
            kk = int(k)
            real_kf[kk] = v
        except:
            pass

    default_dict = {
        "disp_value": max_disp,
        "convergence": convergence,
        "brightness_value": 1.0,
        "gamma_value": 1.0,
        "dilate_h_value": dilate_h,
        "dilate_v_value": dilate_v,
        "blur_ksize_value": blur_ksize,
        "blur_sigma_value": blur_sigma
    }

    stereo = ForwardWarpStereo(occlu_map=True).cuda()
    stereo.warp_exponent_base = warp_exponent_base

    start = 0
    while start < total_frames:
        end = min(start + batch_size, total_frames)
        size_ = end - start
        bf = frames_all[start:end]
        bd = video_depth[start:end]
        bv = depth_vis[start:end]

        for i in range(size_):
            global_idx = start + i
            if enable_interpolation:
                local_params = interpolate_frame_params(real_kf, global_idx, default_dict)
            else:
                if global_idx in real_kf:
                    local_params = {**default_dict, **real_kf[global_idx]}
                else:
                    local_params = default_dict
            disp_val = local_params["disp_value"]
            conv_val = local_params["convergence"]
            bri_val = local_params["brightness_value"]
            gam_val = local_params["gamma_value"]
            dh_val = local_params["dilate_h_value"]
            dv_val = local_params["dilate_v_value"]
            bk_val = local_params["blur_ksize_value"]
            bs_val = local_params["blur_sigma_value"]

            bd[i] = np.clip(bd[i] * bri_val, 0, 1)
            bd[i] = bd[i] ** (1.0 / max(gam_val, 1e-9))
            bd[i] = np.clip(bd[i], 0, 1)
            bd[i] = apply_depth_preprocess(bd[i], dh_val, dv_val, bk_val, bs_val)
            bd[i] = (bd[i] * 2.0 - 1.0) * disp_val + conv_val

        left_t = torch.from_numpy(bf).permute(0, 3, 1, 2).float().cuda()
        disp_t = torch.from_numpy(bd).unsqueeze(1).float().cuda()
        with torch.no_grad():
            right_t, occ_t = stereo(left_t, disp_t, convergence=0.0)
        rn = right_t.cpu().permute(0, 2, 3, 1).numpy()  # shape [B,H,W,3]
        on = occ_t.cpu().permute(0, 2, 3, 1).numpy()    # shape [B,H,W,1]

        for j in range(size_):
            left_frame = bf[j]               # [H,W,3]
            depth_color = bv[j]             # [H,W,3]
            occ_1ch = on[j]                 # [H,W,1]
            occ_3ch = np.repeat(occ_1ch, 3, axis=-1)  # [H,W,3]
            right_frame = rn[j]             # [H,W,3]

            # Build 2x2
            top = np.concatenate([left_frame, depth_color], axis=1)   # horizontally
            bot = np.concatenate([occ_3ch,   right_frame], axis=1)
            grid_2x2 = np.concatenate([top, bot], axis=0)  # vertically => shape ?

            grid_255 = np.clip(grid_2x2*255, 0,255).astype(np.uint8)
            grid_bgr = cv2.cvtColor(grid_255, cv2.COLOR_RGB2BGR)
            out_grid.write(grid_bgr)

            # Anaglyph
            anag = create_anaglyph(left_frame, right_frame)
            anag_255 = np.clip(anag*255, 0,255).astype(np.uint8)
            anag_bgr = cv2.cvtColor(anag_255, cv2.COLOR_RGB2BGR)
            out_anag.write(anag_bgr)

        del left_t, disp_t, right_t, occ_t
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[INFO] => (2x2) frames {start}..{end - 1} => splatted")
        start= end

    out_grid.release()
    out_anag.release()
    print(f"[OK] => (2x2) => {output_video_path}")


################################################
# (10B) DepthSplattingTriple
################################################
def DepthSplattingTriple(
    input_video_path,
    output_base,
    video_depth,
    max_disp,
    process_length,
    batch_size,
    convergence=0.0,
    dilate_h=4,
    dilate_v=1,
    blur_ksize=3,
    blur_sigma=2.0,
    keyframes_params=None,
    enable_interpolation=True,
    warp_exponent_base=1.414
):
    """
    Triple mode: 
      - left video,
      - mask video (holes),
      - right video (final splatted).
    All references to cropping or final scaling have been removed.
    """
    import cv2
    import numpy as np
    import torch
    import gc
    from decord import VideoReader, cpu

    video_path_fixed = ensure_clean_video(input_video_path)
    vr= VideoReader(video_path_fixed, ctx=cpu(0))
    fps= vr.get_avg_fps()
    frames_all= vr[:].asnumpy().astype(np.float32)/255.0

    if process_length!=-1 and process_length< len(frames_all):
        frames_all = frames_all[:process_length]
        video_depth= video_depth[:process_length]

    total_frames= len(frames_all)
    if total_frames==0:
        print(f"[WARN] => No frames => skip => {input_video_path}")
        return

    H, W, _= frames_all[0].shape
    fourcc= cv2.VideoWriter_fourcc(*"mp4v")

    dir_   = os.path.dirname(output_base)
    base_  = os.path.basename(output_base)
    os.makedirs(dir_, exist_ok=True)

    out_spl_path = os.path.join(dir_, base_ + "_splatted.mp4")
    out_left_path= os.path.join(dir_, base_ + "_left.mp4")
    out_mask_path= os.path.join(dir_, base_ + "_mask.mp4")

    out_spl = cv2.VideoWriter(out_spl_path, fourcc, fps, (W,H))
    out_left= cv2.VideoWriter(out_left_path, fourcc, fps, (W,H))
    out_mask= cv2.VideoWriter(out_mask_path, fourcc, fps, (W,H))

    print(f"[INFO] => Splat Mode (triple) => right:   {out_spl_path}")
    print(f"[INFO] =>                        left:    {out_left_path}")
    print(f"[INFO] =>                        mask:    {out_mask_path}")

    if keyframes_params is None:
        keyframes_params={}
    real_kf={}
    for k,v in keyframes_params.items():
        try:
            kk= int(k)
            real_kf[kk]= v
        except:
            pass

    default_dict={
        "disp_value": max_disp,
        "convergence": convergence,
        "brightness_value": 1.0,
        "gamma_value": 1.0,
        "dilate_h_value": dilate_h,
        "dilate_v_value": dilate_v,
        "blur_ksize_value": blur_ksize,
        "blur_sigma_value": blur_sigma
    }

    stereo= ForwardWarpStereo(occlu_map=True).cuda()
    stereo.warp_exponent_base = warp_exponent_base

    start=0
    while start< total_frames:
        end= min(start+ batch_size, total_frames)
        size_ = end- start

        bf= frames_all[start:end]
        bd= video_depth[start:end]

        for i in range(size_):
            global_idx= start+ i
            if enable_interpolation:
                local_params= interpolate_frame_params(real_kf, global_idx, default_dict)
            else:
                if global_idx in real_kf:
                    local_params= {**default_dict, **real_kf[global_idx]}
                else:
                    local_params= default_dict

            disp_val= local_params["disp_value"]
            conv_val= local_params["convergence"]
            bri_val = local_params["brightness_value"]
            gam_val = local_params["gamma_value"]
            dh_val  = local_params["dilate_h_value"]
            dv_val  = local_params["dilate_v_value"]
            bk_val  = local_params["blur_ksize_value"]
            bs_val  = local_params["blur_sigma_value"]

            bd[i]= np.clip(bd[i]* bri_val, 0,1)
            bd[i]= bd[i]**(1.0/max(gam_val,1e-9))
            bd[i]= np.clip(bd[i],0,1)
            bd[i]= apply_depth_preprocess(bd[i], dh_val, dv_val, bk_val, bs_val)
            disp_ = (bd[i]*2.0 -1.0)* disp_val

            bd[i]  = disp_

        left_t= torch.from_numpy(bf).permute(0,3,1,2).float().cuda()
        disp_t= torch.from_numpy(bd).unsqueeze(1).float().cuda()
        with torch.no_grad():
            right_t, occ_t= stereo(left_t, disp_t, convergence=conv_val)

        rn= right_t.cpu().permute(0,2,3,1).numpy()  # [B,H,W,3]
        on= occ_t.cpu().permute(0,2,3,1).numpy()    # [B,H,W,1]

        for j in range(size_):
            left_bgr= (bf[j]*255).astype(np.uint8)
            left_bgr= cv2.cvtColor(left_bgr, cv2.COLOR_RGB2BGR)

            mask_01= on[j][:,:,0] 
            mask_bin = (mask_01>0.5).astype(np.uint8)*255
            mask_3ch = cv2.merge([mask_bin, mask_bin, mask_bin]) 

            right_255= np.clip(rn[j]*255,0,255).astype(np.uint8)
            right_bgr= cv2.cvtColor(right_255, cv2.COLOR_RGB2BGR)

            hole_bool = (mask_bin>127)
            filled_bgr = fast_inpaint_distance_transform(right_bgr, hole_bool)

            out_left.write(left_bgr)
            out_mask.write(mask_3ch)
            out_spl.write(filled_bgr)

        del left_t, disp_t, right_t, occ_t
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[INFO] => (triple) frames {start}..{end-1}")
        start= end

    out_spl.release()
    out_left.release()
    out_mask.release()
    print("[OK] => (triple) => 3 videos =>")
    print(f"   Right => {out_spl_path}")
    print(f"   Left  => {out_left_path}")
    print(f"   Mask  => {out_mask_path}")


################################################
# (11) Polylines => from numba import ...
################################################
from numba import njit, prange

@njit(parallel=True)
def apply_stereo_divergence_polylines(
    original_bgr: np.ndarray,
    normalized_depth: np.ndarray,
    disp_value: float,
    separation_px: float,
    stereo_offset_exponent: float,
    fill_technique: str = "polylines"
) -> np.ndarray:

    EPSILON = 1e-7
    if fill_technique == 'polylines_sharp':
        PIXEL_HALF_WIDTH = 0.45
    else:
        PIXEL_HALF_WIDTH = 0.0

    h, w, _ = original_bgr.shape
    derived_image = np.zeros_like(original_bgr)

    for row in prange(h):
        pt = np.zeros((2*w+2, 3), dtype=np.float64)
        pt_end = 0

        pt[pt_end] = [-w, 0.0, 0.0]  
        pt_end += 1

        for col in range(w):
            d_val = normalized_depth[row, col]
            offset = (d_val**stereo_offset_exponent) * disp_value
            coord_d = -offset
            coord_x = (col + 0.5) + coord_d - separation_px

            if PIXEL_HALF_WIDTH < EPSILON:
                pt[pt_end] = [coord_x, abs(coord_d), col]
                pt_end += 1
            else:
                pt[pt_end]   = [coord_x - PIXEL_HALF_WIDTH, abs(coord_d), col]
                pt[pt_end+1] = [coord_x + PIXEL_HALF_WIDTH, abs(coord_d), col]
                pt_end += 2

        pt[pt_end] = [2*w, 0.0, w-1]
        pt_end += 1

        # insertion sort by x
        for i in range(1, pt_end):
            j = i - 1
            key_pt = pt[i].copy()
            while j >= 0 and pt[j,0] > key_pt[0]:
                pt[j+1] = pt[j]
                j -= 1
            pt[j+1] = key_pt

        idx_pt = 0
        for out_col in range(w):
            color_acum = np.zeros(3, dtype=np.float64)
            while idx_pt < pt_end-1 and pt[idx_pt+1,0] < out_col:
                idx_pt += 1

            x_start = float(out_col)
            x_end   = float(out_col+1)

            local_i = idx_pt
            while local_i < pt_end-1:
                xA = pt[local_i, 0]
                xB = pt[local_i+1,0]
                if xA >= x_end:
                    break
                if xB <= x_start:
                    local_i += 1
                    continue

                seg_start = max(x_start, xA)
                seg_end   = min(x_end,   xB)
                length_ = seg_end - seg_start
                if length_ <= EPSILON:
                    local_i += 1
                    continue

                cA = pt[local_i, 1]
                cB = pt[local_i+1,1]
                colA = int(pt[local_i, 2])
                colB = int(pt[local_i+1, 2])

                mid_x = 0.5*(seg_start + seg_end)
                frac  = 0.0
                if abs(xB - xA) > EPSILON:
                    frac = (mid_x - xA)/(xB - xA)

                if colA==colB:
                    color_acum += original_bgr[row, colA]* length_
                else:
                    colorA = original_bgr[row, colA]
                    colorB = original_bgr[row, colB]
                    color_mix = (1.0 - frac)* colorA + frac* colorB
                    color_acum += color_mix* length_

                local_i += 1

            derived_image[row, out_col] = np.clip(color_acum, 0,255).astype(np.uint8)

    return derived_image


def generate_stereo_video_polylines(
    input_video_path,
    output_video_path,
    video_depth,
    process_length,
    batch_size,
    keyframes_params=None,
    enable_interpolation=True,
    base_default_params=None 
):
    from decord import VideoReader, cpu
    import os
    import cv2
    import numpy as np
    import subprocess
    import imageio_ffmpeg

    if base_default_params is None:
        base_default_params = {}

    print("[INFO] => generate_stereo_video_polylines called.")
    vid_fixed = ensure_clean_video(input_video_path)
    vr = VideoReader(vid_fixed, ctx=cpu(0))
    fps= vr.get_avg_fps()

    frames_all = vr[:].asnumpy().astype(np.float32)/255.0
    if process_length!=-1 and process_length< len(frames_all):
        frames_all = frames_all[:process_length]
        video_depth= video_depth[:process_length]

    total_frames= len(frames_all)
    if total_frames==0:
        print("[WARN] => No frames => skip.")
        return

    H, W, _= frames_all[0].shape

    default_dict = {
            "disp_value": base_default_params.get("disp_value", 20.0),
            "convergence": base_default_params.get("convergence", 0.0),
            "brightness_value": base_default_params.get("brightness_value", 1.0),
            "gamma_value": base_default_params.get("gamma_value", 1.0),
            "dilate_h_value": base_default_params.get("dilate_h_value", 4),
            "dilate_v_value": base_default_params.get("dilate_v_value", 1),
            "blur_ksize_value": base_default_params.get("blur_ksize_value", 3),
            "blur_sigma_value": base_default_params.get("blur_sigma_value", 2.0),
            "stereo_offset_exponent_value": base_default_params.get("stereo_offset_exponent_value", 1.0),
            "separation_px_value": base_default_params.get("separation_px_value", 0.0),
            "fill_technique_value": base_default_params.get("fill_technique_value", "polylines"),
            "sbs_mode_value": base_default_params.get("sbs_mode_value", "half"),
            "encoder_value": base_default_params.get("encoder_value", "x264")
        }

    if keyframes_params is None:
        keyframes_params = {}
    real_kf={}
    for k,v in keyframes_params.items():
        try:
            real_kf[int(k)] = v
        except:
            pass

    if total_frames>0:
        first_params = interpolate_frame_params(real_kf, 0, default_dict)
        sbs_mode  = first_params["sbs_mode_value"]
        encoder   = first_params["encoder_value"]
    else:
        sbs_mode = "half"
        encoder  = "x264"

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    tmp_output = output_video_path
    if encoder=="x265":
        base_, ext_ = os.path.splitext(output_video_path)
        tmp_output = base_ + "_temp_x264.mp4"

    if sbs_mode=="full":
        out_w= W*2
        out_h= H
    else:
        out_w= W
        out_h= H

    fourcc= cv2.VideoWriter_fourcc(*"avc1")
    out_vid= cv2.VideoWriter(tmp_output, fourcc, fps, (out_w, out_h))

    start=0
    while start< total_frames:
        end= min(start+ batch_size, total_frames)
        subsize= end - start

        bf= frames_all[start:end].copy()
        bd= video_depth[start:end].copy()

        for i in range(subsize):
            frame_idx= start + i
            if enable_interpolation:
                local_p = interpolate_frame_params(real_kf, frame_idx, default_dict)
            else:
                if frame_idx in real_kf:
                    local_p = {**default_dict, **real_kf[frame_idx]}
                else:
                    local_p = default_dict

            disp_val   = float(local_p["disp_value"])
            conv_val   = float(local_p["convergence"])
            bri_val    = float(local_p["brightness_value"])
            gam_val    = float(local_p["gamma_value"])
            dh_val     = int(local_p["dilate_h_value"])
            dv_val     = int(local_p["dilate_v_value"])
            bk_val     = int(local_p["blur_ksize_value"])
            bs_val     = float(local_p["blur_sigma_value"])
            s_off_exp  = float(local_p["stereo_offset_exponent_value"])
            sep_px     = float(local_p["separation_px_value"])
            fill_tech  = str(local_p["fill_technique_value"])

            bd[i] = np.clip(bd[i]*bri_val, 0,1)
            bd[i] = bd[i]**(1.0/max(gam_val,1e-9))
            bd[i] = np.clip(bd[i], 0,1)
            bd[i] = apply_depth_preprocess(bd[i], dh_val, dv_val, bk_val, bs_val)

            disp_frame = (bd[i]*2.0 -1.0)* disp_val + conv_val
            d_min = disp_frame.min()
            d_max = disp_frame.max()
            if (d_max - d_min) < 1e-9:
                disp_norm = np.zeros_like(disp_frame, dtype=np.float32)
            else:
                disp_norm = (disp_frame - d_min)/(d_max - d_min)

            left_bgr = (bf[i]*255).astype(np.uint8)
            left_bgr = cv2.cvtColor(left_bgr, cv2.COLOR_RGB2BGR)
            right_bgr = apply_stereo_divergence_polylines(
                left_bgr, disp_norm, disp_val, sep_px, s_off_exp, fill_tech
            )

            if sbs_mode=="full":
                combo_bgr = np.concatenate([left_bgr, right_bgr], axis=1)
            else:
                half_w = W//2
                left_half  = cv2.resize(left_bgr,  (half_w, H), interpolation=cv2.INTER_AREA)
                right_half = cv2.resize(right_bgr, (half_w, H), interpolation=cv2.INTER_AREA)
                combo_bgr = np.concatenate([left_half, right_half], axis=1)

            out_vid.write(combo_bgr)

        start= end
        print(f"[INFO] => polylines frames => {start}/{total_frames}")

    out_vid.release()
    print(f"[OK] => Freed polylines writer => {tmp_output}")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if encoder == "x265":
        cmd = [
            "ffmpeg_exe", "-y",
            "-i", tmp_output,
            "-c:v", "libx265",
            "-preset", "veryslow",
            "-crf", "14",
            "-c:a", "copy",
            output_video_path
        ]
        print("[INFO] => x265 =>", cmd)
        ret= subprocess.run(cmd, capture_output=True)
        if ret.returncode!=0:
            print("[ERROR] => x265 fail => leaving temp file")
        else:
            if os.path.exists(tmp_output):
                os.remove(tmp_output)

    print(f"[DONE] => polylines => {output_video_path}")
    if os.path.exists(output_video_path):
        sz_ = os.path.getsize(output_video_path)
        print(f"[INFO] => final SBS file => {output_video_path}, size={sz_} bytes")
    else:
        print(f"[ERROR] => final SBS file not found => {output_video_path}")


################################################
# (13) MAIN 
################################################
import OpenEXR
import Imath
import array

def float32_to_half_bytes(frame_2d):
    H, W = frame_2d.shape
    arr_ = array.array('H')
    for val in frame_2d.flatten():
        half_bits = Imath.FloatToHalf(val)
        arr_.append(half_bits)
    return arr_.tobytes()

def save_exr_sequence_depth(depth_array, out_dir, half_float=False):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    T, H, W = depth_array.shape
    for i in range(T):
        frame = depth_array[i]
        header = OpenEXR.Header(W, H)
        if half_float:
            pix_type = Imath.PixelType(Imath.PixelType.HALF)
        else:
            pix_type = Imath.PixelType(Imath.PixelType.FLOAT)
        header["channels"] = {"Z": Imath.Channel(pix_type)}

        if half_float:
            frame_16 = np.float16(frame)
            bits_16  = frame_16.view(np.uint16)
            arr_16   = array.array('H', bits_16.flatten().tolist())
            packed_data = arr_16.tobytes()
        else:
            packed_data = frame.tobytes()

        out_exr = os.path.join(out_dir, f"frame_{i:04d}.exr")
        of = OpenEXR.OutputFile(out_exr, header)
        of.writePixels({"Z": packed_data})
        of.close()

        print(f"[EXR] => wrote {out_exr}")


def main(
    input_source_clips="./input_source_clips",
    input_depth_maps="./pre_rendered_depth",
    output_splatted="./output_splatted",
    single_video=None,
    warp_exponent_base=1.414,
    depth_output_mode="mp4",
    unet_path="./weights/DepthCrafter",
    pre_trained_path="./weights/stable-video-diffusion-img2vid-xt-1-1",
    process_length=-1,
    num_denoising_steps=8,
    guidance_scale=1.2,
    window_size=70,
    overlap=25,
    seed=42,
    max_res=1024,
    pad_to_multiple_of=64,
    max_disp=20.0,
    batch_size=10,
    convergence=0.0,
    dilate_h=4,
    dilate_v=1,
    blur_ksize=3,
    blur_sigma=2.0,
    orig_brightness_value=1.0,
    orig_gamma_value=1.0,
    depth_brightness_value=1.0,
    depth_gamma_value=1.0,
    depth_only=False,
    keyframes_json="",
    enable_interpolation=True,
    splat_mode="2x2",
    generate_stereo=False,
    encoder="x264",
    sbs_mode="half",
    stereo_offset_exponent=1.0,
    separation_px=0.0,
    fill_technique="polylines"
):
    print(f"[INFO MAIN] => generate_stereo={generate_stereo}, splat_mode={splat_mode}")

    kf_data = {}
    if keyframes_json and os.path.isfile(keyframes_json):
        try:
            with open(keyframes_json, "r", encoding="utf-8") as f:
                kf_data = json.load(f)
            print(f"[INFO] => Keyframes loaded => {keyframes_json}")
        except Exception as e:
            print(f"[WARN] => Could not parse keyframes => {e}")
            kf_data = {}

    if single_video and os.path.isfile(single_video):
        vids = [single_video]
    else:
        exts = ("*.mp4", "*.mov", "*.avi", "*.mkv")
        vids = []
        for e_ in exts:
            vids.extend(glob.glob(os.path.join(input_source_clips, e_)))
        vids = sorted(vids)
        if not vids:
            print("[WARN] => No input videos found => exiting.")
            return

    depthcrafter_demo = None
    if (DepthCrafterPipeline and DiffusersUNetSpatioTemporalConditionModelDepthCrafter
        and unet_path and pre_trained_path):
        try:
            depthcrafter_demo = DepthCrafterDemo(unet_path, pre_trained_path)
        except Exception as e:
            print("[WARN] => Can't init DepthCrafter =>", e)
            depthcrafter_demo = None
    else:
        print("[WARN] => DepthCrafter not available => only pre-rendered usage")

    os.makedirs(output_splatted, exist_ok=True)
    os.makedirs(input_depth_maps, exist_ok=True)

    for vp in vids:
        bn = os.path.splitext(os.path.basename(vp))[0]
        print(f"\n[INFO] => Processing => {bn}")

        need_adj = (abs(orig_brightness_value - 1.0) > 1e-5 or abs(orig_gamma_value - 1.0) > 1e-5)
        reencoded_path = None
        dummy_out_depthvis = None
        if need_adj:
            reencoded_path = "temp_reencode_original.mp4"
            if os.path.exists(reencoded_path):
                os.remove(reencoded_path)
            print(f"[INFO] => Re-encoding original => bri={orig_brightness_value}, gamma={orig_gamma_value}")
            reencode_brightness_gamma(
                vp, reencoded_path,
                brightness=orig_brightness_value,
                gamma=orig_gamma_value
            )
            video_for_depth = ensure_clean_video(reencoded_path)
        else:
            video_for_depth = ensure_clean_video(vp)

        dp = os.path.join(input_depth_maps, bn + "_depth.mp4")
        if depth_only:
            if depthcrafter_demo is not None:
                dummy_out_depthvis = os.path.join(output_splatted, bn + "_dummy_depthvis.mp4")
                print("[INFO] => depth_only => generating fresh depth =>", bn)
                vd, dv, fps_depth = depthcrafter_demo.infer(
                    input_video_path=video_for_depth,
                    output_video_path=dummy_out_depthvis,
                    process_length=process_length,
                    num_denoising_steps=num_denoising_steps,
                    guidance_scale=guidance_scale,
                    window_size=window_size,
                    overlap=overlap,
                    seed=seed,
                    max_res=max_res,
                    pad_to_multiple_of=pad_to_multiple_of
                )
                from decord import VideoReader, cpu
                if fps_depth is None:
                    fps_depth = VideoReader(video_for_depth, ctx=cpu(0)).get_avg_fps()
                if depth_output_mode == "mp4":
                    save_grayscale_depth_video(vd, dp, fps_depth)
                elif depth_output_mode == "exr16":
                    out_exr_dir = dp.replace(".mp4","_exrseq")
                    save_exr_sequence_depth(vd, out_exr_dir, half_float=True)
                elif depth_output_mode == "exr32":
                    out_exr_dir = dp.replace(".mp4","_exrseq")
                    save_exr_sequence_depth(vd, out_exr_dir, half_float=False)
                else:
                    save_grayscale_depth_video(vd, dp, fps_depth)
            else:
                print("[WARN] => DepthCrafter not available => cannot generate => skipping =>", bn)

            if reencoded_path and os.path.exists(reencoded_path):
                try:
                    os.remove(reencoded_path)
                except:
                    pass
            if dummy_out_depthvis and os.path.exists(dummy_out_depthvis):
                try:
                    os.remove(dummy_out_depthvis)
                except:
                    pass

            print("[INFO] => depth_only => done => skipping splatting.")
            continue
        else:
            if os.path.exists(dp):
                print("[INFO] => Found pre-rendered depth =>", dp)
                vd, dv = load_pre_rendered_depth(dp, process_length=process_length)
            else:
                if depthcrafter_demo is None:
                    print("[WARN] => No pipeline => cannot generate => skipping =>", bn)
                    if reencoded_path and os.path.exists(reencoded_path):
                        try:
                            os.remove(reencoded_path)
                        except:
                            pass
                    continue
                dummy_out_depthvis = os.path.join(output_splatted, bn + "_dummy_depthvis.mp4")
                print("[INFO] => Generating new depth =>", dp)
                vd, dv, fps_depth = depthcrafter_demo.infer(
                    input_video_path=video_for_depth,
                    output_video_path=dummy_out_depthvis,
                    process_length=process_length,
                    num_denoising_steps=num_denoising_steps,
                    guidance_scale=guidance_scale,
                    window_size=window_size,
                    overlap=overlap,
                    seed=seed,
                    max_res=max_res,
                    pad_to_multiple_of=pad_to_multiple_of
                )
                from decord import VideoReader, cpu
                if fps_depth is None:
                    fps_depth = VideoReader(video_for_depth, ctx=cpu(0)).get_avg_fps()
                save_grayscale_depth_video(vd, dp, fps_depth)

            if reencoded_path and os.path.exists(reencoded_path):
                try:
                    os.remove(reencoded_path)
                except:
                    pass
            if dummy_out_depthvis and os.path.exists(dummy_out_depthvis):
                try:
                    os.remove(dummy_out_depthvis)
                except:
                    pass

        if abs(depth_brightness_value - 1.0) > 1e-5 or abs(depth_gamma_value - 1.0) > 1e-5:
            print(f"[INFO] => Adjusting final depth => bri={depth_brightness_value}, gamma={depth_gamma_value}")
            vd = np.clip(vd * depth_brightness_value, 0, 1)
            vd = vd ** (1.0 / max(depth_gamma_value, 1e-9))
            vd = np.clip(vd, 0, 1)
            dv = chunked_vis_sequence_depth(vd)

        local_kf = {}
        if bn in kf_data and isinstance(kf_data[bn], dict):
            local_kf = kf_data[bn]

        if generate_stereo:
            default_dict_polylines = {
                "disp_value": max_disp,
                "convergence": convergence,
                "brightness_value": depth_brightness_value,
                "gamma_value": depth_gamma_value,
                "dilate_h_value": dilate_h,
                "dilate_v_value": dilate_v,
                "blur_ksize_value": blur_ksize,
                "blur_sigma_value": blur_sigma,
                "stereo_offset_exponent_value": stereo_offset_exponent,
                "separation_px_value": separation_px,
                "fill_technique_value": fill_technique,
                "sbs_mode_value": sbs_mode,
                "encoder_value": encoder
            }
            out_final = os.path.join(output_splatted, bn + "_stereo_sbs.mp4")
            print(f"[INFO] => generate_stereo => polylines => {out_final}")

            generate_stereo_video_polylines(
                input_video_path=vp,
                output_video_path=out_final,
                video_depth=vd,
                process_length=process_length,
                batch_size=batch_size,
                keyframes_params=local_kf,
                enable_interpolation=enable_interpolation,
                base_default_params=default_dict_polylines
            )
            print(f"[DONE] => polylines => {out_final}")
            continue

        if splat_mode == "2x2":
            out_final = os.path.join(output_splatted, bn + "_splatted.mp4")
            DepthSplatting2x2(
                input_video_path=vp,
                output_video_path=out_final,
                video_depth=vd,
                depth_vis=dv,
                max_disp=max_disp,
                process_length=process_length,
                batch_size=batch_size,
                convergence=convergence,
                dilate_h=dilate_h,
                dilate_v=dilate_v,
                blur_ksize=blur_ksize,
                blur_sigma=blur_sigma,
                keyframes_params=local_kf,
                enable_interpolation=enable_interpolation,
                warp_exponent_base=warp_exponent_base
            )
            print(f"[DONE] => (2x2) => {out_final}")
        elif splat_mode == "triple":
            out_final = os.path.join(output_splatted, bn)
            DepthSplattingTriple(
                input_video_path=vp,
                output_base=out_final,
                video_depth=vd,
                max_disp=max_disp,
                process_length=process_length,
                batch_size=batch_size,
                convergence=convergence,
                dilate_h=dilate_h,
                dilate_v=dilate_v,
                blur_ksize=blur_ksize,
                blur_sigma=blur_sigma,
                keyframes_params=local_kf,
                enable_interpolation=enable_interpolation,
                warp_exponent_base=warp_exponent_base
            )
            print(f"[DONE] => (triple) => base= {out_final}")
        else:
            print(f"[ERROR] => Unknown splat_mode='{splat_mode}'. Use '2x2' or 'triple'.")

    print("\n==> All done.")


if __name__=="__main__":
    Fire(main)
