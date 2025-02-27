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

try:
    from dependency.DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
    from dependency.DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
except ImportError:
    DepthCrafterPipeline = None
    DiffusersUNetSpatioTemporalConditionModelDepthCrafter = None

try:
    from Forward_Warp import forward_warp
except ImportError:
    forward_warp = None


def ensure_clean_video(video_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        if num_frames == 0:
            raise ValueError("Video has 0 frames. Possibly corrupted.")
        return video_path
    except Exception as e:
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
        ret = subprocess.run(cmd, capture_output=True)
        if ret.returncode != 0:
            return video_path
        original_backup = f"{video_path}.backup"
        try:
            os.rename(video_path, original_backup)
            os.rename(tmp_name, video_path)
            os.remove(original_backup)
        except Exception:
            if os.path.exists(original_backup):
                os.rename(original_backup, video_path)
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        return video_path


def reencode_brightness_gamma(in_path, out_path, brightness=1.0, gamma=1.0):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"Could not open video: {in_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    bri_eq = brightness - 1.0
    bri_eq = max(-1.0, min(1.0, bri_eq))
    gamma = max(0.1, min(10.0, gamma))
    idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_float = frame_bgr.astype(np.float32) / 255.0
        frame_float = np.clip(frame_float + bri_eq, 0, 1)
        if abs(gamma - 1.0) > 1e-4:
            frame_float = np.power(frame_float, gamma)
        frame_float = np.clip(frame_float, 0, 1)
        frame_u8 = (frame_float * 255.0).astype(np.uint8)
        out.write(frame_u8)
        idx += 1
    cap.release()
    out.release()


def chunked_vis_sequence_depth(depth_array, chunk_size=50):
    import cv2
    T, H, W = depth_array.shape
    out = []
    start = 0
    while start < T:
        end = min(start + chunk_size, T)
        chunk = depth_array[start:end]
        chunk_c = []
        for f_ in chunk:
            f255 = (f_ * 255).astype(np.uint8)
            c_ = cv2.applyColorMap(f255, cv2.COLORMAP_JET)
            chunk_c.append(c_.astype(np.float32) / 255.0)
        chunk_c = np.stack(chunk_c, axis=0)
        out.append(chunk_c)
        start = end
    return np.concatenate(out, axis=0)


def save_grayscale_depth_video(depth_data, out_path, fps):
    import cv2
    T, H, W = depth_data.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    outv = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for i in range(T):
        f_ = depth_data[i]
        f255 = (f_ * 255).astype(np.uint8)
        f3 = cv2.cvtColor(f255, cv2.COLOR_GRAY2BGR)
        outv.write(f3)
    outv.release()


def load_pre_rendered_depth(depth_video_path, process_length=-1):
    import cv2
    vr = VideoReader(depth_video_path, ctx=cpu(0))
    frames = vr[:].asnumpy().astype(np.float32) / 255.0
    if process_length != -1 and process_length < len(frames):
        frames = frames[:process_length]
    vd = frames.mean(axis=-1)
    vd = np.clip(vd, 0, 1)
    dv_list = []
    for f_ in vd:
        c_ = cv2.applyColorMap((f_ * 255).astype(np.uint8), cv2.COLORMAP_JET)
        dv_list.append(c_.astype(np.float32) / 255.0)
    dv = np.stack(dv_list, axis=0)
    return vd, dv


class DepthCrafterDemo:
    def __init__(self, unet_path, pre_trained_path, cpu_offload="model"):
        if DepthCrafterPipeline is None or DiffusersUNetSpatioTemporalConditionModelDepthCrafter is None:
            raise ValueError("DepthCrafter not available.")
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
        )
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_trained_path, unet=unet, torch_dtype=torch.float16, variant="fp16"
        )
        if cpu_offload == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        elif cpu_offload == "model":
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
        vr = VideoReader(input_video_path, ctx=cpu(0))
        orig_len = len(vr)
        oh, ow = vr.get_batch([0]).shape[1:3]
        fps = vr.get_avg_fps()
        frames_all = vr[:].asnumpy().astype(np.float32) / 255.0
        if process_length != -1 and process_length < orig_len:
            frames_all = frames_all[:process_length]
        T = len(frames_all)
        torch.manual_seed(seed)

        newh, neww = oh, ow
        if max(oh, ow) > max_res > 0:
            scale = max_res / max(oh, ow)
            newh = int(round(oh * scale))
            neww = int(round(ow * scale))

        import math
        nH = int(math.ceil(newh / pad_to_multiple_of) * pad_to_multiple_of)
        nW = int(math.ceil(neww / pad_to_multiple_of) * pad_to_multiple_of)
        phT = (nH - newh) // 2
        phB = nH - newh - phT
        pwL = (nW - neww) // 2
        pwR = nW - neww - pwL

        frames_padded = []
        for fr_ in frames_all:
            if (newh != oh) or (neww != ow):
                fr_ = cv2.resize(fr_, (neww, newh), interpolation=cv2.INTER_AREA)
            can = np.zeros((nH, nW, 3), dtype=np.float32)
            can[phT:phT+newh, pwL:pwL+neww] = fr_
            frames_padded.append(can)
        frames_padded = np.stack(frames_padded, axis=0)

        effective_window = window_size - overlap
        self._n_chunks_total = max(1, (T + (effective_window - 1)) // effective_window)
        self._current_chunk = 0
        self._num_inference_steps = num_denoising_steps

        def global_callback(step: int, timestep: int, latents: torch.FloatTensor):
            pass

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

        depth_1c = out.mean(axis=-1)
        mn, mx = depth_1c.min(), depth_1c.max()
        depth_norm = (depth_1c - mn) / (mx - mn + 1e-8)
        depth_unpad = depth_norm[:, phT:phT+newh, pwL:pwL+neww]

        if (newh != oh) or (neww != ow):
            import torch.nn.functional as F
            t_ = torch.from_numpy(depth_unpad).unsqueeze(1).float().cuda()
            t_ = F.interpolate(t_, size=(oh, ow), mode="bilinear", align_corners=False)
            depth_final = t_[:, 0].cpu().numpy()
        else:
            depth_final = depth_unpad

        depth_vis = chunked_vis_sequence_depth(depth_final)
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            outv = cv2.VideoWriter(output_video_path, fourcc, fps, (ow, oh))
            for i in range(T):
                f_ = depth_vis[i]
                f255 = np.clip(f_ * 255, 0, 255).astype(np.uint8)
                frBGR = cv2.cvtColor(f255, cv2.COLOR_RGB2BGR)
                outv.write(frBGR)
            outv.release()

        return depth_final, depth_vis, fps


def interpolate_frame_params(kframes, frame_idx, default_params):
    if not kframes:
        return default_params
    frames_sorted = sorted(kframes.keys())
    if frame_idx <= frames_sorted[0]:
        return {p: kframes[frames_sorted[0]].get(p, default_params[p]) for p in default_params}
    if frame_idx >= frames_sorted[-1]:
        return {p: kframes[frames_sorted[-1]].get(p, default_params[p]) for p in default_params}
    left_f = frames_sorted[0]
    right_f = frames_sorted[-1]
    for i in range(len(frames_sorted) - 1):
        f0 = frames_sorted[i]
        f1 = frames_sorted[i + 1]
        if f0 <= frame_idx <= f1:
            left_f = f0
            right_f = f1
            break
    if left_f == right_f:
        return {p: kframes[left_f].get(p, default_params[p]) for p in default_params}
    ratio = (frame_idx - left_f) / float(right_f - left_f + 1e-9)
    result = {}
    for p in default_params:
        lv = kframes[left_f].get(p, default_params[p])
        rv = kframes[right_f].get(p, default_params[p])
        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
            result[p] = lv + (rv - lv) * ratio
        else:
            result[p] = lv
    return result


def apply_depth_preprocess(depth_frame_01, dilate_h=0, dilate_v=0, blur_ksize=3, blur_sigma=2.0):
    import cv2
    dh_i = max(0, int(round(dilate_h)))
    dv_i = max(0, int(round(dilate_v)))
    if dh_i > 0 or dv_i > 0:
        depth_u8 = (depth_frame_01 * 255).astype(np.uint8)
        kernel = np.ones((dv_i, dh_i), np.uint8)
        depth_u8 = cv2.dilate(depth_u8, kernel, iterations=1)
        depth_frame_01 = depth_u8.astype(np.float32) / 255.0
    bk_i = int(round(blur_ksize))
    if bk_i < 1:
        bk_i = 1
    if bk_i % 2 == 0:
        bk_i += 1
    if bk_i > 1:
        depth_255 = (depth_frame_01 * 255).astype(np.float32)
        depth_255 = cv2.GaussianBlur(depth_255, (bk_i, bk_i), sigmaX=blur_sigma)
        depth_frame_01 = np.clip(depth_255 / 255.0, 0, 1)
    return depth_frame_01


def inpaint_small_holes_inplace(rgb_image, occ_1ch, max_hole_area=100, inpaintRadius=3):
    import cv2
    import numpy as np
    H, W, _ = rgb_image.shape
    mask_bin = (occ_1ch[..., 0] > 0.5).astype(np.uint8)
    num_labels, labels_im = cv2.connectedComponents(mask_bin, connectivity=8)
    small_holes_mask = np.zeros((H, W), dtype=np.uint8)
    for lbl in range(1, num_labels):
        area = np.sum(labels_im == lbl)
        if area < max_hole_area:
            small_holes_mask[labels_im == lbl] = 255
    if np.any(small_holes_mask > 0):
        rgb_u8 = np.clip(rgb_image, 0, 255).astype(np.uint8)
        dst = cv2.inpaint(rgb_u8, small_holes_mask, inpaintRadius, cv2.INPAINT_NS)
        rgb_image[...] = dst.astype(np.float32)
        labels_im[small_holes_mask > 0] = 0
    new_mask = (labels_im > 0).astype(np.float32)[..., None]
    return rgb_image, new_mask


class ForwardWarpStereo(nn.Module):
    def __init__(self, warp_interpolation_mode=1, eps=1e-6, occlu_map=True):
        super().__init__()
        self.eps = eps
        self.occlu_map = occlu_map
        self.warp_interpolation_mode = warp_interpolation_mode
        if forward_warp is None:
            raise ValueError("forward_warp.py is missing. Cannot perform splatting.")
        self.fw = forward_warp()
        self.warp_exponent_base = 1.414

    def forward(self, im, disp, convergence=0.0):
        im = im.contiguous()
        disp = disp.contiguous() + convergence
        wmap = disp - disp.min()
        wmap = (self.warp_exponent_base) ** wmap
        flow_x = -disp.squeeze(1)
        flow_y = torch.zeros_like(flow_x)
        flow = torch.stack((flow_x, flow_y), dim=-1)
        res_accum = self.fw(im * wmap, flow)
        mask = self.fw(wmap, flow)
        mask.clamp_(min=self.eps)
        res = res_accum / mask
        if not self.occlu_map:
            return res, None
        else:
            ones = torch.ones_like(disp)
            occ = self.fw(ones, flow)
            occ.clamp_(0, 1)
            occ = 1.0 - occ
            return res, occ


def create_anaglyph(left_frame, right_frame):
    out = np.zeros_like(left_frame)
    out[..., 0] = left_frame[..., 0]
    out[..., 1] = right_frame[..., 1]
    out[..., 2] = right_frame[..., 2]
    return out


def symmetrical_crop_and_upscale_subimage(subimg_rgb, crop_border_px):
    import cv2
    h, w, _ = subimg_rgb.shape
    if crop_border_px <= 0:
        return subimg_rgb
    if crop_border_px * 2 >= h or crop_border_px * 2 >= w:
        return subimg_rgb
    top = crop_border_px
    left = crop_border_px
    bottom = h - crop_border_px
    right = w - crop_border_px
    cropped = subimg_rgb[top:bottom, left:right]
    final = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)
    return final


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
    warp_exponent_base=1.414,
    warp_interpolation_mode=1,
    max_small_hole_area=100,
    inpaintRadius=3,
    orig_brightness_value=1.0,
    orig_gamma_value=1.0,
    crop_border_px=0
):
    import numpy as np
    import torch
    import cv2
    video_path_fixed = ensure_clean_video(input_video_path)
    vr = VideoReader(video_path_fixed, ctx=cpu(0))
    fps = vr.get_avg_fps()
    frames_all = vr[:].asnumpy().astype(np.float32) / 255.0
    if process_length != -1 and process_length < len(frames_all):
        frames_all = frames_all[:process_length]
        video_depth = video_depth[:process_length]
        depth_vis = depth_vis[:process_length]
    total_frames = len(frames_all)
    if total_frames == 0:
        return

    H, W, _ = frames_all[0].shape
    out_w = W * 2
    out_h = H * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out_grid = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

    anag_dir = os.path.join(os.path.dirname(output_video_path), "output_anaglyph")
    os.makedirs(anag_dir, exist_ok=True)
    anag_name = os.path.splitext(os.path.basename(output_video_path))[0] + "_anaglyph.mp4"
    anag_path = os.path.join(anag_dir, anag_name)
    out_anag = cv2.VideoWriter(anag_path, fourcc, fps, (W, H))

    if keyframes_params is None:
        keyframes_params = {}
    real_kf = {}
    for k, v in keyframes_params.items():
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
        "blur_sigma_value": blur_sigma,
        "left_brightness_value": orig_brightness_value,
        "left_gamma_value": orig_gamma_value
    }

    stereo = ForwardWarpStereo(warp_interpolation_mode=warp_interpolation_mode, occlu_map=True).cuda()
    stereo.warp_exponent_base = warp_exponent_base

    start = 0
    while start < total_frames:
        end = min(start + batch_size, total_frames)
        sub_batch = end - start
        bf = frames_all[start:end]
        bd = video_depth[start:end]
        bv = depth_vis[start:end]

        for i in range(sub_batch):
            frame_idx = start + i
            if enable_interpolation:
                local_params = interpolate_frame_params(real_kf, frame_idx, default_dict)
            else:
                if frame_idx in real_kf:
                    local_params = {**default_dict, **real_kf[frame_idx]}
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
            lbri_val = local_params["left_brightness_value"]
            lgam_val = local_params["left_gamma_value"]

            bf[i] = np.clip(bf[i] * lbri_val, 0, 1)
            bf[i] = bf[i] ** (1.0 / max(lgam_val, 1e-9))
            bf[i] = np.clip(bf[i], 0, 1)

            bd[i] = np.clip(bd[i] * bri_val, 0, 1)
            bd[i] = bd[i] ** (1.0 / max(gam_val, 1e-9))
            bd[i] = np.clip(bd[i], 0, 1)
            bd[i] = apply_depth_preprocess(bd[i], dh_val, dv_val, bk_val, bs_val)
            bd[i] = (bd[i] * 2.0 - 1.0) * disp_val + conv_val

        left_t = torch.from_numpy(bf).permute(0, 3, 1, 2).float().cuda()
        disp_t = torch.from_numpy(bd).unsqueeze(1).float().cuda()
        with torch.no_grad():
            right_t, occ_t = stereo(left_t, disp_t, convergence=0.0)
        rn = right_t.cpu().permute(0, 2, 3, 1).numpy()
        on = occ_t.cpu().permute(0, 2, 3, 1).numpy()

        for j in range(sub_batch):
            left_frame = bf[j]
            depth_frame = bv[j]
            occ_1ch = on[j]
            right_frame = rn[j]

            right_u8 = np.clip(right_frame * 255, 0, 255).astype(np.float32)
            updated_rf, updated_occ = inpaint_small_holes_inplace(
                right_u8, occ_1ch, max_hole_area=max_small_hole_area, inpaintRadius=inpaintRadius
            )
            final_right = np.clip(updated_rf, 0, 255).astype(np.uint8) / 255.0

            hole_mask = (updated_occ[..., 0] * 255).astype(np.uint8)
            mask_3ch = cv2.merge([hole_mask, hole_mask, hole_mask])

            left_255 = np.clip(left_frame * 255, 0, 255).astype(np.uint8)
            depth_255 = np.clip(depth_frame * 255, 0, 255).astype(np.uint8)
            mask_255 = mask_3ch
            right_255 = np.clip(final_right * 255, 0, 255).astype(np.uint8)

            left_c = symmetrical_crop_and_upscale_subimage(left_255, crop_border_px)
            depth_c = symmetrical_crop_and_upscale_subimage(depth_255, crop_border_px)
            mask_c = symmetrical_crop_and_upscale_subimage(mask_255, crop_border_px)
            right_c = symmetrical_crop_and_upscale_subimage(right_255, crop_border_px)

            left_f = left_c.astype(np.float32) / 255.0
            depth_f = depth_c.astype(np.float32) / 255.0
            mask_f = mask_c.astype(np.float32) / 255.0
            right_f = right_c.astype(np.float32) / 255.0

            top = np.concatenate([left_f, depth_f], axis=1)
            bot = np.concatenate([mask_f, right_f], axis=1)
            grid_2x2 = np.concatenate([top, bot], axis=0)

            grid_255 = np.clip(grid_2x2 * 255, 0, 255).astype(np.uint8)
            grid_bgr = cv2.cvtColor(grid_255, cv2.COLOR_RGB2BGR)
            out_grid.write(grid_bgr)

            anag = create_anaglyph(left_f, right_f)
            anag_255 = np.clip(anag * 255, 0, 255).astype(np.uint8)
            anag_bgr = cv2.cvtColor(anag_255, cv2.COLOR_RGB2BGR)
            out_anag.write(anag_bgr)

        # Progress print for each batch processed
        print(f"[INFO] => (2x2) Splatted frames {start}..{end - 1} out of {total_frames}")

        del left_t, disp_t, right_t, occ_t
        torch.cuda.empty_cache()
        gc.collect()
        start = end

    out_grid.release()
    out_anag.release()
    print(f"[INFO] => (2x2) Output saved => {output_video_path}")


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
    warp_exponent_base=1.414,
    warp_interpolation_mode=1,
    max_small_hole_area=100,
    inpaintRadius=3,
    orig_brightness_value=1.0,
    orig_gamma_value=1.0,
    crop_border_px=0
):
    import cv2
    import numpy as np
    import torch
    import gc
    video_path_fixed = ensure_clean_video(input_video_path)
    vr = VideoReader(video_path_fixed, ctx=cpu(0))
    fps = vr.get_avg_fps()
    frames_all = vr[:].asnumpy().astype(np.float32) / 255.0
    if process_length != -1 and process_length < len(frames_all):
        frames_all = frames_all[:process_length]
        video_depth = video_depth[:process_length]
    total_frames = len(frames_all)
    if total_frames == 0:
        return

    H, W, _ = frames_all[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dir_ = os.path.dirname(output_base)
    base_ = os.path.basename(output_base)
    os.makedirs(dir_, exist_ok=True)
    out_spl_path = os.path.join(dir_, base_ + "_splatted.mp4")
    out_left_path = os.path.join(dir_, base_ + "_left.mp4")
    out_mask_path = os.path.join(dir_, base_ + "_mask.mp4")

    out_spl = cv2.VideoWriter(out_spl_path, fourcc, fps, (W, H))
    out_left = cv2.VideoWriter(out_left_path, fourcc, fps, (W, H))
    out_mask = cv2.VideoWriter(out_mask_path, fourcc, fps, (W, H))

    if keyframes_params is None:
        keyframes_params = {}
    real_kf = {}
    for k, v in keyframes_params.items():
        try:
            real_kf[int(k)] = v
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
        "blur_sigma_value": blur_sigma,
        "left_brightness_value": orig_brightness_value,
        "left_gamma_value": orig_gamma_value
    }

    stereo = ForwardWarpStereo(warp_interpolation_mode=warp_interpolation_mode, occlu_map=True).cuda()
    stereo.warp_exponent_base = warp_exponent_base

    start = 0
    while start < total_frames:
        end = min(start + batch_size, total_frames)
        sub_batch = end - start
        bf = frames_all[start:end]
        bd = video_depth[start:end]

        for i in range(sub_batch):
            frame_idx = start + i
            if enable_interpolation:
                local_params = interpolate_frame_params(real_kf, frame_idx, default_dict)
            else:
                if frame_idx in real_kf:
                    local_params = {**default_dict, **real_kf[frame_idx]}
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
            lbri_val = local_params["left_brightness_value"]
            lgam_val = local_params["left_gamma_value"]

            bf[i] = np.clip(bf[i] * lbri_val, 0, 1)
            bf[i] = bf[i] ** (1.0 / max(lgam_val, 1e-9))
            bf[i] = np.clip(bf[i], 0, 1)

            bd[i] = np.clip(bd[i] * bri_val, 0, 1)
            bd[i] = bd[i] ** (1.0 / max(gam_val, 1e-9))
            bd[i] = np.clip(bd[i], 0, 1)
            bd[i] = apply_depth_preprocess(bd[i], dh_val, dv_val, bk_val, bs_val)
            bd[i] = (bd[i] * 2.0 - 1.0) * disp_val + conv_val

        left_t = torch.from_numpy(bf).permute(0, 3, 1, 2).float().cuda()
        disp_t = torch.from_numpy(bd).unsqueeze(1).float().cuda()
        with torch.no_grad():
            right_t, occ_t = stereo(left_t, disp_t, convergence=0.0)
        rn = right_t.cpu().permute(0, 2, 3, 1).numpy()
        on = occ_t.cpu().permute(0, 2, 3, 1).numpy()

        for j in range(sub_batch):
            left_bgr = (bf[j] * 255).astype(np.uint8)
            left_bgr = cv2.cvtColor(left_bgr, cv2.COLOR_RGB2BGR)

            right_u8 = np.clip(rn[j] * 255, 0, 255).astype(np.float32)
            occ_01_ch = on[j][..., 0][..., None].astype(np.float32)

            updated_rf, updated_occ = inpaint_small_holes_inplace(
                right_u8, occ_01_ch, max_hole_area=max_small_hole_area, inpaintRadius=inpaintRadius
            )
            final_spl_u8 = np.clip(updated_rf, 0, 255).astype(np.uint8)
            final_spl_bgr = cv2.cvtColor(final_spl_u8, cv2.COLOR_RGB2BGR)

            hole_bool = (updated_occ[..., 0] > 0.5)
            mask_3ch = np.repeat(hole_bool[..., None], 3, axis=-1).astype(np.uint8) * 255

            left_bgr_c = symmetrical_crop_and_upscale_subimage(left_bgr, crop_border_px)
            right_bgr_c = symmetrical_crop_and_upscale_subimage(final_spl_bgr, crop_border_px)
            mask_3ch_c = symmetrical_crop_and_upscale_subimage(mask_3ch, crop_border_px)

            out_left.write(left_bgr_c)
            out_spl.write(right_bgr_c)
            out_mask.write(mask_3ch_c)

        # Progress print for each batch processed
        print(f"[INFO] => (triple) Splatted frames {start}..{end - 1} out of {total_frames}")

        del left_t, disp_t, right_t, occ_t
        torch.cuda.empty_cache()
        gc.collect()
        start = end

    out_spl.release()
    out_left.release()
    out_mask.release()
    print(f"[INFO] => (triple) Output saved => right={out_spl_path}, left={out_left_path}, mask={out_mask_path}")


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
            bits_16 = frame_16.view(np.uint16)
            arr_16 = array.array('H', bits_16.flatten().tolist())
            packed_data = arr_16.tobytes()
        else:
            packed_data = frame.tobytes()
        out_exr = os.path.join(out_dir, f"frame_{i:04d}.exr")
        of = OpenEXR.OutputFile(out_exr, header)
        of.writePixels({"Z": packed_data})
        of.close()


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
    fill_technique="polylines",
    warp_interpolation_mode=1,
    max_small_hole_area=100,
    inpaintRadius=3,
    crop_border_px=0
):
    kf_data = {}
    if keyframes_json and os.path.isfile(keyframes_json):
        try:
            with open(keyframes_json, "r", encoding="utf-8") as f:
                kf_data = json.load(f)
        except Exception:
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
            return

    depthcrafter_demo = None
    if (DepthCrafterPipeline and DiffusersUNetSpatioTemporalConditionModelDepthCrafter
            and unet_path and pre_trained_path):
        try:
            depthcrafter_demo = DepthCrafterDemo(unet_path, pre_trained_path)
        except:
            depthcrafter_demo = None

    os.makedirs(output_splatted, exist_ok=True)
    os.makedirs(input_depth_maps, exist_ok=True)

    for vp in vids:
        bn = os.path.splitext(os.path.basename(vp))[0]
        need_adj = (abs(orig_brightness_value - 1.0) > 1e-5 or abs(orig_gamma_value - 1.0) > 1e-5)
        reencoded_path = None
        dummy_out_depthvis = None
        if need_adj:
            reencoded_path = "temp_reencode_original.mp4"
            if os.path.exists(reencoded_path):
                os.remove(reencoded_path)
            reencode_brightness_gamma(vp, reencoded_path,
                                      brightness=orig_brightness_value,
                                      gamma=orig_gamma_value)
            video_for_depth = ensure_clean_video(reencoded_path)
        else:
            video_for_depth = ensure_clean_video(vp)

        dp = os.path.join(input_depth_maps, bn + "_depth.mp4")

        if depth_only:
            if depthcrafter_demo is not None:
                dummy_out_depthvis = os.path.join(output_splatted, bn + "_dummy_depthvis.mp4")
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
                    out_exr_dir = dp.replace(".mp4", "_exrseq")
                    save_exr_sequence_depth(vd, out_exr_dir, half_float=True)
                elif depth_output_mode == "exr32":
                    out_exr_dir = dp.replace(".mp4", "_exrseq")
                    save_exr_sequence_depth(vd, out_exr_dir, half_float=False)
                else:
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
            continue
        else:
            if os.path.exists(dp):
                vd, dv = load_pre_rendered_depth(dp, process_length=process_length)
            else:
                if depthcrafter_demo is None:
                    if reencoded_path and os.path.exists(reencoded_path):
                        try:
                            os.remove(reencoded_path)
                        except:
                            pass
                    continue
                dummy_out_depthvis = os.path.join(output_splatted, bn + "_dummy_depthvis.mp4")
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
                    out_exr_dir = dp.replace(".mp4", "_exrseq")
                    save_exr_sequence_depth(vd, out_exr_dir, half_float=True)
                elif depth_output_mode == "exr32":
                    out_exr_dir = dp.replace(".mp4", "_exrseq")
                    save_exr_sequence_depth(vd, out_exr_dir, half_float=False)
                else:
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
            vd = np.clip(vd * depth_brightness_value, 0, 1)
            vd = vd ** (1.0 / max(depth_gamma_value, 1e-9))
            vd = np.clip(vd, 0, 1)
            dv = chunked_vis_sequence_depth(vd)

        local_kf = {}
        if bn in kf_data and isinstance(kf_data[bn], dict):
            local_kf = kf_data[bn]

        # -- Removed polylines logic and references to it --

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
                warp_exponent_base=warp_exponent_base,
                warp_interpolation_mode=warp_interpolation_mode,
                max_small_hole_area=max_small_hole_area,
                inpaintRadius=inpaintRadius,
                orig_brightness_value=orig_brightness_value,
                orig_gamma_value=orig_gamma_value,
                crop_border_px=crop_border_px
            )
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
                warp_exponent_base=warp_exponent_base,
                warp_interpolation_mode=warp_interpolation_mode,
                max_small_hole_area=max_small_hole_area,
                inpaintRadius=inpaintRadius,
                orig_brightness_value=orig_brightness_value,
                orig_gamma_value=orig_gamma_value,
                crop_border_px=crop_border_px
            )


if __name__ == "__main__":
    Fire(main)
