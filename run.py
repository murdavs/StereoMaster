import argparse
import numpy as np
import os
import torch
import time
import cv2
import sys
import warnings
import OpenEXR
import Imath
import array

warnings.filterwarnings(
    "error",
    message=".*invalid value encountered in.*",
    category=RuntimeWarning
)

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

def save_exr_sequence_depth(depth_array, out_dir, half_float=False):
    """
    Saves each frame in depth_array to a separate .exr file.

    Arguments:
    - depth_array: a float32 NumPy array of shape [T, H, W], normalized to [0,1].
    - out_dir: directory in which the .exr files will be saved.
    - half_float: if True, data is saved as half-float (16-bit), otherwise 32-bit float.
    """
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

        exr_path = os.path.join(out_dir, f"frame_{i:04d}.exr")
         
        of = OpenEXR.OutputFile(exr_path, header)
        try:
            of.writePixels({"Z": packed_data})
        finally:
            of.close()

        print(f"[EXR] => wrote {exr_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything - Inference script.')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--target_fps', type=int, default=-1)
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--use_torchscript', action='store_true')
    parser.add_argument('--use_cudnn_benchmark', action='store_true')

    # NEW: single argument for depth output mode
    parser.add_argument(
        '--depth_output_mode',
        type=str,
        default='mp4',
        choices=['mp4', 'exr16', 'exr32'],
        help="Depth output mode: 'mp4' (8-bit), 'exr16' (16-bit half float) or 'exr32' (32-bit float)."
    )

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.use_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # Model configuration
    model_configs = {
        'vits': {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384]
        },
        'vitl': {
            'encoder': 'vitl',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024]
        },
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(
        torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'),
        strict=True
    )
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    if args.use_torchscript:
        dummy_input = torch.randn(1, 32, 3, args.input_size, args.input_size).to(DEVICE)
        with torch.no_grad():
            video_depth_anything = torch.jit.trace(video_depth_anything, dummy_input)
        print("Model converted to TorchScript.")

    try:
        frames, target_fps, original_res = read_video_frames(
            video_path=args.input_video,
            process_length=args.max_len,
            target_fps=args.target_fps,
            max_res=args.max_res
        )
        original_height, original_width = original_res

        start_time = time.time()
        with torch.no_grad():
            if args.use_fp16:
                with torch.amp.autocast(device_type='cuda'):
                    depths, fps = video_depth_anything.infer_video_depth(
                        frames,
                        target_fps,
                        input_size=args.input_size,
                        device=DEVICE
                    )
            else:
                depths, fps = video_depth_anything.infer_video_depth(
                    frames,
                    target_fps,
                    input_size=args.input_size,
                    device=DEVICE
                )
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.2f} seconds")

        # Resize to original resolution if needed
        inf_h, inf_w = depths.shape[1], depths.shape[2]
        if (inf_h, inf_w) != (original_height, original_width):
            print(f"Resizing depth maps from ({inf_h}, {inf_w}) to ({original_height}, {original_width})...")
            upscaled_depths = []
            for i in range(depths.shape[0]):
                d = depths[i]
                d = np.ascontiguousarray(d)
                if d.dtype == np.float16:
                    d = d.astype(np.float32)
                if d.ndim == 3 and d.shape[2] == 1:
                    d = d[:, :, 0]
                if d.size == 0:
                    raise ValueError(f"Frame {i} is empty and cannot be resized.")
                d_up = cv2.resize(d, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
                upscaled_depths.append(d_up)
            depths = np.stack(upscaled_depths, axis=0)

        # Clean and normalize in [0,1]
        cleaned_depths = []
        for i in range(depths.shape[0]):
            d = depths[i].astype(np.float32)
            d_min, d_max = d.min(), d.max()
            denom = max(d_max - d_min, 1e-5)
            d_norm = (d - d_min) / denom
            d_norm = np.nan_to_num(d_norm, nan=0.0, posinf=1.0, neginf=0.0)
            d_norm = np.clip(d_norm, 0.0, 1.0)
            cleaned_depths.append(d_norm)
        depths = np.stack(cleaned_depths, axis=0)

        # Create output directory
        video_name = os.path.basename(args.input_video)
        base_name, _ = os.path.splitext(video_name)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # Save depth output according to the chosen mode
        if args.depth_output_mode == "mp4":
            # 8-bit MP4 (visualization)
            depth_vis_path = os.path.join(args.output_dir, base_name + '_depth.mp4')
            save_video(depths, depth_vis_path, fps=fps, is_depths=True)
            print(f"[OK] => Saved 8-bit depth video: {depth_vis_path}")

        elif args.depth_output_mode == "exr16":
            # 16-bit EXR (half float)
            exr_folder = os.path.join(args.output_dir, base_name + '_exr16')
            save_exr_sequence_depth(depths, exr_folder, half_float=True)
            print(f"[OK] => Saved 16-bit EXR sequence in: {exr_folder}")

        elif args.depth_output_mode == "exr32":
            # 32-bit EXR (float)
            exr_folder = os.path.join(args.output_dir, base_name + '_exr32')
            save_exr_sequence_depth(depths, exr_folder, half_float=False)
            print(f"[OK] => Saved 32-bit EXR sequence in: {exr_folder}")

        print("[OK] => Process finished successfully.")

    except RuntimeWarning as e:
        print("[ERROR] => A runtime warning was raised! Aborting execution...")
        print(f"Details: {e}")
        sys.exit(1)
