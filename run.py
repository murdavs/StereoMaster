# run.py
# Copyright (2025) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import numpy as np
import os
import torch
import time
import cv2  # To rescale the depth map

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280, help="Maximum resolution for inference (if the video is larger, it will be resized)")
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='Maximum number of frames to process; -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='Target FPS for the input video; -1 keeps the original')
    
    # Parameters for inference optimization
    parser.add_argument('--use_fp16', action='store_true', help='Use mixed precision (FP16) for inference')
    parser.add_argument('--use_torchscript', action='store_true', help='Convert the model to TorchScript to speed up inference')
    parser.add_argument('--use_cudnn_benchmark', action='store_true', help='Enable cuDNN benchmark (useful if input sizes are constant)')

    args = parser.parse_args()

    # Select device: GPU is preferred if available
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Enable cuDNN benchmark if required
    if args.use_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
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

    # Read video frames
    frames, target_fps, original_res = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    original_height, original_width = original_res

    # Inference
    start_time = time.time()
    with torch.no_grad():
        if args.use_fp16:
            # --- Keep the original line commented ---
            # with torch.cuda.amp.autocast():
            
            # --- Add the new form of autocast ---
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

    # Rescale the depth maps if they differ from the original resolution
    inferred_height, inferred_width = depths.shape[1], depths.shape[2]
    if (inferred_height, inferred_width) != (original_height, original_width):
        print(f"Resizing depth map from ({inferred_height}, {inferred_width}) to ({original_height}, {original_width})")
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

    # Create output directory if it does not exist
    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save the depth video
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0] + '_depth.mp4')
    save_video(depths, depth_vis_path, fps=fps, is_depths=True)
