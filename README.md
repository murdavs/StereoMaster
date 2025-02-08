# StereoMaster

**StereoMaster** is a GUI application to transform 2D videos into impressive 3D content using AI. It leverages:
- [DepthCrafter](https://huggingface.co/tencent/DepthCrafter)  
- [Video Depth Anything](https://huggingface.co/depth-anything/Video-Depth-Anything)  
- [StereoCrafter](https://huggingface.co/TencentARC/StereoCrafter)

## Features

- **Depth map generation** with either DepthCrafter or Video Depth Anything.
- **Splatting** (stereo shift) and **hole-filling** (inpainting) using StereoCrafter.
- **User-friendly GUI** with:
  - Dynamic adjustments for disparity, convergence, and depth parameters.
  - Keyframes to modify 3D intensity over time.
  - Preview modes (Original, Depth, Anaglyph, etc.).
  - Scene detect & merge options for long videos.

## Requirements

- Windows 10/11 (64-bit).
- **NVIDIA** GPU >= 12 GB VRAM (16 GB recommended).
- Python 3.10 and Git (the `.bat` script can install them for you).
- Internet for initial model downloads.

## Installation

1. Download or clone this repository.
2. In the root folder, run:
   - `1.Install StereoMaster.bat`
   - Follow the on-screen prompts; a virtual environment will be created and dependencies installed.
3. To launch the app:
   - `2.Launch StereoMaster.bat`

## Usage

1. **Open** StereoMaster.
2. **Create** a project or select an existing one.
3. **Place** your video in `input_videos`.
4. In the **GUI**:
   - Generate Depth (Crafter/VDA).
   - Splat + Inpaint.
   - Adjust disparity, convergence, etc.
   - Check Polylines or Anaglyph preview modes.
5. **Export** to side-by-side format (HSBS or FSBS).

## Limitations

- Designed for **short videos**. For longer ones, split them first.
- Uses a lot of VRAM; watch out for high resolutions.
- If your PC freezes, reduce the video length or resolution.

## Ko-fi

If you like this project, feel free to [buy me a coffee on Ko-fi](https://ko-fi.com/3dultraenhancer)!

## License

Provided “as is.” Use at your own risk; no warranty is given regarding possible data loss or system issues.
