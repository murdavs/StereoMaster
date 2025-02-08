# StereoMaster

**StereoMaster** is a GUI application for converting 2D videos into impressive 3D content using AI. It combines:
- [DepthCrafter](https://huggingface.co/tencent/DepthCrafter)  
- [Video Depth Anything](https://huggingface.co/depth-anything/Video-Depth-Anything)  
- [StereoCrafter](https://huggingface.co/TencentARC/StereoCrafter)

## Features

- **Depth map generation** using DepthCrafter or Video Depth Anything.
- **Splatting** (stereo shift) and **filling** (inpainting) with StereoCrafter.
- **User-friendly GUI** with:
  - Dynamic adjustments for disparity, convergence, and depth parameters.
  - Keyframes to modify 3D intensity over time.
  - Multiple preview modes (Original, Depth, Anaglyph, etc.).
  - Scene detection and merging of clips for longer videos.

## Requirements

- Windows 10/11 (64-bit).
- **NVIDIA** GPU with at least 12 GB VRAM (16 GB recommended).
- Python 3.10 and Git (the `.bat` script can install them).
- Internet connection for initial model downloads.

## Installation

1. **Download or clone** this repository.
2. In the root folder, run:
   - `1.Install StereoMaster.bat`
   - Follow the prompts; a virtual environment will be created and dependencies installed.
3. To launch the application:
   - `2.Launch StereoMaster.bat`

## Usage

1. **Open** StereoMaster.
2. **Create** a new project or select an existing one.
3. **Place** your video in the `input_videos` folder.
4. In the **GUI**:
   - Generate depth maps (using Crafter or VDA).
   - Perform splatting and inpainting to fill any holes.
   - Adjust disparity, convergence, etc.
   - Check Anaglyph preview modes.
5. **Export** your result in side-by-side format (HSBS or FSBS).

## Limitations

- Geared towards **short videos**; for longer ones, please split them first.
- **Not compatible with 4K videos** (may cause crashes or out-of-memory errors).
- Consumes significant VRAM; watch out for overly high resolutions.
- If your PC freezes, try reducing the video length or resolution.

## Ko-fi

If you like this project, consider [buying me a coffee on Ko-fi](https://ko-fi.com/3dultraenhancer). Your support helps keep development going and enables more features in the future.

## License

Provided “as is.” Use at your own risk; no warranty is given for potential data loss or system problems.

---

## Example Screenshot

Below is a sample image of **StereoMaster** in action:

![StereoMaster Screenshot](assets/screenshot.png)
