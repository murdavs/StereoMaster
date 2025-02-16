# StereoMaster

**StereoMaster** is a GUI application for converting 2D videos into impressive 3D content using AI. It combines:
- [DepthCrafter](https://huggingface.co/tencent/DepthCrafter)  
- [Video Depth Anything](https://huggingface.co/depth-anything/Video-Depth-Anything)  
- [StereoCrafter](https://huggingface.co/TencentARC/StereoCrafter)


## 📣 News
- `2025/02/09` Initial commit.
- `2025/02/16` Update: Added downscale inpainting, anaglyph file merging, enhanced partial frames, improved StereoCrafter (mask dilation, blur, configurable chunking), refined color matcher (fixed preview vs. SBS mismatch), new outputs (4KHSBS, Right-Only, EXR export), plus various bug fixes.



## Features

- **Depth map generation** using DepthCrafter or Video Depth Anything.
- **Splatting** (stereo shift) and **filling** (inpainting) with StereoCrafter.
- **User-friendly GUI** with:
  - Dynamic adjustments for disparity, convergence, and depth parameters.
  - Keyframes to modify 3D intensity over time.
  - Multiple preview modes (Original, Depth, Anaglyph, etc.).
  - Scene detection and merging of clips for longer videos.


## System Requirements

- **OS:** Windows 10/11 (64-bit)  
- **GPU:** NVIDIA (RTX 3000/4000 series or newer)  
  - **Required VRAM:** 12 GB (16 GB recommended)  
- **CUDA:** 12.6 (older versions will not work)  
- **Python:** 3.12  
- **Visual Studio Build Tools:** 2019 or 2022 (MSVC 14.36 - 14.40)  
- **Git:** Required for dependency management  

---

## Installation Guide (Python 3.12 & CUDA 12.6)

> **Note:** After you **complete all these steps**, you can **automatically launch StereoMaster** by double-clicking the file **`Launch StereoMaster.bat`** in the repository’s root folder.

---

### 1. Install Python 3.12.8

1. Download here:  
   [Python 3.12.8 (64-bit) for Windows](https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe)

2. Run the installer and **check** the box **“Add Python to PATH.”**

3. Click **Customize Installation** and enable:
    - Install pip  
    - Install venv  

4. Click **Next** → **Install** → **Finish**.

5. Open **CMD** (Win + R → type `cmd` → press Enter) and run:

    ```
    python --version
    ```

   You should see:
   
    ```
    Python 3.12.8
    ```

---

### 2. Remove Microsoft Store Python (If Present)

Windows sometimes has a placeholder Python alias in `WindowsApps` that can interfere with the real Python installation.  
Delete it with:

    del C:\Users\%USERNAME%\AppData\Local\Microsoft\WindowsApps\python.exe

Then verify again:

    where python

Ensure it points to the real Python installation

---

### 3. Install Visual Studio Build Tools

1. Download from:  
   [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

2. Run the installer and select:
    - C++ CMake tools for Windows  
    - Windows 10/11 SDK    
    - MSVC v143 - VS 2022 C++ x64/x86 (14.40-17.10)  
	- C++ compilation tools for MSVC v143-VS 2022 for x64/x86 (latest)
    - Spectre-mitigated libraries for MSVC v143 – VS 2022 C++ x64/x86 (v14.40–17.10) 

   *(If you don’t see these options, check **Individual Components**.)*


---

### 4. Install CUDA 12.6 & cuDNN

1. Download CUDA 12.6 (local `.exe`) from:  
   [NVIDIA CUDA 12.6.3 Downloads](https://developer.nvidia.com/cuda-12-6-3-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

2. On the website, select:
    - Operating System: Windows  
    - Architecture: x86_64  
    - Version: Windows 10/11  
    - Installer Type: exe (local)

3. Run the installer, **check**:
    - CUDA Toolkit  
    - CUDA Runtime  
    - NVCC Compiler  

4. Restart your PC.

**Verify CUDA** by running:

    nvcc --version

Expected output:

    release 12.6, V12.6.3

---

### 5. Configure Environment Variables

1. Open **CMD as Administrator** (Win + R → type `cmd` → press **Ctrl + Shift + Enter**).

2. Run these commands:

    ```
    setx DISTUTILS_USE_SDK "1" /M
    setx USE_CUDA "1" /M
    setx CMAKE_GENERATOR "Visual Studio 17 2022" /M
    setx CMAKE_GENERATOR_PLATFORM "x64" /M
    setx PATH "%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin" /M
    setx CUDA_HOME "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" /M
    ```

3. Activate the correct Visual Studio compiler:

   **For VS 2022 Build Tools**:

       "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

   **For VS 2019 Build Tools**:

       "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
	   
	Verify running:

    ```	
    cl	
    ```
    
	You should see something like:
   
    ```
    Microsoft (R) C/C++ Optimizing Compiler Version 19.36.32530 for x64
	
	```

---

### 6. Install Git

1. Download from:  
   [Git for Windows](https://git-scm.com/download/win)

2. Run the installer (accepting the default options is usually fine).


---	


### 7. Clone the StereoMaster Repository

    cd C:\
    git clone https://github.com/murdavs/StereoMaster.git
    cd StereoMaster

---

### 8. Create a Virtual Environment & Install Dependencies

1. Create a new virtual environment:

       python -m venv stereomaster_env

2. Activate it:

       call stereomaster_env\Scripts\activate

3. Upgrade pip:

       pip install --upgrade pip setuptools wheel


4. Install other dependencies:

       pip install --use-pep517 --no-cache-dir -r requirements.txt

**Verify PyTorch & CUDA**:

    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

Expected:

    2.6.0
    True

---

### 9. Install Triton (Updated for Python 3.12)

1. Navigate to the `assets` folder:

       cd assets

2. Install the Triton wheel:

       pip install triton-3.1.0-cp312-cp312-win_amd64.whl

---

### 10. Compile Forward Warp

1. Go to the Forward-Warp folder:

       cd ..\dependency\Forward-Warp\

2. Clean and install:

       python setup.py clean
       pip install .

3. Enter the `cuda` folder, then clean and install again:

       cd Forward-Warp\cuda
       python setup.py clean
       pip install .

---


### 11. Login in Hugging Face CLI & Download Model Weights

1. Login in Hugging Face CLI:

       
       huggingface-cli login

   > Get your token here: [Hugging Face Token Settings](https://huggingface.co/settings/tokens)

2. Clone the required models into a `weights` folder:

       cd C:\StereoMaster
       mkdir weights
       cd weights
       git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
       git clone https://huggingface.co/tencent/DepthCrafter
       git clone https://huggingface.co/TencentARC/StereoCrafter

3. Download depth models:

       cd C:\StereoMaster
       mkdir checkpoints
	   cd checkpoints
       curl -k -L -o video_depth_anything_vits.pth https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
       curl -k -L -o video_depth_anything_vitl.pth https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth

---

## Launch StereoMaster

After **all steps** are completed:

1. Go to the root folder of StereoMaster:

       cd C:\StereoMaster

2. Activate the virtual environment:

       call stereomaster_env\Scripts\activate

3. Run the script:

       python StereoMaster.py

4. **Alternatively**, **double-click**:
   **`Launch StereoMaster.bat`** (in the root directory) to launch automatically.

---

## Forward Warp CUDA Extension Installation Troubleshooting (Windows)

1. **Check Microsoft Visual C++ Build Tools Installation**  
   - Make sure that all the packages listed in step 3 are installed with the specified versions.
   - This ensures `cl.exe` and associated tools are available.

2. **Open the Correct Developer Command Prompt**  
   - Run the following command before running the Forward-Warp CUDA installation script to configure the environment for 64-bit builds:
     ```
     "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
     ```
   - This ensures that the compiler (`cl.exe`) is on your PATH and recognized.
   
   
---

## Ko-fi Support

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/3dultraenhancer)

If you find StereoMaster helpful or would like to support further development, consider [buying me a coffee](https://ko-fi.com/3dultraenhancer)). Thank you!

---

## License

Provided “as is.” Use at your own risk; no warranty is given for potential data loss or system problems.

---

## Screenshot

Below is a sample image of **StereoMaster** in action:

![StereoMaster Screenshot](assets/screenshot.png)








