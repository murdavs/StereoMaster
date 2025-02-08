import sys
import os
import cv2
import glob
import json
import numpy as np
import subprocess
import gc
from PyQt5.QtCore import QLocale
from PyQt5.QtCore import QTimer        
from numba import njit, prange
import numpy as np

from PyQt5.QtCore import QUrl, QSize
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QMessageBox
from decord._ffi.base import DECORDError 
from decord import VideoReader, cpu
from PyQt5.QtGui import (
    QIcon, QPixmap, QImage, QPainter, QPen, QColor, QMouseEvent,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QSlider, QHBoxLayout, QVBoxLayout, QFileDialog, QCheckBox,
    QListWidget, QListWidgetItem, QMessageBox, QSpacerItem, QSizePolicy,
    QPlainTextEdit, QLineEdit, QGroupBox, QFormLayout, QComboBox, QTabWidget,
    QProgressBar, QToolTip, QSpinBox, QDoubleSpinBox, QInputDialog, QAbstractItemView, QButtonGroup
)

import ctypes

try:
   
    myappid = u"StereoMaster.1.0"  
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception as e:
    print("Could not set AppUserModelID:", e)
    
    
@njit(parallel=True)
def apply_stereo_divergence_polylines(
    original_image, 
    normalized_depth,
    divergence_px: float,
    separation_px: float,
    stereo_offset_exponent: float,
    fill_technique: str
):
    """
    This code treats each row of 'original_image' as a polyline. It applies a
    horizontal offset to each pixel based on the depth (normalized_depth) and
    some exponent. The result is a stereo-shifted image with no holes.

    original_image => [H, W, 3], uint8  (LEFT)
    normalized_depth => [H, W], in [0..1]
    divergence_px => maximum horizontal shift (pixels) for depth=1
    separation_px => additional constant separation (pixels)
    stereo_offset_exponent => exponent for depth scaling
    fill_technique => 'polylines' or 'polylines_sharp' => 
                      si 'polylines_sharp', desplazamos medio pixel extra.
    Returns derived_image => [H, W, 3], uint8 (RIGHT)
    """
    EPSILON = 1e-7
    PIXEL_HALF_WIDTH = 0.45 if fill_technique == 'polylines_sharp' else 0.0

    h, w, c = original_image.shape
    derived_image = np.zeros_like(original_image)

    for row in prange(h):
        pt = np.zeros((5 + 2*w, 3), dtype=np.float64)
        pt_end = 0

   
        pt[pt_end] = [-1.0*w, 0.0, 0.0]
        pt_end += 1

        for col in range(w):
            d_val = normalized_depth[row, col]
            coord_d = -(d_val**stereo_offset_exponent)* divergence_px
            coord_x = col + 0.5 + coord_d - separation_px

            if PIXEL_HALF_WIDTH < EPSILON:
                pt[pt_end] = [coord_x, abs(coord_d), col]
                pt_end += 1
            else:
                pt[pt_end]   = [coord_x - PIXEL_HALF_WIDTH, abs(coord_d), col]
                pt[pt_end+1] = [coord_x + PIXEL_HALF_WIDTH, abs(coord_d), col]
                pt_end += 2

     
        pt[pt_end] = [2.0*w, 0.0, w-1]
        pt_end += 1

  
        sg_end = pt_end - 1
        sg = np.zeros((sg_end, 6), dtype=np.float64)
        for i in range(sg_end):
            sg[i, 0] = pt[i, 0]
            sg[i, 1] = pt[i, 1]
            sg[i, 2] = pt[i, 2]
            sg[i, 3] = pt[i+1, 0]
            sg[i, 4] = pt[i+1, 1]
            sg[i, 5] = pt[i+1, 2]

      
        for i in range(1, sg_end):
            j = i - 1
            key_pt = pt[i].copy()
            key_sg = sg[i].copy()
            while j >= 0 and pt[j,0] > key_pt[0]:
                pt[j+1] = pt[j]
                sg[j+1] = sg[j]
                j -= 1
            pt[j+1] = key_pt
            sg[j+1] = key_sg

        
        csg = np.zeros((5*(int(abs(divergence_px))+5), 6), dtype=np.float64)
        csg_end = 0
        sg_pointer = 0
        pt_i = 0

        for col in range(w):
            color_mix = np.zeros(c, dtype=np.float64)

            while pt[pt_i,0] < col:
                pt_i += 1
                if pt_i>= pt_end-1:
                    break
            pt_i -= 1
            if pt_i<0:
                pt_i=0

            while pt[pt_i,0] < col+1.0:
                coord_from = max(col, pt[pt_i,0]) + EPSILON
                coord_to   = min(col+1.0, pt[pt_i+1,0]) - EPSILON
                if coord_to < coord_from:
                    pt_i += 1
                    if pt_i>= pt_end-1:
                        break
                    continue

                significance = coord_to - coord_from
                coord_center = coord_from + 0.5* significance

         
                while sg_pointer< sg_end and sg[sg_pointer,0] < coord_center:
                    csg[csg_end] = sg[sg_pointer]
                    csg_end += 1
                    sg_pointer += 1

               
                csg_i = 0
                while csg_i< csg_end:
                    if csg[csg_i,3] < coord_center:
                        csg[csg_i] = csg[csg_end-1]
                        csg_end-=1
                    else:
                        csg_i+=1

                
                best_idx= 0
                best_cl = -999999.0
                for csg_i in range(csg_end):
                    denom= (csg[csg_i,3] - csg[csg_i,0])
                    if abs(denom)< EPSILON:
                        continue
                    ip_k= (coord_center - csg[csg_i,0])/ denom
                    if ip_k<0.0 or ip_k>1.0:
                        continue
                    closeness= (1.0-ip_k)* csg[csg_i,1] + ip_k* csg[csg_i,4]
                    if closeness> best_cl:
                        best_cl = closeness
                        best_idx= csg_i

          
                col_l = int(csg[best_idx,2])
                col_r = int(csg[best_idx,5])

                if col_l== col_r:
                    color_mix += original_image[row, col_l]* significance
                else:
                    denom2= (csg[best_idx,3] - csg[best_idx,0])
                    if abs(denom2)< EPSILON:
                        color_mix += original_image[row, col_l]* significance
                    else:
                        ip_k2= (coord_center - csg[best_idx,0])/ denom2
                        color_mix += (original_image[row, col_l]*(1.0- ip_k2) +
                                      original_image[row, col_r]* ip_k2)* significance

                pt_i+=1
                if pt_i>= pt_end-1:
                    break

            derived_image[row, col] = np.clip(color_mix, 0,255).astype(np.uint8)

    return derived_image

###########################################################
# (A) Subclass QSlider with ticks for keyframes + richer tooltip
###########################################################
class KeyframeSlider(QSlider):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.keyframes_set = set()     # Set of frames that have a keyframe
        self.keyframes_dict = {}       # Dictionary: frame -> parameters for tooltip
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        # Show ticks
        self.setTickPosition(QSlider.TicksBelow)
        self.setTickInterval(1)

    def set_keyframes(self, frames_list):
        """
        Set which frames are keyframes (for painting the small marks).
        """
        self.keyframes_set = set(frames_list)
        self.update()

    def set_keyframes_dict(self, kf_dict):
        """
        kf_dict: dict of frame_index -> {all param info}
        This is for showing richer tooltip info.
        """
        self.keyframes_dict = kf_dict

    def paintEvent(self, event):
        super().paintEvent(event)
        # Draw small red marks where keyframes exist
        painter = QPainter(self)
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)
        h = self.height()
        slider_min = self.minimum()
        slider_max = self.maximum()
        # A bit more margin from the bar (adjust as desired):
        # originally was y_mark = h // 2 + 10
        # Increase to get more space above or below. Adjust as needed:
        y_mark = (h // 5) + 5  # <-- Moved further down from the handle

        for kf in self.keyframes_set:
            if slider_min <= kf <= slider_max:
                ratio = (kf - slider_min) / float(slider_max - slider_min) if slider_max > slider_min else 0
                # We'll subtract some padding so the line doesn't go off the edge:
                x_coord = int(ratio * (self.width() - 20)) + 10
                painter.drawLine(x_coord, y_mark, x_coord, y_mark + 8)
        painter.end()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos_x = event.pos().x()
        slider_min = self.minimum()
        slider_max = self.maximum()
        if slider_max == slider_min:
            super().mouseMoveEvent(event)
            return

        ratio = (pos_x - 10) / float(self.width() - 20)
        ratio = max(0, min(1, ratio))
        frame_approx = int(slider_min + ratio * (slider_max - slider_min))

        # Find nearest keyframe
        dist_min = 999999
        nearest_kf = None
        for kf in self.keyframes_set:
            dist_ = abs(kf - frame_approx)
            if dist_ < dist_min:
                dist_min = dist_
                nearest_kf = kf
        # If close enough to show tooltip
        if nearest_kf is not None and dist_min <= 1:
            # Build a richer tooltip: show frame + all parameters
            if nearest_kf in self.keyframes_dict:
                params_info = self.keyframes_dict[nearest_kf]
                # Example formatting:
                # Keyframe: frame 10
                # disp=20.0, conv=0.0, bri=1.0, gamma=1.0, ...
                tooltip_lines = [f"Keyframe: frame {nearest_kf}"]
                # We can show any keys from the dict:
                # (Customize the names you want to show in the tooltip)
                for p_name, p_val in params_info.items():
                    tooltip_lines.append(f"{p_name}={p_val}")
                    final_tooltip = "\n".join(tooltip_lines)
            else:
                final_tooltip = f"Keyframe: frame {nearest_kf}"
       

            QToolTip.showText(self.mapToGlobal(QPoint(pos_x, self.height() // 2)), final_tooltip)
        else:
            QToolTip.hideText()

        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        """
        If the user clicks near a keyframe, jump to it.
        Otherwise normal slider behavior.
        """
        pos_x = event.pos().x()
        slider_min = self.minimum()
        slider_max = self.maximum()
        if slider_max == slider_min:
            super().mousePressEvent(event)
            return

        ratio = (pos_x - 10) / float(self.width() - 20)
        ratio = max(0, min(1, ratio))
        frame_approx = int(slider_min + ratio * (slider_max - slider_min))

        # Check if near a keyframe
        dist_min = 999999
        nearest_kf = None
        for kf in self.keyframes_set:
            dist_ = abs(kf - frame_approx)
            if dist_ < dist_min:
                dist_min = dist_
                nearest_kf = kf
        if nearest_kf is not None and dist_min <= 1:
            self.setValue(nearest_kf)
        else:
            super().mousePressEvent(event)


###########################################################
# 0) Worker Thread => to avoid freezing the GUI
###########################################################
class SubprocessWorker(QThread):
    """
    Launch a subprocess in a separate thread, read stdout/stderr in real-time,
    emit lineReady(str) for each line, and finishedSignal(bool) when done.
    """
    lineReady = pyqtSignal(str)
    finishedSignal = pyqtSignal(bool)

    def __init__(self, cmd, parent=None):
        super().__init__(parent)
        self.cmd = cmd
        self.returncode = None
        self.process = None
        self._cancel_requested = False

    def run(self):
        self.lineReady.emit(f"[INFO] => Running: {' '.join(self.cmd)}")
        
        creation_flags = 0
        if os.name == "nt":  # Windows only           
            creation_flags = subprocess.CREATE_NO_WINDOW
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=creation_flags
            )
            for line in self.process.stdout:
                if self._cancel_requested:
                    self.lineReady.emit("[CANCEL] => Cancel requested. Terminating process.")
                    self.process.kill()
                    self.finishedSignal.emit(False)
                    return
                line_str = line.rstrip("\n\r")
                self.lineReady.emit(line_str)
            self.process.stdout.close()
            ret = self.process.wait()
            self.returncode = ret
            if ret != 0:
                self.lineReady.emit(f"[ERROR] => Process returned code {ret}")
                self.finishedSignal.emit(False)
            else:
                self.lineReady.emit("[OK] => Process finished successfully.")
                self.finishedSignal.emit(True)
        except Exception as e:
            self.lineReady.emit(f"[EXCEPTION] => {e}")
            self.finishedSignal.emit(False)

    def request_cancel(self):
        self._cancel_requested = True
        if self.process is not None:
            self.lineReady.emit("[CANCEL] => Killing process now...")
            try:
                self.process.kill()
            except Exception as e:
                self.lineReady.emit(f"[EXCEPTION] => {e}")


###########################################################
# 1) Forward Warp (anaglyph) => for preview
###########################################################
from Forward_Warp import forward_warp
import torch
import torch.nn as nn

class ForwardWarpStereo(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.fw = forward_warp()
        self.warp_exponent_base = 1.0

    def forward(self, left_im, depth_map, convergence=0.0):
    
        disp = depth_map + convergence  
        wmap = disp - disp.min()
        wmap = (self.warp_exponent_base) ** wmap
     
        flow_x = -disp.squeeze(1)  
        flow_y = torch.zeros_like(flow_x)
        flow   = torch.stack((flow_x, flow_y), dim=-1)
        B, C, H, W = left_im.shape      
        res_accum = torch.zeros_like(left_im)
        for c in range(C):
            chan_c = left_im[:, c : c+1] * wmap
            res_accum[:, c : c+1] = self.fw(chan_c.contiguous(), flow.contiguous())   
        mask_acc = self.fw(wmap.contiguous(), flow.contiguous())
        mask_acc.clamp_(min=self.eps)
        res = res_accum / mask_acc
        
        ones = torch.ones_like(depth_map).contiguous()
        occ_acc = self.fw(ones, flow.contiguous())
        occ_acc.clamp_(0.0, 1.0)
        occlusion = 1.0 - occ_acc

        return res, occlusion




###########################################################
# 2) Depth Preprocess => dilate + blur
###########################################################
def apply_depth_preprocess(depth_frame_01, dilate_h=0, dilate_v=0,
                           blur_ksize=0, blur_sigma=0.0):
    import cv2
    import numpy as np
    if dilate_h>0 or dilate_v>0:
        depth_u8 = (depth_frame_01*255).astype(np.uint8)
        kernel = np.ones((dilate_v,dilate_h), np.uint8)
        try:
            depth_u8 = cv2.dilate(depth_u8, kernel, iterations=1)
        except Exception as e:
            # Just in case
            pass
        depth_frame_01 = depth_u8.astype(np.float32)/255.0

    if blur_ksize>0:
        if blur_ksize%2==0:
            blur_ksize+=1
        depth_255 = (depth_frame_01*255).astype(np.float32)
        try:
            depth_255 = cv2.GaussianBlur(depth_255, (blur_ksize,blur_ksize), sigmaX=blur_sigma)
        except Exception as e:
            # Just in case
            pass
        depth_frame_01 = np.clip(depth_255/255.0,0,1)
    return depth_frame_01


###########################################################
# 3) Main class => "StereoMasterGUI"
###########################################################
class StereoMasterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon("./assets/icon.ico"))
        self.resize(1920, 1024)
        self.setWindowTitle("StereoMaster")
        self.presets = [None, None, None, None]  # 4 presets
        self.current_batch_worker = None

        # Per-video config
        self.settings_file = "video_params.json"   # Storing video-based parameters
        self.paths_file    = "config_paths.json"   # Storing folder paths
        
        self.warp_exponent_base = 1.414 
        self.depth_global_file = "depth_global_params.json"
        
        self.crafter_maxres_value = 512
        self.vda_maxres_value = 512

        self.disclaimer_accepted = False
        
         # Inpaint single
        self.inpaint_num_inference_steps_batch = 10
  
        # Inpaint batch
        self.inpaint_overlap_batch = 3
        self.inpaint_tile_num_batch = 2
        
        self.poly_exponent = 1.0         # Valor actual del exponent
        self.poly_fill_tech = "polylines"

        
        # Load folder paths
        self.load_paths()


        # Load per-video parameter dict
        self.video_params = {}
        self.load_params()

        right_tab_widget= QTabWidget()
        right_tab_widget.setMinimumWidth(320)      
        
        if not self.disclaimer_accepted:
            result = self.show_disclaimer_dialog()
            if result is None:
              
                import sys
                sys.exit(0)
            elif result == "accept_no_show":
                self.disclaimer_accepted = True
                self.save_disclaimer_accepted()         
        
        self.slider_timer = QTimer(self)
        self.slider_timer.setInterval(150)  # milisegundos de espera
        self.slider_timer.setSingleShot(True)
        self.slider_timer.timeout.connect(self.on_slider_timer_timeout)

        self.pending_slider_value = 0  # guardará el último valor

        # Variables
        self.vid_original     = None
        self.vid_depth        = None   # Depth1 (Crafter)
        self.vid_depth2       = None   # Depth2 (VDA)
        self.vid_depth_merged = None 
        self.vid_splatted     = None  
        self.vid_inpainted    = None        # DepthMerged
        self.num_frames       = 0
        self.current_frame_idx= 0

        # Rutas
        self.selected_video_path   = ""
        self.selected_depth_path   = ""  # Depth1
        self.selected_depth_path2  = ""  # Depth2
        self.selected_depth_merged = ""
        self.splatted_video_path   = ""
        self.inpainted_video_path  = ""

     
        self.enable_interpolation_local = True


        self.disp_value       = 20.0
        self.conv_value       = 0.0
        self.brightness_value = 1.0
        self.gamma_value      = 1.0
        self.dilate_h_value   = 3
        self.dilate_v_value   = 1
        self.blur_ksize_value = 3
        self.blur_sigma_value = 2.0

        # Param => Original
        self.orig_brightness = 1.0
        self.orig_gamma      = 1.0

        # Depth max_res => for Crafter/VDA
        self.depth_maxres_value = 512

        # Crafter overlap/window
        self.crafter_overlap = 25
        self.crafter_window  = 70

        # Crafter additional params
        self.crafter_denoising_steps = 8
        self.crafter_guidance_scale  = 1.2
        self.crafter_seed            = 42

        # VDA
        self.vda_use_fp16  = True
        self.vda_use_ts    = False
        self.vda_use_cudnn = False
        self.vda_input_size = 512
        self.vda_encoder    = "vitl"
        
        self.preview_bri = 1.0
        self.preview_gamma = 1.0

        # Depth fusion alpha
        self.fusion_alpha= 50

        # Depth used for Splat => "Depth1", "Depth2", "Merged"
        self.selected_splat_depth = "Depth1"

        # For the preview => "No Depth", "Depth1", "Depth2", "Merged"
        self.selected_preview_depth= "Depth1"



        # Threshold Mask for Inpainting (BATCH)
        self.inpaint_threshold_batch = 0.005

        # Build combos
        self.all_state_combos = []
        self.build_state_combos()

        # Stereo warper for preview
        self.stereo_warper = ForwardWarpStereo().cuda()
        
        # Load global depth/inpaint params
        self.load_depth_global_params()

        # UI
        self.initUI()
        self.scan_projects()
        self.set_styles()
        self.scan_and_list_videos()
        
        
    def show_disclaimer_dialog(self):
        """
        Muestra un diálogo modal con dos opciones:
         - “I Accept (remind me next time)”
         - “I Accept (do not show again)”
         - “Exit”
        Devuelve un string con “accept_once”, “accept_no_show” o None si cierra sin aceptar.
        """
        from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("StereoMaster - Important Disclaimer")

        label_text = (
            "<h2 style='color:#BB0000;'>Important Disclaimer</h2>"
            "<p><b>Use at your own risk!</b> StereoMaster is a powerful tool for creating "
            "high-quality stereo content by leveraging large-scale AI models. However, please read the following carefully:</p>"
            "<ul>"
            "<li><b>High VRAM Usage:</b> A minimum of 16GB of VRAM on an NVIDIA GPU is strongly recommended. "
            "Attempting to run on lower VRAM may lead to crashes or system freezes.</li>"
            "<li><b>Resource Intensive:</b> This application can consume significant CPU/GPU/memory resources. "
            "Your system may become unresponsive if resources are insufficient.</li>"
            "<li><b>No Warranty:</b> This software is provided <i>“as is”</i> without any express or implied warranties. "
            "The developer disclaims all liability for potential data loss, hardware issues, or system instability.</li>"
            "<li><b>Acceptance of Risk:</b> By proceeding, you acknowledge and accept that the author is "
            "not responsible for damages to your system or files, nor for the need to reboot or reinstall your OS.</li>"
            "</ul>"
            "<p>Please make sure you meet the hardware requirements and understand these terms before using StereoMaster.</p>"
            "<p>Do you wish to continue?</p>"
        )

        label = QLabel(label_text, dialog)
        label.setWordWrap(True)

        btn_accept_once = QPushButton("I Accept (remind me next time)")
        btn_accept_no_show = QPushButton("I Accept (do not show again)")
        btn_cancel = QPushButton("Exit")

        btn_accept_once.clicked.connect(lambda: self.close_dialog_return(dialog, "accept_once"))
        btn_accept_no_show.clicked.connect(lambda: self.close_dialog_return(dialog, "accept_no_show"))
        btn_cancel.clicked.connect(lambda: self.close_dialog_return(dialog, None))

        # Layout
        vbox = QVBoxLayout(dialog)
        vbox.addWidget(label)
        hbox = QHBoxLayout()
        hbox.addWidget(btn_accept_once)
        hbox.addWidget(btn_accept_no_show)
        hbox.addWidget(btn_cancel)
        vbox.addLayout(hbox)

        dialog.setLayout(vbox)
        dialog.setModal(True)
        dialog.exec_()

 
        return getattr(dialog, "_result", None)


    def close_dialog_return(self, dialog, result_value):
        # Guardamos el resultado en el dialog y cerramos
        dialog._result = result_value
        dialog.close()

    def save_disclaimer_accepted(self):
        # Aquí actualizas tu config JSON:
        # por ejemplo, si usas config_paths.json:

        try:
            # lee config_paths.json
            import json
            with open(self.paths_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            data = {}

        data["disclaimer_accepted"] = True

        # guardar
        with open(self.paths_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)        
        
        
    ###########################################################
    # LOAD/SAVE GLOBAL DEPTH PARAMS
    ###########################################################
    def load_depth_global_params(self):
        """
        Loads global depth/inpaint parameters from a JSON file.
        If not found or parse error, keep defaults.
        """
        if not os.path.isfile(self.depth_global_file):
            return
        try:
            with open(self.depth_global_file, "r", encoding="utf-8") as f:
                dd = json.load(f)
            # Overwrite from file if keys exist
            self.crafter_overlap         = dd.get("crafter_overlap", self.crafter_overlap)
            self.crafter_window          = dd.get("crafter_window", self.crafter_window)
            self.crafter_maxres_value    = dd.get("crafter_maxres_value", self.crafter_maxres_value)
            self.crafter_denoising_steps = dd.get("crafter_denoising_steps", self.crafter_denoising_steps)
            self.crafter_guidance_scale  = dd.get("crafter_guidance_scale", self.crafter_guidance_scale)
            self.crafter_seed            = dd.get("crafter_seed", self.crafter_seed)

            self.vda_use_fp16            = dd.get("vda_use_fp16", self.vda_use_fp16)
            self.vda_use_cudnn           = dd.get("vda_use_cudnn", self.vda_use_cudnn)
            self.vda_input_size          = dd.get("vda_input_size", self.vda_input_size)
            self.vda_encoder             = dd.get("vda_encoder", self.vda_encoder)
            self.vda_maxres_value        = dd.get("vda_maxres_value", self.vda_maxres_value)

            # Also load the single/batch inpaint threshold
            self.inpaint_threshold_batch  = dd.get("inpaint_threshold_batch", self.inpaint_threshold_batch)
            self.sbs_mode = dd.get("sbs_mode", "HSBS")
            self.output_encoder = dd.get("output_encoder", "x264")

        except Exception as e:
            self.log(f"[WARN] => Could not parse {self.depth_global_file}: {e}")

    def save_depth_global_params(self):
        """
        Saves the current depth/inpaint global params to a JSON file.
        """
        data = {
            "crafter_overlap":         self.crafter_overlap,
            "crafter_window":          self.crafter_window,
            "crafter_maxres_value":    self.crafter_maxres_value,
            "crafter_denoising_steps": self.crafter_denoising_steps,
            "crafter_guidance_scale":  self.crafter_guidance_scale,
            "crafter_seed":            self.crafter_seed,
            
            "vda_maxres_value":  self.vda_maxres_value, 
            "vda_use_fp16":  self.vda_use_fp16,          
            "vda_use_cudnn": self.vda_use_cudnn,
            "vda_input_size": self.vda_input_size,
            "vda_encoder":    self.vda_encoder,


            "inpaint_threshold_batch":  self.inpaint_threshold_batch,
            "sbs_mode": self.sbs_mode,
            "output_encoder": self.output_encoder,
            
        }
        try:
            with open(self.depth_global_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.log(f"[WARN] => Could not save {self.depth_global_file}: {e}")

    ###########################################################
    # LOAD/SAVE PATHS
    ###########################################################
    def load_paths(self):
        self.dir_input_videos       = "./input_videos"
        self.dir_pre_depth          = "./pre_rendered_depth"
        self.dir_pre_depth2         = "./pre_rendered_depth2"
        self.dir_pre_depth_final    = "./pre_rendered_depth_final"
        self.dir_output_splattings  = "./output_splatted_gui"
        self.dir_inpainting_out     = "./inpainting_results_gui"
        self.dir_completed_merged   = "./completed_video_merged"
        
        # Flag por defecto para el disclaimer
        self.disclaimer_accepted    = False

        if os.path.isfile(self.paths_file):
            try:
                with open(self.paths_file, "r", encoding="utf-8") as f:
                    dd = json.load(f)
                
                self.dir_input_videos       = dd.get("dir_input_videos",      self.dir_input_videos)
                self.dir_pre_depth          = dd.get("dir_pre_depth",         self.dir_pre_depth)
                self.dir_pre_depth2         = dd.get("dir_pre_depth2",        self.dir_pre_depth2)
                self.dir_pre_depth_final    = dd.get("dir_pre_depth_final",   self.dir_pre_depth_final)
                self.dir_output_splattings  = dd.get("dir_output_splattings", self.dir_output_splattings)
                self.dir_inpainting_out     = dd.get("dir_inpainting_out",    self.dir_inpainting_out)
                self.dir_completed_merged   = dd.get("dir_completed_merged",  self.dir_completed_merged)

            
                self.disclaimer_accepted = dd.get("disclaimer_accepted", False)

            except Exception as e:
                self.log("[WARN] => can't parse config_paths.json:", e)
        else:
            self.log("[INFO] => No config_paths.json found, using defaults.")



    def save_paths(self):
        dd={
            "dir_input_videos": self.input_videos_edit.text(),
            "dir_pre_depth": self.pre_depth_edit.text(),
            "dir_pre_depth2": self.pre_depth2_edit.text(),
            "dir_pre_depth_final": self.pre_depth_final_edit.text(),
            "dir_output_splattings": self.output_splattings_edit.text(),
            "dir_inpainting_out": self.inpainting_out_edit.text(),
            "dir_completed_merged": self.completed_merged_edit.text(),
            "disclaimer_accepted": self.disclaimer_accepted
        }
        try:
            with open(self.paths_file,"w",encoding="utf-8") as f:
                json.dump(dd,f,indent=2)
        except:
            pass

    ###########################################################
    # LOAD/SAVE PER-VIDEO PARAMS
    ###########################################################
    def load_params(self):
        if os.path.isfile(self.settings_file):
            try:
                with open(self.settings_file,"r",encoding="utf-8") as f:
                    self.video_params=json.load(f)
            except:
                self.log("[WARN] => can't parse video_params.json")
                self.video_params={}
        else:
            self.video_params={}

    def save_params(self):
        try:
            with open(self.settings_file,"w",encoding="utf-8") as f:
                json.dump(self.video_params,f,indent=2)
        except:
            pass

    def build_state_combos(self):
        """
        Construye la lista de filtros fijos solicitados.
        """
        self.all_state_combos = [
            "ALL",
            "ORIG",
            "DEPTH",
            "SPLATTED",
            "INPAINTED",
            "ORIG + DEPTH",
            "ORIG + DEPTH + SPLATTED",
            "ORIG + DEPTH + SPLATTED + INPAINTED"
        ]


    def set_styles(self):

        self.setStyleSheet("""
            QMainWindow {
                background-color: #F0F0F0;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                font-weight: bold;
                font-size: 12px;
                border-radius: 5px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #1F78B4;
            }
            QLabel {
                font-size: 12px;
            }
            QPlainTextEdit {
                background-color: #FAFAFA;
                font-size: 11px;
            }
            QCheckBox {
                font-size: 11px;
            }
            QComboBox {
                font-size: 11px;
            }
            QLineEdit {
                font-size: 11px;              
            }
            
            QSlider::groove:horizontal {
                height: 5px;
                background: #D0D0D0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background-color: #3498DB;
                border: 1px solid #444;
                width: 14px;
                height: 14px;
                border-radius: 7px; 
                margin: -5px 0; 
            }

            QSlider::handle:horizontal:enabled {
                border-radius: 7px;
            }
            QSlider::handle:horizontal:disabled {
                border-radius: 7px;
            }
        """)
        
        self.log_box.setStyleSheet("""
            QPlainTextEdit {
                background-color: #000000;
                color: #FFFFFF;
                font-family: Consolas, "Courier New", monospace;
                font-size: 12px;
        }
        """)

    ###########################################################
    # UI
    ###########################################################
    def initUI(self):
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        ##############################
        # (A) Left Panel => Folders, List, Scenes
        ##############################
        left_layout= QVBoxLayout()
        
     
        project_box = QGroupBox("Project Management")
        vproj_layout = QVBoxLayout()

      
        row_project_sel = QHBoxLayout()
        lbl_proj = QLabel("Select Project:")
        self.combo_projects = QComboBox()
        self.combo_projects.currentIndexChanged.connect(self.on_project_selected)

        row_project_sel.addWidget(lbl_proj)
        row_project_sel.addWidget(self.combo_projects)

        vproj_layout.addLayout(row_project_sel)


        btn_create_project = QPushButton("Create New Project")
        btn_create_project.setFixedHeight(28)
        btn_create_project.clicked.connect(self.on_create_project_clicked)
        vproj_layout.addWidget(btn_create_project)

        project_box.setLayout(vproj_layout)
        left_layout.addWidget(project_box)
  


        dir_box= QGroupBox("Folders config")
        form_lay= QFormLayout()

        # --- Row Input Videos ---
        row_input= QHBoxLayout()
        self.input_videos_edit= QLineEdit(self.dir_input_videos)
        btn_go_input= QPushButton("GO")
        btn_go_input.setFixedWidth(32)
        btn_go_input.clicked.connect(lambda: self.open_folder(self.input_videos_edit.text()))
        btn_browse_input= QPushButton("...")
        btn_browse_input.setFixedWidth(32)
        btn_browse_input.clicked.connect(lambda: self.browse_and_set_folder(self.input_videos_edit))
        row_input.addWidget(self.input_videos_edit)
        row_input.addWidget(btn_go_input)
        row_input.addWidget(btn_browse_input)
        form_lay.addRow("Input Videos:", row_input)

        # --- Row Depth1 ---
        row_depth= QHBoxLayout()
        self.pre_depth_edit= QLineEdit(self.dir_pre_depth)
        btn_go_depth= QPushButton("GO")
        btn_go_depth.setFixedWidth(32)
        btn_go_depth.clicked.connect(lambda: self.open_folder(self.pre_depth_edit.text()))
        btn_browse_depth= QPushButton("...")
        btn_browse_depth.setFixedWidth(32)
        btn_browse_depth.clicked.connect(lambda: self.browse_and_set_folder(self.pre_depth_edit))
        row_depth.addWidget(self.pre_depth_edit)
        row_depth.addWidget(btn_go_depth)
        row_depth.addWidget(btn_browse_depth)
        form_lay.addRow("Depth1 DepthCrafter:", row_depth)

        # --- Row Depth2 ---
        row_depth2= QHBoxLayout()
        self.pre_depth2_edit= QLineEdit(self.dir_pre_depth2)
        btn_go_depth2= QPushButton("GO")
        btn_go_depth2.setFixedWidth(32)
        btn_go_depth2.clicked.connect(lambda: self.open_folder(self.pre_depth2_edit.text()))
        btn_browse_depth2= QPushButton("...")
        btn_browse_depth2.setFixedWidth(32)
        btn_browse_depth2.clicked.connect(lambda: self.browse_and_set_folder(self.pre_depth2_edit))
        row_depth2.addWidget(self.pre_depth2_edit)
        row_depth2.addWidget(btn_go_depth2)
        row_depth2.addWidget(btn_browse_depth2)
        form_lay.addRow("Depth2 VDA:", row_depth2)

        # --- Row Depth Merged ---
        row_depthf= QHBoxLayout()
        self.pre_depth_final_edit= QLineEdit(self.dir_pre_depth_final)
        btn_go_depthf= QPushButton("GO")
        btn_go_depthf.setFixedWidth(32)
        btn_go_depthf.clicked.connect(lambda: self.open_folder(self.pre_depth_final_edit.text()))
        btn_browse_depthf= QPushButton("...")
        btn_browse_depthf.setFixedWidth(32)
        btn_browse_depthf.clicked.connect(lambda: self.browse_and_set_folder(self.pre_depth_final_edit))
        row_depthf.addWidget(self.pre_depth_final_edit)
        row_depthf.addWidget(btn_go_depthf)
        row_depthf.addWidget(btn_browse_depthf)
        form_lay.addRow("DepthMerged:", row_depthf)

        # --- Row Splatted ---
        row_spl= QHBoxLayout()
        self.output_splattings_edit= QLineEdit(self.dir_output_splattings)
        btn_go_spl= QPushButton("GO")
        btn_go_spl.setFixedWidth(32)
        btn_go_spl.clicked.connect(lambda: self.open_folder(self.output_splattings_edit.text()))
        btn_browse_spl= QPushButton("...")
        btn_browse_spl.setFixedWidth(32)
        btn_browse_spl.clicked.connect(lambda: self.browse_and_set_folder(self.output_splattings_edit))
        row_spl.addWidget(self.output_splattings_edit)
        row_spl.addWidget(btn_go_spl)
        row_spl.addWidget(btn_browse_spl)
        form_lay.addRow("Splatted:", row_spl)

        # --- Row Inpainting ---
        row_inp= QHBoxLayout()
        self.inpainting_out_edit= QLineEdit(self.dir_inpainting_out)
        btn_go_inp= QPushButton("GO")
        btn_go_inp.setFixedWidth(32)
        btn_go_inp.clicked.connect(lambda: self.open_folder(self.inpainting_out_edit.text()))
        btn_browse_inp= QPushButton("...")
        btn_browse_inp.setFixedWidth(32)
        btn_browse_inp.clicked.connect(lambda: self.browse_and_set_folder(self.inpainting_out_edit))
        row_inp.addWidget(self.inpainting_out_edit)
        row_inp.addWidget(btn_go_inp)
        row_inp.addWidget(btn_browse_inp)
        form_lay.addRow("Inpainted:", row_inp)

        # --- Row Completed Merged ---
        row_merged= QHBoxLayout()
        self.completed_merged_edit= QLineEdit(self.dir_completed_merged)
        btn_go_merged= QPushButton("GO")
        btn_go_merged.setFixedWidth(32)
        btn_go_merged.clicked.connect(lambda: self.open_folder(self.completed_merged_edit.text()))
        btn_browse_merged= QPushButton("...")
        btn_browse_merged.setFixedWidth(32)
        btn_browse_merged.clicked.connect(lambda: self.browse_and_set_folder(self.completed_merged_edit))
        row_merged.addWidget(self.completed_merged_edit)
        row_merged.addWidget(btn_go_merged)
        row_merged.addWidget(btn_browse_merged)
        form_lay.addRow("Completed Merged:", row_merged)

        dir_box.setLayout(form_lay)
        left_layout.addWidget(dir_box)

        row_btns_top= QHBoxLayout()
        btn_save_paths= QPushButton("Save Paths")
        btn_save_paths.setFixedHeight(28)
        btn_save_paths.clicked.connect(self.save_paths)
        row_btns_top.addWidget(btn_save_paths)

        btn_refresh_list= QPushButton("Refresh")
        btn_refresh_list.setFixedHeight(28)
        btn_refresh_list.clicked.connect(self.scan_and_list_videos)
        row_btns_top.addWidget(btn_refresh_list)
        left_layout.addLayout(row_btns_top)

        btn_select_all_filtered= QPushButton("Select All Filtered")
        btn_select_all_filtered.setFixedHeight(28)
        btn_select_all_filtered.clicked.connect(self.select_all_filtered)
        left_layout.addWidget(btn_select_all_filtered)
        
        btn_unselect_all_filtered= QPushButton("Unselect All Filtered")
        btn_unselect_all_filtered.setFixedHeight(28)
        btn_unselect_all_filtered.clicked.connect(self.unselect_all_filtered)
        left_layout.addWidget(btn_unselect_all_filtered)
        
        self.btn_delete_selected = QPushButton("Delete Selected")
        self.btn_delete_selected.setFixedHeight(28)
        self.btn_delete_selected.clicked.connect(self.on_delete_selected_clicked)
        left_layout.addWidget(self.btn_delete_selected)

        self.filter_combo= QComboBox()
        self.filter_combo.currentIndexChanged.connect(self.apply_filter_list)
        left_layout.addWidget(self.filter_combo)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.itemClicked.connect(self.on_item_selected)
        self.list_widget.setMinimumHeight(400)
        left_layout.addWidget(self.list_widget)

        btn_load_newvideo= QPushButton("Load Video -> Input Folder")
        btn_load_newvideo.setFixedHeight(28)
        btn_load_newvideo.clicked.connect(self.load_new_video_to_input)
        left_layout.addWidget(btn_load_newvideo)

        row_scene = QHBoxLayout()
        self.label_threshold = QLabel("Threshold:")
        self.combo_scene_threshold = QComboBox()
        self.combo_scene_threshold.addItems(["10","15","20","25","30","35","45"])

        self.btn_scene_detect = QPushButton("Scene Detect / Split")
        self.btn_scene_detect.setFixedHeight(28)
        self.btn_scene_detect.clicked.connect(self.do_scene_detect)

        row_scene.addWidget(self.btn_scene_detect)
        row_scene.addWidget(self.label_threshold)
        row_scene.addWidget(self.combo_scene_threshold)
   

        left_layout.addLayout(row_scene)


        left_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        main_layout.addLayout(left_layout)

        ##############################
        # (B) Center => Preview + Logs
        ##############################
        center_layout= QVBoxLayout()

        self.label_preview= QLabel("Preview")
        self.label_preview.setFixedSize(1000,640)
        self.label_preview.setStyleSheet("background-color:black;")
        center_layout.addWidget(self.label_preview)

        row_h= QHBoxLayout()
        self.lbl_slider_frames= QLabel("Frame:")

        # Use KeyframeSlider here
        self.slider_frames= KeyframeSlider(Qt.Horizontal)
        self.slider_frames.valueChanged.connect(self.on_slider_frame_changed)

        row_h.addWidget(self.lbl_slider_frames)
        row_h.addWidget(self.slider_frames)
        center_layout.addLayout(row_h)

        row_preview_opts = QHBoxLayout()
        lbl_opts = QLabel("Preview:")

        self.chk_orig      = QCheckBox("Orig")
        self.chk_depth     = QCheckBox("Depth")
        self.chk_polylines = QCheckBox("Polylines")
        self.chk_anaglyph  = QCheckBox("Forward Warp")
        self.chk_splatted  = QCheckBox("Splatted")
        self.chk_inpaint   = QCheckBox("Completed/Inpainted")
      

   
        self.preview_group = QButtonGroup(self)
        self.preview_group.setExclusive(True)

    
        self.preview_group.addButton(self.chk_orig)
        self.preview_group.addButton(self.chk_depth)
        self.preview_group.addButton(self.chk_splatted)
        self.preview_group.addButton(self.chk_inpaint)
        self.preview_group.addButton(self.chk_polylines)
        self.preview_group.addButton(self.chk_anaglyph)

    
        self.chk_orig.setChecked(True)

     
        # (cualquier cambio llamará a show_preview)
        self.chk_orig.stateChanged.connect(lambda s: self.show_preview(self.current_frame_idx))
        self.chk_depth.stateChanged.connect(lambda s: self.show_preview(self.current_frame_idx))
        self.chk_splatted.stateChanged.connect(lambda s: self.show_preview(self.current_frame_idx))
        self.chk_inpaint.stateChanged.connect(lambda s: self.show_preview(self.current_frame_idx))
        self.chk_polylines.stateChanged.connect(lambda s: self.show_preview(self.current_frame_idx))
        self.chk_anaglyph.stateChanged.connect(lambda s: self.show_preview(self.current_frame_idx))

        # Los metemos en un layout horizontal
        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Preview mode:"))
        row_mode.addWidget(self.chk_orig)
        row_mode.addWidget(self.chk_depth)
        row_mode.addWidget(self.chk_anaglyph)
        row_mode.addWidget(self.chk_polylines)
        row_mode.addWidget(self.chk_splatted)
        row_mode.addWidget(self.chk_inpaint)


        center_layout.addLayout(row_mode)


        self.preview_depth_combo= QComboBox()
        self.preview_depth_combo.addItems(["Depth1", "Depth2", "Merged"])
        self.preview_depth_combo.setCurrentIndex(0)
        self.preview_depth_combo.currentIndexChanged.connect(self.on_preview_depth_changed)
        center_layout.addWidget(self.preview_depth_combo)

        row_fus= QHBoxLayout()
        self.lbl_fusion= QLabel("Merge alpha:")
        self.fusion_slider= QSlider(Qt.Horizontal)
        self.fusion_slider.setRange(0,100)
        self.fusion_slider.setValue(self.fusion_alpha)
        self.fusion_slider.valueChanged.connect(self.on_fusion_alpha_changed)
        row_fus.addWidget(self.lbl_fusion)
        row_fus.addWidget(self.fusion_slider)
        center_layout.addLayout(row_fus)

        self.log_box= QPlainTextEdit()
        self.log_box.setReadOnly(True)
        center_layout.addWidget(self.log_box)

        row_prog= QHBoxLayout()
        self.batch_progress_bar= QProgressBar()
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setRange(0,1)
        row_prog.addWidget(self.batch_progress_bar)

        self.btn_cancel_batch= QPushButton("Cancel Batch")
        self.btn_cancel_batch.setFixedHeight(28)
        self.btn_cancel_batch.clicked.connect(self.cancel_current_batch)
        row_prog.addWidget(self.btn_cancel_batch)
        center_layout.addLayout(row_prog)

        center_layout.addSpacerItem(QSpacerItem(20,20,QSizePolicy.Minimum,QSizePolicy.Expanding))
        main_layout.addLayout(center_layout)

        ##############################
        # (C) Right => TABS
        ##############################
        right_tab_widget= QTabWidget()
        right_tab_widget.setMinimumWidth(320)

        # Tab 1 => Depth Tools
        tab_depth_tools= QWidget()
        v_layout_dt= QVBoxLayout(tab_depth_tools)

        # --- VDA box
        vda_box= QGroupBox("VDA Params")
        vda_form= QFormLayout()

        self.combo_vda_res = QComboBox()
        for x_ in [512,640,768,960,1024,1280,1478,1920]:
            self.combo_vda_res.addItem(str(x_))
        self.combo_vda_res.setCurrentText(str(self.vda_maxres_value))
        self.combo_vda_res.currentIndexChanged.connect(self.on_vda_res_changed)
        vda_form.addRow("MaxRes:", self.combo_vda_res)

        self.combo_vda_input_size= QComboBox()
        for x_ in [512,640,768]:
            self.combo_vda_input_size.addItem(str(x_))
        self.combo_vda_input_size.setCurrentText(str(self.vda_input_size))
        self.combo_vda_input_size.currentIndexChanged.connect(self.on_vda_input_size_changed)
        vda_form.addRow("InputSize:", self.combo_vda_input_size)

        self.chk_vda_fp16= QCheckBox("FP16")
        self.chk_vda_fp16.setChecked(self.vda_use_fp16)
        self.chk_vda_fp16.stateChanged.connect(lambda s: setattr(self,'vda_use_fp16', bool(s)))
        vda_form.addRow(self.chk_vda_fp16)

        self.chk_vda_cudnn= QCheckBox("cuDNN Bench")
        self.chk_vda_cudnn.setChecked(self.vda_use_cudnn)
        self.chk_vda_cudnn.stateChanged.connect(lambda s: setattr(self,'vda_use_cudnn', bool(s)))
        vda_form.addRow(self.chk_vda_cudnn)

        self.combo_vda_enc= QComboBox()
        self.combo_vda_enc.addItems(["vits","vitl"])
        self.combo_vda_enc.setCurrentText(self.vda_encoder)
        self.combo_vda_enc.currentIndexChanged.connect(self.on_vda_encoder_changed)
        vda_form.addRow("Encoder:", self.combo_vda_enc)

        vda_box.setLayout(vda_form)
        v_layout_dt.addWidget(vda_box)

        # --- Crafter box
        crafter_box= QGroupBox("DepthCrafter Params")
        crafter_form= QFormLayout()
        self.combo_overlap= QComboBox()
        for ov_ in [0,10,25,40,70,100]:
            self.combo_overlap.addItem(str(ov_))
        self.combo_overlap.setCurrentText(str(self.crafter_overlap))
        self.combo_overlap.currentIndexChanged.connect(self.on_crafter_overlap_changed)
        crafter_form.addRow("Overlap:", self.combo_overlap)

        self.combo_window= QComboBox()
        for w_ in [20,40,70,100,150]:
            self.combo_window.addItem(str(w_))
        self.combo_window.setCurrentText(str(self.crafter_window))
        self.combo_window.currentIndexChanged.connect(self.on_crafter_window_changed)
        crafter_form.addRow("Window Size:", self.combo_window)

        self.max_res_combo = QComboBox()
        for x_ in [512,640,768,960,1024,1280,1478]:
            self.max_res_combo.addItem(str(x_))
        self.max_res_combo.setCurrentText(str(self.crafter_maxres_value))
        self.max_res_combo.currentIndexChanged.connect(self.on_maxres_changed)
        crafter_form.addRow("MaxRes:", self.max_res_combo)

        self.spin_crafter_denoising = QSpinBox()
        self.spin_crafter_denoising.setRange(1,999)
        self.spin_crafter_denoising.setValue(self.crafter_denoising_steps)
        self.spin_crafter_denoising.valueChanged.connect(self.on_crafter_denoising_changed)
        crafter_form.addRow("Denoising Steps:", self.spin_crafter_denoising)

        self.spin_crafter_guidance = QDoubleSpinBox()
        self.spin_crafter_guidance.setDecimals(1)
        self.spin_crafter_guidance.setRange(1.0, 1.3)
        self.spin_crafter_guidance.setSingleStep(0.1)
        self.spin_crafter_guidance.setValue(self.crafter_guidance_scale)

        # Conecta el spinbox a tu función "on_crafter_guidance_changed"
        self.spin_crafter_guidance.valueChanged.connect(self.on_crafter_guidance_changed)

        crafter_form.addRow("Guidance Scale:", self.spin_crafter_guidance)


        self.spin_crafter_seed = QSpinBox()
        self.spin_crafter_seed.setRange(0,999999999)
        self.spin_crafter_seed.setValue(self.crafter_seed)
        self.spin_crafter_seed.valueChanged.connect(self.on_crafter_seed_changed)
        crafter_form.addRow("Seed:", self.spin_crafter_seed)

        crafter_box.setLayout(crafter_form)
        v_layout_dt.addWidget(crafter_box)
        
        # === Batch Depth Source ===
        row_bd = QHBoxLayout()
        lbl_bd = QLabel("Depth Model:") 
        self.combo_batch_depth_source = QComboBox()
        self.combo_batch_depth_source.addItems(["DepthCrafter","VDA","Both"])
        row_bd.addWidget(lbl_bd)
        row_bd.addWidget(self.combo_batch_depth_source)
        v_layout_dt.addLayout(row_bd)
        
        self.check_overwrite_depth = QCheckBox("Overwrite existing depth (Don't ask)")
        self.check_overwrite_depth.setChecked(False)
        v_layout_dt.addWidget(self.check_overwrite_depth)

        btn_batch_depth = QPushButton("Generate Depth")
        btn_batch_depth.setFixedHeight(28)
        btn_batch_depth.clicked.connect(self.batch_generate_depth_selected)
        v_layout_dt.addWidget(btn_batch_depth)
        
        self.btn_merge_depths= QPushButton("Merge Depth1/Depth2 => Merged")
        self.btn_merge_depths.setFixedHeight(28)
        self.btn_merge_depths.clicked.connect(self.merge_depths)
        v_layout_dt.addWidget(self.btn_merge_depths)
       

        v_layout_dt.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        right_tab_widget.addTab(tab_depth_tools, "Depth Tools")

        # Tab 3 => Sliders
        tab_sliders = QWidget()
        v_layout_sl = QVBoxLayout(tab_sliders)

        # === 1) Disparity (slider_disp) ===
        disp_layout = QHBoxLayout()
        lbl_disp = QLabel("Disparity:")
        self.slider_disp = QSlider(Qt.Horizontal)
        self.slider_disp.setRange(0, 100)
        self.slider_disp.setValue(int(self.disp_value))
        self.slider_disp.valueChanged.connect(self.on_disp_slider_changed)
        self.lbl_disp_val = QLabel(str(self.disp_value))
        btn_reset_disp = QPushButton("R")
        btn_reset_disp.setFixedWidth(28)
        btn_reset_disp.clicked.connect(self.reset_disp)

        disp_layout.addWidget(lbl_disp)
        disp_layout.addWidget(self.slider_disp)
        disp_layout.addWidget(self.lbl_disp_val)
        disp_layout.addWidget(btn_reset_disp)

        v_layout_sl.addLayout(disp_layout)

        # === 2) Convergence (slider_conv) ===
        conv_layout = QHBoxLayout()
        lbl_conv = QLabel("Convergence:")
        self.slider_conv = QSlider(Qt.Horizontal)
        self.slider_conv.setRange(-50, 50)
        self.slider_conv.setValue(int(self.conv_value))
        self.slider_conv.valueChanged.connect(self.on_conv_slider_changed)
        self.lbl_conv_val = QLabel(str(self.conv_value))
        btn_reset_conv = QPushButton("R")
        btn_reset_conv.setFixedWidth(28)
        btn_reset_conv.clicked.connect(self.reset_conv)

        conv_layout.addWidget(lbl_conv)
        conv_layout.addWidget(self.slider_conv)
        conv_layout.addWidget(self.lbl_conv_val)
        conv_layout.addWidget(btn_reset_conv)

        v_layout_sl.addLayout(conv_layout)

        # === 3) Depth Bri (slider_brightness) ===
        bri_layout = QHBoxLayout()
        lbl_bri_d = QLabel("Depth Bri:")
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(0, 200)
        self.slider_brightness.setValue(int(self.brightness_value * 100))
        self.slider_brightness.valueChanged.connect(self.on_brightness_slider_changed)
        self.lbl_bri_val = QLabel(str(self.brightness_value))
        btn_reset_bri = QPushButton("R")
        btn_reset_bri.setFixedWidth(28)
        btn_reset_bri.clicked.connect(self.reset_brightness_depth)

        bri_layout.addWidget(lbl_bri_d)
        bri_layout.addWidget(self.slider_brightness)
        bri_layout.addWidget(self.lbl_bri_val)
        bri_layout.addWidget(btn_reset_bri)

        v_layout_sl.addLayout(bri_layout)

        # === 4) Depth Gam (slider_gamma) ===
        gam_layout = QHBoxLayout()
        lbl_gam_d = QLabel("Depth Gam:")
        self.slider_gamma = QSlider(Qt.Horizontal)
        self.slider_gamma.setRange(1, 300)
        self.slider_gamma.setValue(int(self.gamma_value * 100))
        self.slider_gamma.valueChanged.connect(self.on_gamma_slider_changed)
        self.lbl_gam_val = QLabel(str(self.gamma_value))
        btn_reset_gam = QPushButton("R")
        btn_reset_gam.setFixedWidth(28)
        btn_reset_gam.clicked.connect(self.reset_gamma_depth)

        gam_layout.addWidget(lbl_gam_d)
        gam_layout.addWidget(self.slider_gamma)
        gam_layout.addWidget(self.lbl_gam_val)
        gam_layout.addWidget(btn_reset_gam)

        v_layout_sl.addLayout(gam_layout)

        # === 5) Dilat.H(px) (slider_dilate_h) ===
        dh_layout = QHBoxLayout()
        lbl_dh = QLabel("Dilat.H(px):")
        self.slider_dilate_h = QSlider(Qt.Horizontal)
        self.slider_dilate_h.setRange(0, 50)
        self.slider_dilate_h.setValue(self.dilate_h_value)
        self.slider_dilate_h.valueChanged.connect(self.on_dilate_h_changed)
        self.lbl_dh_val = QLabel(str(self.dilate_h_value))
        btn_reset_dh = QPushButton("R")
        btn_reset_dh.setFixedWidth(28)
        btn_reset_dh.clicked.connect(self.reset_dilate_h)

        dh_layout.addWidget(lbl_dh)
        dh_layout.addWidget(self.slider_dilate_h)
        dh_layout.addWidget(self.lbl_dh_val)
        dh_layout.addWidget(btn_reset_dh)

        v_layout_sl.addLayout(dh_layout)

        # === 6) Dilat.V(px) (slider_dilate_v) ===
        dv_layout = QHBoxLayout()
        lbl_dv = QLabel("Dilat.V(px):")
        self.slider_dilate_v = QSlider(Qt.Horizontal)
        self.slider_dilate_v.setRange(0, 50)
        self.slider_dilate_v.setValue(self.dilate_v_value)
        self.slider_dilate_v.valueChanged.connect(self.on_dilate_v_changed)
        self.lbl_dv_val = QLabel(str(self.dilate_v_value))
        btn_reset_dv = QPushButton("R")
        btn_reset_dv.setFixedWidth(28)
        btn_reset_dv.clicked.connect(self.reset_dilate_v)

        dv_layout.addWidget(lbl_dv)
        dv_layout.addWidget(self.slider_dilate_v)
        dv_layout.addWidget(self.lbl_dv_val)
        dv_layout.addWidget(btn_reset_dv)

        v_layout_sl.addLayout(dv_layout)

        # === 7) BlurK(px) (slider_blur_ksize) ===
        bk_layout = QHBoxLayout()
        lbl_bk = QLabel("BlurK(px):")
        self.slider_blur_ksize = QSlider(Qt.Horizontal)
        self.slider_blur_ksize.setRange(0, 30)
        self.slider_blur_ksize.setValue(self.blur_ksize_value)
        self.slider_blur_ksize.valueChanged.connect(self.on_blur_ksize_changed)
        self.lbl_bk_val = QLabel(str(self.blur_ksize_value))
        btn_reset_bk = QPushButton("R")
        btn_reset_bk.setFixedWidth(28)
        btn_reset_bk.clicked.connect(self.reset_blur_ksize)

        bk_layout.addWidget(lbl_bk)
        bk_layout.addWidget(self.slider_blur_ksize)
        bk_layout.addWidget(self.lbl_bk_val)
        bk_layout.addWidget(btn_reset_bk)

        v_layout_sl.addLayout(bk_layout)

        # === 8) BlurSigma (slider_blur_sigma) ===
        bs_layout = QHBoxLayout()
        lbl_bs = QLabel("BlurSigma:")
        self.slider_blur_sigma = QSlider(Qt.Horizontal)
        self.slider_blur_sigma.setRange(0, 50)
        self.slider_blur_sigma.setValue(int(self.blur_sigma_value * 10))
        self.slider_blur_sigma.valueChanged.connect(self.on_blur_sigma_changed)
        self.lbl_bs_val = QLabel(str(self.blur_sigma_value))
        btn_reset_bs = QPushButton("R")
        btn_reset_bs.setFixedWidth(28)
        btn_reset_bs.clicked.connect(self.reset_blur_sigma)

        bs_layout.addWidget(lbl_bs)
        bs_layout.addWidget(self.slider_blur_sigma)
        bs_layout.addWidget(self.lbl_bs_val)
        bs_layout.addWidget(btn_reset_bs)

        v_layout_sl.addLayout(bs_layout)

        warp_box = QGroupBox("Forward Warp Weighted")
        warp_layout = QVBoxLayout()

        self.slider_warp_exponent = QSlider(Qt.Horizontal)
        self.slider_warp_exponent.setRange(100, 900)  # i.e. 1.00..3.00 => factor= value/100
        self.slider_warp_exponent.setValue(141)       # default => 1.41 ~ sqrt(2)
        self.slider_warp_exponent.valueChanged.connect(self.on_warp_exponent_changed)

        self.lbl_warp_exponent_val = QLabel("1.41")

        btn_reset_we = QPushButton("R")
        btn_reset_we.setFixedWidth(28)
        btn_reset_we.clicked.connect(self.reset_warp_exponent)

        row_wexp = QHBoxLayout()
        row_wexp.addWidget(QLabel("Base:"))
        row_wexp.addWidget(self.slider_warp_exponent)
        row_wexp.addWidget(self.lbl_warp_exponent_val)
        row_wexp.addWidget(btn_reset_we)

        warp_layout.addLayout(row_wexp)
        warp_box.setLayout(warp_layout)
        v_layout_sl.addWidget(warp_box)  # to place it in your tab_sliders

        
        
        # === 9) Polylines Box (ya estaba) ===
        polylines_box = QGroupBox("Polylines")
        poly_layout = QVBoxLayout(polylines_box)

        # (A) Slider stereo exponent
        row_exp = QHBoxLayout()
        lbl_exp = QLabel("Stereo Exponent:")
        self.slider_poly_exp = QSlider(Qt.Horizontal)
        self.slider_poly_exp.setRange(1, 300)   # 1..300 => factor value/100 => 0.01..3.0
        self.slider_poly_exp.setValue(100)      # 1.00 por defecto
        self.slider_poly_exp.valueChanged.connect(self.on_poly_exp_changed)
        self.lbl_poly_exp_val = QLabel("1.00")

        btn_reset_exp = QPushButton("R")
        btn_reset_exp.setFixedWidth(28)
        btn_reset_exp.clicked.connect(self.reset_poly_exp)

        row_exp.addWidget(lbl_exp)
        row_exp.addWidget(self.slider_poly_exp)
        row_exp.addWidget(self.lbl_poly_exp_val)
        row_exp.addWidget(btn_reset_exp)
        poly_layout.addLayout(row_exp)

        # (B) Combo fill technique
        row_fill = QHBoxLayout()
        lbl_fill = QLabel("Fill Technique:")
        self.combo_poly_fill = QComboBox()
        self.combo_poly_fill.addItems(["polylines", "polylines_sharp"])
        self.combo_poly_fill.setCurrentIndex(0)
        self.combo_poly_fill.currentIndexChanged.connect(self.on_poly_fill_changed)

        row_fill.addWidget(lbl_fill)
        row_fill.addWidget(self.combo_poly_fill)
        poly_layout.addLayout(row_fill)

        polylines_box.setLayout(poly_layout)
        v_layout_sl.addWidget(polylines_box)


        # === 10) Nuevo GroupBox "For Depth Map Creation" (para Orig Bri / Orig Gam) ===
        depth_map_box = QGroupBox("For Depth Map Creation")
        depth_map_layout = QVBoxLayout(depth_map_box)

        # 10A) Orig Bri
        ob_layout = QHBoxLayout()
        lbl_ob = QLabel("Orig Bri:")
        self.slider_orig_bri = QSlider(Qt.Horizontal)
        self.slider_orig_bri.setRange(0, 200)
        self.slider_orig_bri.setValue(100)
        self.slider_orig_bri.valueChanged.connect(self.on_orig_brightness_changed)
        self.lbl_orig_bri_val = QLabel("1.0")
        btn_reset_ob = QPushButton("R")
        btn_reset_ob.setFixedWidth(28)
        btn_reset_ob.clicked.connect(self.reset_orig_bri)

        ob_layout.addWidget(lbl_ob)
        ob_layout.addWidget(self.slider_orig_bri)
        ob_layout.addWidget(self.lbl_orig_bri_val)
        ob_layout.addWidget(btn_reset_ob)

        depth_map_layout.addLayout(ob_layout)

        # 10B) Orig Gam
        og_layout = QHBoxLayout()
        lbl_og = QLabel("Orig Gam:")
        self.slider_orig_gam = QSlider(Qt.Horizontal)
        self.slider_orig_gam.setRange(1, 300)
        self.slider_orig_gam.setValue(100)
        self.slider_orig_gam.valueChanged.connect(self.on_orig_gamma_changed)
        self.lbl_orig_gam_val = QLabel("1.0")
        btn_reset_og = QPushButton("R")
        btn_reset_og.setFixedWidth(28)
        btn_reset_og.clicked.connect(self.reset_orig_gamma)

        og_layout.addWidget(lbl_og)
        og_layout.addWidget(self.slider_orig_gam)
        og_layout.addWidget(self.lbl_orig_gam_val)
        og_layout.addWidget(btn_reset_og)

        depth_map_layout.addLayout(og_layout)

        v_layout_sl.addWidget(depth_map_box)


        # === 11) Nuevo GroupBox "For Edit in Preview" (Preview Bri / Preview Gam) ===
        preview_box = QGroupBox("For Edit in Preview")
        preview_box_layout = QVBoxLayout(preview_box)

        # 11A) Preview Bri
        pbri_layout = QHBoxLayout()
        lbl_pbri = QLabel("Preview Bri:")
        self.slider_preview_bri = QSlider(Qt.Horizontal)
        self.slider_preview_bri.setRange(0, 200)  # factor 0..2.0
        self.slider_preview_bri.setValue(100)     # 1.0 factor por defecto
        self.slider_preview_bri.valueChanged.connect(self.on_preview_bri_changed)

        self.lbl_preview_bri_val = QLabel("1.0")
        btn_reset_pbri = QPushButton("R")
        btn_reset_pbri.setFixedWidth(28)
        btn_reset_pbri.clicked.connect(self.reset_preview_bri)

        pbri_layout.addWidget(lbl_pbri)
        pbri_layout.addWidget(self.slider_preview_bri)
        pbri_layout.addWidget(self.lbl_preview_bri_val)
        pbri_layout.addWidget(btn_reset_pbri)

        preview_box_layout.addLayout(pbri_layout)

        # 11B) Preview Gam
        pgam_layout = QHBoxLayout()
        lbl_pgam = QLabel("Preview Gam:")
        self.slider_preview_gamma = QSlider(Qt.Horizontal)
        self.slider_preview_gamma.setRange(10, 300)  # 0.1 .. 3.0
        self.slider_preview_gamma.setValue(100)      # 1.0
        self.slider_preview_gamma.valueChanged.connect(self.on_preview_gamma_changed)

        self.lbl_preview_gamma_val = QLabel("1.0")
        btn_reset_pgam = QPushButton("R")
        btn_reset_pgam.setFixedWidth(28)
        btn_reset_pgam.clicked.connect(self.reset_preview_gamma)

        pgam_layout.addWidget(lbl_pgam)
        pgam_layout.addWidget(self.slider_preview_gamma)
        pgam_layout.addWidget(self.lbl_preview_gamma_val)
        pgam_layout.addWidget(btn_reset_pgam)

        preview_box_layout.addLayout(pgam_layout)

        v_layout_sl.addWidget(preview_box)


        # === 12) Keyframes GroupBox ===
        keyf_box = QGroupBox("Keyframes")
        keyf_layout = QVBoxLayout()

        self.btn_add_keyframe = QPushButton("Add Keyframe (current frame)")
        self.btn_add_keyframe.clicked.connect(self.add_keyframe)
        keyf_layout.addWidget(self.btn_add_keyframe)

        self.keyframes_combo = QComboBox()
        self.keyframes_combo.setToolTip("Select a keyframe to jump or remove it.")
        self.keyframes_combo.currentIndexChanged.connect(self.on_keyframe_combo_selected)
        keyf_layout.addWidget(self.keyframes_combo)

        row_kf_btns = QHBoxLayout()
        self.btn_remove_kf = QPushButton("Remove Selected KF")
        self.btn_remove_kf.clicked.connect(self.remove_selected_keyframe)
        row_kf_btns.addWidget(self.btn_remove_kf)

        self.btn_clear_kf = QPushButton("Clear All KF")
        self.btn_clear_kf.clicked.connect(self.clear_all_keyframes)
        row_kf_btns.addWidget(self.btn_clear_kf)

        keyf_layout.addLayout(row_kf_btns)
        keyf_box.setLayout(keyf_layout)
        v_layout_sl.addWidget(keyf_box)

        # === 13) Presets Box (ahora inmediatamente después de Keyframes) ===
        presets_box = QGroupBox("Presets")
        presets_layout = QVBoxLayout()

        # Preset 1
        row1 = QHBoxLayout()
        lbl_p1 = QLabel("Preset 1:")
        btn_save_p1 = QPushButton("Save")
        btn_load_p1 = QPushButton("Load")
        btn_save_p1.clicked.connect(lambda: self.save_preset(0))
        btn_load_p1.clicked.connect(lambda: self.load_preset(0))
        row1.addWidget(lbl_p1)
        row1.addWidget(btn_save_p1)
        row1.addWidget(btn_load_p1)
        presets_layout.addLayout(row1)

        # Preset 2
        row2 = QHBoxLayout()
        lbl_p2 = QLabel("Preset 2:")
        btn_save_p2 = QPushButton("Save")
        btn_load_p2 = QPushButton("Load")
        btn_save_p2.clicked.connect(lambda: self.save_preset(1))
        btn_load_p2.clicked.connect(lambda: self.load_preset(1))
        row2.addWidget(lbl_p2)
        row2.addWidget(btn_save_p2)
        row2.addWidget(btn_load_p2)
        presets_layout.addLayout(row2)

        # Preset 3
        row3 = QHBoxLayout()
        lbl_p3 = QLabel("Preset 3:")
        btn_save_p3 = QPushButton("Save")
        btn_load_p3 = QPushButton("Load")
        btn_save_p3.clicked.connect(lambda: self.save_preset(2))
        btn_load_p3.clicked.connect(lambda: self.load_preset(2))
        row3.addWidget(lbl_p3)
        row3.addWidget(btn_save_p3)
        row3.addWidget(btn_load_p3)
        presets_layout.addLayout(row3)

        # Preset 4
        row4 = QHBoxLayout()
        lbl_p4 = QLabel("Preset 4:")
        btn_save_p4 = QPushButton("Save")
        btn_load_p4 = QPushButton("Load")
        btn_save_p4.clicked.connect(lambda: self.save_preset(3))
        btn_load_p4.clicked.connect(lambda: self.load_preset(3))
        row4.addWidget(lbl_p4)
        row4.addWidget(btn_save_p4)
        row4.addWidget(btn_load_p4)
        presets_layout.addLayout(row4)

        presets_box.setLayout(presets_layout)
        v_layout_sl.addWidget(presets_box)

        # === 14) Spacer, para que el resto suba. ===
        v_layout_sl.addSpacerItem(QSpacerItem(20,20,QSizePolicy.Minimum,QSizePolicy.Expanding))

        # === 15) Finalmente añadimos la pestaña al QTabWidget ===
        right_tab_widget.addTab(tab_sliders, "Adjust Stereo")
        
        
        # Tab 4 => Stereo Generation
        
        tab_batch = QWidget()
        v_layout_bt = QVBoxLayout(tab_batch)
         


        # === Batch Splat Source ===
        row_bs = QHBoxLayout()
        lbl_bs2 = QLabel("Splat Src:")
        self.combo_batch_splat_source = QComboBox()
        self.combo_batch_splat_source.addItems(["depth1","depth2","Merged"])
        row_bs.addWidget(lbl_bs2)
        row_bs.addWidget(self.combo_batch_splat_source)
        v_layout_bt.addLayout(row_bs)
        
        self.check_overwrite_splat = QCheckBox("Overwrite existing splat (Don't ask)")
        self.check_overwrite_splat.setChecked(False)
        v_layout_bt.addWidget(self.check_overwrite_splat)

        btn_batch_splat = QPushButton("Generate Splat 2x2 + Anaglyph ")
        btn_batch_splat.setFixedHeight(28)
        btn_batch_splat.clicked.connect(self.batch_splat_selected)
        v_layout_bt.addWidget(btn_batch_splat)

        # === Widgets para inpainting batch (Threshold, Overlap, Tile, Steps) ===
        inpaint_threshold_batch_label = QLabel("Inpaint Threshold Mask:")
        self.spin_inpaint_threshold_batch = QDoubleSpinBox()
        self.spin_inpaint_threshold_batch.setRange(0.0, 1.0)
        self.spin_inpaint_threshold_batch.setDecimals(3)
        self.spin_inpaint_threshold_batch.setSingleStep(0.001)
        self.spin_inpaint_threshold_batch.setValue(self.inpaint_threshold_batch)
        self.spin_inpaint_threshold_batch.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.spin_inpaint_threshold_batch.valueChanged.connect(self.on_inpaint_threshold_batch_changed)

        inpaint_overlap_batch_label = QLabel("Inpaint Overlap:")
        self.spin_inpaint_overlap_batch = QSpinBox()
        self.spin_inpaint_overlap_batch.setRange(0, 100)
        self.spin_inpaint_overlap_batch.setValue(self.inpaint_overlap_batch)
        self.spin_inpaint_overlap_batch.valueChanged.connect(
            lambda val: setattr(self, 'inpaint_overlap_batch', val)
        )

        inpaint_tile_num_batch_label = QLabel("Inpaint Tile Num:")
        self.spin_inpaint_tile_batch = QSpinBox()
        self.spin_inpaint_tile_batch.setRange(1, 10)
        self.spin_inpaint_tile_batch.setValue(self.inpaint_tile_num_batch)
        self.spin_inpaint_tile_batch.valueChanged.connect(
            lambda val: setattr(self, 'inpaint_tile_num_batch', val)
        )

        steps_label_batch = QLabel("Num Inference Steps:")
        self.spin_inpaint_steps_batch = QSpinBox()
        self.spin_inpaint_steps_batch.setRange(1, 50)
        self.spin_inpaint_steps_batch.setValue(self.inpaint_num_inference_steps_batch)
        self.spin_inpaint_steps_batch.valueChanged.connect(self.on_inpaint_steps_batch_changed)

        # === Creamos un QGroupBox para agruparlos: "StereoCrafter Params" ===
        sc_box_batch = QGroupBox("StereoCrafter Params")
        sc_layout_batch = QVBoxLayout(sc_box_batch)

        # Añadimos los controles al layout del groupbox
        sc_layout_batch.addWidget(inpaint_threshold_batch_label)
        sc_layout_batch.addWidget(self.spin_inpaint_threshold_batch)

        sc_layout_batch.addWidget(inpaint_overlap_batch_label)
        sc_layout_batch.addWidget(self.spin_inpaint_overlap_batch)

        sc_layout_batch.addWidget(inpaint_tile_num_batch_label)
        sc_layout_batch.addWidget(self.spin_inpaint_tile_batch)

        sc_layout_batch.addWidget(steps_label_batch)
        sc_layout_batch.addWidget(self.spin_inpaint_steps_batch)

        sc_box_batch.setLayout(sc_layout_batch)


        v_layout_bt.addWidget(sc_box_batch)
        
        self.check_overwrite_inpaint = QCheckBox("Overwrite existing inpaint (Don't ask)")
        self.check_overwrite_inpaint.setChecked(False)
        v_layout_bt.addWidget(self.check_overwrite_inpaint)

        # === Botón Batch Inpaint ===
        btn_batch_inpaint = QPushButton("Generate SBS Inpaint (Splat needed)")
        btn_batch_inpaint.setFixedHeight(28)
        btn_batch_inpaint.clicked.connect(self.batch_inpaint_selected)
        v_layout_bt.addWidget(btn_batch_inpaint)
                         

        self.btn_direct_stereo = QPushButton("Generate Direct Inpainted Stereo (All Steps)")
        self.btn_direct_stereo.setFixedHeight(50)  
        self.btn_direct_stereo.clicked.connect(self.batch_direct_stereo)
        v_layout_bt.addWidget(self.btn_direct_stereo)            
                    
        v_layout_bt.addSpacing(5)
             
        self.btn_polylines_sbs = QPushButton("Generate Stereo SBS Polylines (experimental)")
        self.btn_polylines_sbs.setFixedHeight(28)
        self.btn_polylines_sbs.clicked.connect(self.do_polylines_sbs)
        v_layout_bt.addWidget(self.btn_polylines_sbs)
        self.btn_polylines_sbs.setStyleSheet("background-color: #5DADE2; color: white;")

        v_layout_bt.addSpacing(20)

        # === Merge Source ===
        row_merge_combo = QHBoxLayout()
        lbl_merge_source = QLabel("Merge Source:")
        self.combo_merge_source = QComboBox()
        self.combo_merge_source.addItems(["ORIG","DEPTH1","DEPTH2","MERGED_DEPTH","SPLATTED","INPAINTED"])
        row_merge_combo.addWidget(lbl_merge_source)
        row_merge_combo.addWidget(self.combo_merge_source)
        v_layout_bt.addLayout(row_merge_combo)

        btn_merge_videos = QPushButton("Merge Selected Videos")
        btn_merge_videos.setFixedHeight(28)
        btn_merge_videos.clicked.connect(self.merge_selected_videos)
        v_layout_bt.addWidget(btn_merge_videos)      
       
   
        v_layout_bt.addSpacerItem(QSpacerItem(20,20,QSizePolicy.Minimum,QSizePolicy.Expanding))
        
  
      


        right_tab_widget.addTab(tab_batch, "Generate Stereo")

        # ------------------------------
        # Tab Output (no lo modificamos)
        tab_output = QWidget()
        v_layout_out = QVBoxLayout(tab_output)

        # SBS mode
        row_sbs = QHBoxLayout()
        lbl_sbs = QLabel("SBS Mode:")
        self.combo_sbs = QComboBox()
        self.combo_sbs.addItems(["HSBS", "FSBS"])
        self.combo_sbs.setCurrentText(self.sbs_mode)
        self.combo_sbs.currentIndexChanged.connect(self.on_sbs_mode_changed)
        row_sbs.addWidget(lbl_sbs)
        row_sbs.addWidget(self.combo_sbs)
        v_layout_out.addLayout(row_sbs)

        # Encoder
        row_enc = QHBoxLayout()
        lbl_enc = QLabel("Encoder:")
        self.combo_encoder = QComboBox()
        self.combo_encoder.addItems(["x264", "x265"])
        self.combo_encoder.setCurrentText(self.output_encoder)
        self.combo_encoder.currentIndexChanged.connect(self.on_encoder_changed)
        row_enc.addWidget(lbl_enc)
        row_enc.addWidget(self.combo_encoder)
        v_layout_out.addLayout(row_enc)

        # Expandir resto del espacio
        v_layout_out.addSpacerItem(QSpacerItem(20,20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        tab_output.setLayout(v_layout_out)

        right_tab_widget.addTab(tab_output, "Config Output")
        right_side_layout = QVBoxLayout()
        right_side_layout.addWidget(right_tab_widget)
        
        self.kofi_button = QPushButton()  
        self.kofi_button.setCursor(Qt.PointingHandCursor)  

        class ClickableLabel(QLabel):
            clicked = pyqtSignal()
            def mousePressEvent(self, event):
                if event.button() == Qt.LeftButton:
                    self.clicked.emit()

        # b) Cargamos la imagen si existe, o ponemos texto si no
        self.kofi_label = ClickableLabel()
        self.kofi_label.setAlignment(Qt.AlignCenter)
        self.kofi_label.setCursor(Qt.PointingHandCursor)
        if os.path.isfile("assets/support_me.png"):
            pix = QPixmap("assets/support_me.png")
            self.kofi_label.setPixmap(pix.scaled(200, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.kofi_label.setText("¡Support me on Ko-fi!")
        self.kofi_label.clicked.connect(self.open_kofi_link)

        right_side_layout.addWidget(self.kofi_label)

 
        main_layout.addLayout(right_side_layout)



    ###########################################################
    # == KEYFRAMES: Add / Remove / Clear / Jump ==
    ###########################################################
    
    
    def save_preset(self, preset_index):
        """
        Lee los valores actuales de los sliders y los guarda en self.presets[preset_index].
        """
        # Creamos un diccionario con los valores de todos los sliders:
        preset_data = {
            "disp_value": self.disp_value,
            "conv_value": self.conv_value,
            "brightness_value": self.brightness_value,
            "gamma_value": self.gamma_value,
            "dilate_h_value": self.dilate_h_value,
            "dilate_v_value": self.dilate_v_value,
            "blur_ksize_value": self.blur_ksize_value,
            "blur_sigma_value": self.blur_sigma_value
        }
        self.presets[preset_index] = preset_data
        self.log(f"[PRESET] => Saved preset #{preset_index+1} => {preset_data}")

 
    def load_preset(self, preset_index):
        """
        Toma los valores de self.presets[preset_index] y los aplica a los sliders.
        """
        preset_data = self.presets[preset_index]
        if preset_data is None:
            QMessageBox.information(self, "Info", f"No data in preset #{preset_index+1} yet.")
            return
        
        # Aplicamos los valores a nuestras variables y sliders:
        self.disp_value = preset_data["disp_value"]
        self.slider_disp.setValue(int(self.disp_value))
        self.lbl_disp_val.setText(str(self.disp_value))

        self.conv_value = preset_data["conv_value"]
        self.slider_conv.setValue(int(self.conv_value))
        self.lbl_conv_val.setText(str(self.conv_value))

        self.brightness_value = preset_data["brightness_value"]
        self.slider_brightness.setValue(int(self.brightness_value * 100))
        self.lbl_bri_val.setText(f"{self.brightness_value:.2f}")

        self.gamma_value = preset_data["gamma_value"]
        self.slider_gamma.setValue(int(self.gamma_value * 100))
        self.lbl_gam_val.setText(f"{self.gamma_value:.2f}")

        self.dilate_h_value = preset_data["dilate_h_value"]
        self.slider_dilate_h.setValue(self.dilate_h_value)
        self.lbl_dh_val.setText(str(self.dilate_h_value))

        self.dilate_v_value = preset_data["dilate_v_value"]
        self.slider_dilate_v.setValue(self.dilate_v_value)
        self.lbl_dv_val.setText(str(self.dilate_v_value))

        bk_ = preset_data["blur_ksize_value"]
        if bk_ > 0 and bk_ % 2 == 0:
            bk_ += 1
        self.blur_ksize_value = bk_
        self.slider_blur_ksize.setValue(bk_)
        self.lbl_bk_val.setText(str(bk_))

        self.blur_sigma_value = preset_data["blur_sigma_value"]
        self.slider_blur_sigma.setValue(int(self.blur_sigma_value * 10))
        self.lbl_bs_val.setText(f"{self.blur_sigma_value:.2f}")

        # (Opcional) Actualizar la vista previa
        self.log(f"[PRESET] => Loaded preset #{preset_index+1} => {preset_data}")
        self.show_preview(self.current_frame_idx)
        self.store_current_params()

        self.log(f"[PRESET] => Loaded preset #{preset_index+1} => {preset_data}")

        
    def on_delete_selected_clicked(self):
        if self.current_batch_worker is not None:
            QMessageBox.warning(
                self, "Process Ongoing",
                "A batch/process is running. Please cancel or wait until it finishes "
                "before deleting."
            )
            return

        sel_items = self.list_widget.selectedItems()
        if not sel_items:
            QMessageBox.information(self, "Info", "No videos selected to delete.")
            return

        # 1) Gather all 'base_name' from selected items
        base_names_to_delete = set()
        for item in sel_items:
            dd = item.data(Qt.UserRole)
            if not dd:
                continue
            base_ = dd.get("base_name", "")
            if base_:
                base_names_to_delete.add(base_)

        if not base_names_to_delete:
            QMessageBox.information(self, "Info", "No valid 'base_name' found in the selected items.")
            return

        # Helper to shorten a potentially long list
        def shorten_list(items, max_items=10):
            items_list = list(items)
            total = len(items_list)
            if total <= max_items:
                return "\n".join(f" - {x}" for x in items_list)
            else:
                shown = items_list[:max_items]
                remaining = total - max_items
                return "\n".join(f" - {x}" for x in shown) + f"\n  ... plus {remaining} more"

        # 2) Confirmation dialog (shortened list)
        summary_len = len(base_names_to_delete)
        items_str = shorten_list(base_names_to_delete, max_items=10)

        msg_text = (
            "You are about to delete all related files for:\n\n"
            + items_str + "\n\n"
            "This includes original, depth, splatted, inpainted if present.\n\n"
            "Are you sure?"
        )

        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            msg_text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if confirm == QMessageBox.No:
            return

        # 3) Force-close *all* open decord references before deleting anything
        #    and clear the preview label so Windows won't keep them locked.
        self.vid_original = None
        self.vid_depth = None
        self.vid_depth2 = None
        self.vid_depth_merged = None
        self.vid_splatted = None
        self.vid_inpainted = None
        self.label_preview.clear()
        import gc
        gc.collect()

        # 4) Build a list of all physical files to delete
        files_to_delete = []
        for (b_name, st_list, detail, ddict) in self.all_items_data:
            if b_name in base_names_to_delete:
                # Gather possible video files
                for key_ in ["orig", "depth1", "depth2", "depthf", "splatted", "inpainted"]:
                    path_ = ddict.get(key_, "")
                    if path_ and os.path.isfile(path_):
                        files_to_delete.append(path_)

                        # If it's splatted, also remove the optional _left/_mask if they exist
                        if key_ == "splatted":
                            dirname = os.path.dirname(path_)
                            fname = os.path.basename(path_)
                            name_noext, ext = os.path.splitext(fname)
                            left_candidate = os.path.join(dirname, name_noext + "_left" + ext)
                            mask_candidate = os.path.join(dirname, name_noext + "_mask" + ext)
                            if os.path.isfile(left_candidate):
                                files_to_delete.append(left_candidate)
                            if os.path.isfile(mask_candidate):
                                files_to_delete.append(mask_candidate)

        # 5) Delete each file
        for fpath in set(files_to_delete):
            try:
                os.remove(fpath)
                self.log(f"[DELETE] => removed file => {fpath}")
            except Exception as e:
                self.log(f"[DELETE] => cannot remove {fpath} => {e}")

        # 6) Remove the entries from self.all_items_data
        new_list_data = []
        for entry in self.all_items_data:
            (b_name, st_list, detail, ddict) = entry
            if b_name not in base_names_to_delete:
                new_list_data.append(entry)
            else:
                self.log(f"[DELETE] => Removing data for '{b_name}' from list.")
        self.all_items_data = new_list_data

        # 7) Refresh the list visually
        self.apply_filter_list()

        # 8) Show a short final message as well
        deleted_items_str = shorten_list(base_names_to_delete, max_items=10)
        QMessageBox.information(
            self,
            "Deleted",
            "All files for:\n"
            + deleted_items_str +
            "\n\nhave been deleted."
        )

            
    def add_keyframe(self):
        item = self.list_widget.currentItem()
        if not item:
            QMessageBox.warning(self, "Attention", "No video selected in the list.")
            return
        dd = item.data(Qt.UserRole)
        if not dd:
            return
        base_ = dd.get("base_name","")
        if not base_:
            return
        if base_ not in self.video_params:
            self.video_params[base_] = {}
        if "keyframes" not in self.video_params[base_]:
            self.video_params[base_]["keyframes"] = {}

        current_frame = self.current_frame_idx
        # Store all slider-based param info you want to keep:
        kf_data = {
            "disp_value": self.disp_value,
            "convergence": self.conv_value,
            "brightness_value": self.brightness_value,
            "gamma_value": self.gamma_value,
            "dilate_h_value": self.dilate_h_value,
            "dilate_v_value": self.dilate_v_value,
            "blur_ksize_value": self.blur_ksize_value,
            "blur_sigma_value": self.blur_sigma_value,
            "stereo_offset_exponent_value": self.poly_exponent,
            "fill_technique_value": self.poly_fill_tech,
            "warp_exponent_base": self.warp_exponent_base
        }
        self.video_params[base_]["keyframes"][str(current_frame)] = kf_data
        self.save_params()

        self.log(f"[KEYFRAME] => Added keyframe at frame {current_frame} => {kf_data}")
        self.update_keyframes_ui()

    def remove_selected_keyframe(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        dd = item.data(Qt.UserRole)
        if not dd:
            return
        base_ = dd.get("base_name","")
        if not base_:
            return
        if base_ not in self.video_params:
            return
        if "keyframes" not in self.video_params[base_]:
            return
        kf_dict = self.video_params[base_]["keyframes"]

        sel_kf_str = self.keyframes_combo.currentText()
        if not sel_kf_str:
            return
        try:
            fr_int = int(sel_kf_str)
        except:
            return
        if str(fr_int) in kf_dict:
            del kf_dict[str(fr_int)]
            self.save_params()
            self.log(f"[KEYFRAME] => Removed keyframe at frame {fr_int}")
            self.update_keyframes_ui()

    def clear_all_keyframes(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        dd = item.data(Qt.UserRole)
        if not dd:
            return
        base_ = dd.get("base_name","")
        if not base_:
            return
        if base_ not in self.video_params:
            return
        if "keyframes" in self.video_params[base_]:
            self.video_params[base_]["keyframes"] = {}
            self.save_params()
            self.log(f"[KEYFRAME] => Cleared all keyframes for '{base_}'")
            self.update_keyframes_ui()

    def on_keyframe_combo_selected(self, idx):
        kf_str = self.keyframes_combo.currentText()
        if not kf_str:
            return
        self.log(f"[KEYFRAME] => Selected keyframe: frame={kf_str}")
        try:
            fr_int = int(kf_str)
            self.slider_frames.setValue(fr_int)
        except:
            pass

    def update_keyframes_ui(self, block_signals=False):

        if block_signals:
            self.keyframes_combo.blockSignals(True)

        self.keyframes_combo.clear()
        item = self.list_widget.currentItem()
        if not item:
            # No item => clear
            self.slider_frames.set_keyframes([])
            self.slider_frames.set_keyframes_dict({})
            if block_signals:
                self.keyframes_combo.blockSignals(False)
            return

        dd = item.data(Qt.UserRole)
        if not dd:
            self.slider_frames.set_keyframes([])
            self.slider_frames.set_keyframes_dict({})
            if block_signals:
                self.keyframes_combo.blockSignals(False)
            return

        base_ = dd.get("base_name","")
        if not base_:
            self.slider_frames.set_keyframes([])
            self.slider_frames.set_keyframes_dict({})
            if block_signals:
                self.keyframes_combo.blockSignals(False)
            return

        if base_ not in self.video_params:
            self.slider_frames.set_keyframes([])
            self.slider_frames.set_keyframes_dict({})
            if block_signals:
                self.keyframes_combo.blockSignals(False)
            return

        if "keyframes" not in self.video_params[base_]:
            self.slider_frames.set_keyframes([])
            self.slider_frames.set_keyframes_dict({})
            if block_signals:
                self.keyframes_combo.blockSignals(False)
            return

        kf_dict = self.video_params[base_]["keyframes"]
        # Ordenar frames como int
        frames_list = sorted([int(f) for f in kf_dict.keys()])
        for fr_ in frames_list:
            self.keyframes_combo.addItem(str(fr_))

        # Actualizamos la KeyframeSlider con las posiciones
        self.slider_frames.set_keyframes(frames_list)

        # Construir el diccionario frame->param_dict para los tooltips
        int2param = {}
        for fr_s, paramdict in kf_dict.items():
            try:
                fr_i = int(fr_s)
                int2param[fr_i] = paramdict
            except:
                pass
        self.slider_frames.set_keyframes_dict(int2param)

        if block_signals:
            self.keyframes_combo.blockSignals(False)


    def apply_keyframe_if_exists(self, frame_idx):
        item = self.list_widget.currentItem()
        if not item:
            return
        dd = item.data(Qt.UserRole)
        if not dd:
            return
        base_ = dd.get("base_name","")
        if not base_:
            return

        if base_ not in self.video_params:
            return
        if "keyframes" not in self.video_params[base_]:
            return

        kf_dict = self.video_params[base_]["keyframes"]
        if str(frame_idx) not in kf_dict:
            return

        kf_data = kf_dict[str(frame_idx)]
        self.disp_value = kf_data.get("disp_value", self.disp_value)
        self.slider_disp.setValue(int(self.disp_value))
        self.lbl_disp_val.setText(str(self.disp_value))

        self.conv_value = kf_data.get("convergence", self.conv_value)
        self.slider_conv.setValue(int(self.conv_value))
        self.lbl_conv_val.setText(str(self.conv_value))
        
        self.warp_exponent_base = kf_data.get("warp_exponent_base", self.warp_exponent_base)
        self.slider_warp_exponent.setValue(int(self.warp_exponent_base * 100))
        self.lbl_warp_exponent_val.setText(f"{self.warp_exponent_base:.2f}")

        self.brightness_value = kf_data.get("brightness_value", self.brightness_value)
        self.slider_brightness.setValue(int(self.brightness_value*100))
        self.lbl_bri_val.setText(f"{self.brightness_value:.2f}")

        self.gamma_value = kf_data.get("gamma_value", self.gamma_value)
        self.slider_gamma.setValue(int(self.gamma_value*100))
        self.lbl_gam_val.setText(f"{self.gamma_value:.2f}")

        self.dilate_h_value = kf_data.get("dilate_h_value", self.dilate_h_value)
        self.slider_dilate_h.setValue(self.dilate_h_value)
        self.lbl_dh_val.setText(str(self.dilate_h_value))

        self.dilate_v_value = kf_data.get("dilate_v_value", self.dilate_v_value)
        self.slider_dilate_v.setValue(self.dilate_v_value)
        self.lbl_dv_val.setText(str(self.dilate_v_value))

        bk_ = kf_data.get("blur_ksize_value", self.blur_ksize_value)
        if bk_>0 and bk_%2==0:
            bk_+=1
        self.blur_ksize_value = bk_
        self.slider_blur_ksize.setValue(bk_)
        self.lbl_bk_val.setText(str(bk_))

        self.blur_sigma_value = kf_data.get("blur_sigma_value", self.blur_sigma_value)
        self.slider_blur_sigma.setValue(int(self.blur_sigma_value*10))
        self.lbl_bs_val.setText(f"{self.blur_sigma_value:.2f}")

        self.log(f"[KEYFRAME] => Applied keyframe at frame {frame_idx}")

    ###########################################################
    # SCENE DETECT
    ###########################################################
    
    def do_scene_detect(self, threshold=None, video_path=None):
        """
        Lanza SceneDetect en un SubprocessWorker. 'threshold' y 'video_path'
        son opcionales; si no llegan, usamos la UI (self.combo_scene_threshold
        y self.selected_video_path).
        """
        import os
        import sys

        # 1) Determinar threshold
        if threshold is None:
            threshold_str = self.combo_scene_threshold.currentText().strip()
            if not threshold_str.isdigit():
                QMessageBox.warning(self, "Attention", "Threshold must be numeric.")
                return
            threshold = threshold_str
        else:
            threshold = str(threshold)

        # 2) Determinar la ruta del vídeo original
        if video_path is None:
            if not self.selected_video_path:
                QMessageBox.warning(self, "Attention", "No video selected for scene detect.")
                return
            video_path = self.selected_video_path

        if not os.path.isfile(video_path):
            QMessageBox.warning(self, "Attention", f"Video file not found:\n{video_path}")
            return

        # Convertir a ruta absoluta (para que no haya problemas con os.path)
        video_path_abs = os.path.abspath(video_path)


        # 3) Construir el comando para SceneDetect
        cmd = [
            sys.executable, "-m", "scenedetect",
            "-i", video_path_abs,
            "-o", self.dir_input_videos,  # Carpeta donde se guardan los trozos
            "detect-content",
            "-t", threshold,
            "split-video"
        ]
        self.log(f"[INFO] => scenedetect => {cmd}")

        # 4) Creamos el worker
        self.scene_worker = SubprocessWorker(cmd)
        self.scene_worker.lineReady.connect(self.log)

        # IMPORTANTE: Pasamos video_path_abs en la lambda para que
        # on_scene_detect_finished reciba la ruta como segundo param (original_path).
        self.scene_worker.finishedSignal.connect(
            lambda success: self.on_scene_detect_finished(success, video_path_abs)
        )

        self.current_batch_worker = self.scene_worker
        self.batch_cancelled = False
        self.scene_worker.start()


    def on_scene_detect_finished(self, success, original_path):
        self.current_batch_worker = None
        if success:
            self.log("[OK] => Scene detect done.")

            # Liberar decord para evitar bloqueos en Windows si estaba abierto en preview
            self.vid_original = None
            self.vid_depth = None
            self.vid_depth2 = None
            self.vid_depth_merged = None
            self.label_preview.clear()
            import gc
            gc.collect()

            # Preguntar si se desea borrar el original
            if original_path and os.path.isfile(original_path):
                resp = QMessageBox.question(
                    self,
                    "Remove Original?",
                    f"Scene splitting done. Remove original?\n\n{original_path}",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if resp == QMessageBox.Yes:
                    try:
                        os.remove(original_path)
                        self.log(f"[INFO] => removed original => {original_path}")
                    except Exception as e:
                        self.log(f"[ERROR] => Could not remove file: {e}")
            else:
                self.log(f"[WARN] => Original file not found or already removed: {original_path}")
        else:
            self.log("[ERROR] => scenedetect failed.")

        # Refrescar la lista tras el split
        self.scan_and_list_videos()



    ###########################################################
    # Overlap/Window => Crafter
    ###########################################################
    def on_crafter_overlap_changed(self, idx):
        val_s= self.combo_overlap.currentText()
        self.crafter_overlap= int(val_s)
        self.save_depth_global_params()
        self.log(f"[INFO] => Crafter overlap => {self.crafter_overlap}")

    def on_crafter_window_changed(self, idx):
        val_s= self.combo_window.currentText()
        self.crafter_window= int(val_s)
        self.save_depth_global_params()
        self.log(f"[INFO] => Crafter window => {self.crafter_window}")

    def on_crafter_denoising_changed(self, val):
        self.crafter_denoising_steps = val
        self.save_depth_global_params()
        self.log(f"[INFO] => Crafter denoising steps => {val}")

    def on_crafter_guidance_changed(self, val):
        guidance = round(val, 1)
        self.crafter_guidance_scale = guidance  # <--- actualiza la variable global
        self.log(f"Guidance Scale: {guidance}")           
        self.save_depth_global_params()      
        self.log(f"[INFO] => Crafter guidance scale => {guidance}")  

    def on_crafter_seed_changed(self, val):
        self.crafter_seed = val
        self.save_depth_global_params()
        self.log(f"[INFO] => Crafter seed => {val}")

    ###########################################################
    # GENERATE DEPTH2
    ###########################################################
    def on_vda_res_changed(self, index):
        val_str = self.combo_vda_res.currentText()
        val_int = int(val_str)
        self.vda_maxres_value = val_int  # <--- ya NO tocas self.depth_maxres_value
        self.save_depth_global_params()
        self.log(f"[INFO] => VDA max_res => {val_int}")
        
    def on_vda_input_size_changed(self, index):
        val_str = self.combo_vda_input_size.currentText()
        val_int = int(val_str)
        self.vda_input_size = val_int
        self.save_depth_global_params()
        self.log(f"[INFO] => VDA input_size => {val_int}")

    def on_vda_encoder_changed(self, index):
        val_str = self.combo_vda_enc.currentText()
        self.vda_encoder = val_str
        self.save_depth_global_params()
        self.log(f"[INFO] => VDA encoder => {val_str}")


    ###########################################################
    # MERGE => Depth1 & Depth2 => DepthMerged
    ###########################################################
    def merge_depths(self):
        if not self.selected_depth_path or not self.selected_depth_path2:
            self.log("[WARN] => can't merge => missing depth1 or depth2.")
            return
        base_= os.path.splitext(os.path.basename(self.selected_video_path))[0]
        out_= os.path.join(self.pre_depth_final_edit.text(), base_+"_depth.mp4")
        cmd= [
            sys.executable,"fusion_depths.py",
            "--depth1", self.selected_depth_path,
            "--depth2", self.selected_depth_path2,
            "--alpha", str(self.fusion_alpha/100.0),
            "--output", out_,
            "--brightness_value", str(self.brightness_value),
            "--gamma_value", str(self.gamma_value),
            "--dilate_h", str(self.dilate_h_value),
            "--dilate_v", str(self.dilate_v_value),
            "--blur_ksize", str(self.blur_ksize_value),
            "--blur_sigma", str(self.blur_sigma_value)
        ]
        self.log("[INFO] => merge => "+ " ".join(cmd))
        self.merge_worker= SubprocessWorker(cmd)
        self.merge_worker.lineReady.connect(self.log)
        self.merge_worker.finishedSignal.connect(self.on_merged_depths)
        self.current_batch_worker=self.merge_worker
        self.batch_cancelled = False
        self.merge_worker.start()

    def on_merged_depths(self, success):
        self.current_batch_worker=None
        if success:
            self.log("[OK] => merged => refresh")
            self.scan_and_list_videos()
        else:
            self.log("[ERROR] => merge depth fail")



    def apply_filter_list(self):
        self.list_widget.clear()
        sel_filter = self.filter_combo.currentText()

        for (base_, st_list, detail, ddict) in self.all_items_data:
            # Cuando NO hay vídeo ni depth ni nada => st_list == ["NO VIDEO?"]
            if st_list == ["NO VIDEO?"]:
                # Solo se muestra si el filtro es "ALL"
                if sel_filter != "ALL":
                    continue
                st_str = "NO VIDEO?"
            else:
                # st_list puede ser ["ORIG","DEPTH"] => "ORIG + DEPTH"
                st_str = " + ".join(st_list)

            # Si no es "ALL", filtramos por coincidencia exacta con st_str
            if sel_filter != "ALL" and sel_filter != st_str:
                continue

            # Construimos el texto a mostrar en la lista
            txt = f"[{st_str}] {base_}"
            if detail:
                txt += " (" + ", ".join(detail) + ")"

            lw = QListWidgetItem(txt)
            lw.setData(Qt.UserRole, ddict)
            self.list_widget.addItem(lw)

    def on_item_selected(self, item):
        dd = item.data(Qt.UserRole)
        if not dd:
            return

        # Limpiamos referencias previas y el label
        self.vid_original = None
        self.vid_depth = None
        self.vid_depth2 = None
        self.vid_depth_merged = None
        self.vid_splatted = None
        self.vid_inpainted = None

        self.num_frames = 0
        self.current_frame_idx = 0
        self.label_preview.clear()

        base_ = dd.get("base_name", "")
        self.selected_video_path = dd.get("orig", "")
        self.selected_depth_path = dd.get("depth1", "")
        self.selected_depth_path2 = dd.get("depth2", "")
        self.selected_depth_merged = dd.get("depthf", "")
        self.splatted_video_path = dd.get("splatted", "")
        self.inpainted_video_path = dd.get("inpainted", "")

        # 1) Cargar ORIGINAL (si existe)
        if self.selected_video_path and os.path.isfile(self.selected_video_path):
            try:
                vr = VideoReader(self.selected_video_path, ctx=cpu(0))
                self.vid_original = vr
            except Exception as e:
                self.log("[ERROR] => load original => " + str(e))
                self.vid_original = None

        # 2) Cargar DEPTH1
        if self.selected_depth_path and os.path.isfile(self.selected_depth_path):
            try:
                vr = VideoReader(self.selected_depth_path, ctx=cpu(0))
                self.vid_depth = vr
            except Exception as e:
                self.log("[ERROR] => load depth1 => " + str(e))
                self.vid_depth = None

        # 3) Cargar DEPTH2
        if self.selected_depth_path2 and os.path.isfile(self.selected_depth_path2):
            try:
                vr = VideoReader(self.selected_depth_path2, ctx=cpu(0))
                self.vid_depth2 = vr
            except Exception as e:
                self.log("[ERROR] => load depth2 => " + str(e))
                self.vid_depth2 = None

        # 4) Cargar DEPTH MERGED
        if self.selected_depth_merged and os.path.isfile(self.selected_depth_merged):
            try:
                vr = VideoReader(self.selected_depth_merged, ctx=cpu(0))
                self.vid_depth_merged = vr
            except Exception as e:
                self.log("[ERROR] => load depthF => " + str(e))
                self.vid_depth_merged = None
        
        # 5) Cargar SPLATTED
        if self.splatted_video_path and os.path.isfile(self.splatted_video_path):
            try:
                vr = VideoReader(self.splatted_video_path, ctx=cpu(0))
                self.vid_splatted = vr
                self.log(f"[INFO] => Splatted loaded => {self.splatted_video_path}")
            except Exception as e:
                self.log("[ERROR] => load splatted => " + str(e))
                self.vid_splatted = None

        # 6) Cargar INPAINTED
        if self.inpainted_video_path and os.path.isfile(self.inpainted_video_path):
            try:
                vr = VideoReader(self.inpainted_video_path, ctx=cpu(0))
                self.vid_inpainted = vr
                self.log(f"[INFO] => Inpainted loaded => {self.inpainted_video_path}")
            except Exception as e:
                self.log("[ERROR] => load inpainted => " + str(e))
                self.vid_inpainted = None

        # 7) Determinar num_frames = máx de TODOS los videos cargados
        potential_counts = []
        if self.vid_original:
            potential_counts.append(len(self.vid_original))
        if self.vid_depth:
            potential_counts.append(len(self.vid_depth))
        if self.vid_depth2:
            potential_counts.append(len(self.vid_depth2))
        if self.vid_depth_merged:
            potential_counts.append(len(self.vid_depth_merged))
        if self.vid_splatted:
            potential_counts.append(len(self.vid_splatted))
        if self.vid_inpainted:
            potential_counts.append(len(self.vid_inpainted))

        if potential_counts:
            self.num_frames = max(potential_counts)
            self.slider_frames.setMaximum(self.num_frames - 1)
            self.slider_frames.setValue(0)
            self.current_frame_idx = 0
        else:
            # No hay ningún video cargado
            self.num_frames = 0
            self.slider_frames.setMaximum(0)
            self.slider_frames.setValue(0)
            self.log("[WARN] => No video available to preview.")

        # (A) => Si el original existe y es muy largo (>650 frames), mostrar advertencia
        if self.vid_original and len(self.vid_original) > 650:
            import PyQt5.QtWidgets as QtW
            msg = (
                f"The selected video has {len(self.vid_original)} frames.\n\n"
                "StereoMaster is NOT optimized for long videos.\n"
                "It is strongly recommended to SPLIT it using Scene Detect.\n\n"
                "Do you want to run scene detect now?\n"
                " - YES => ask threshold + do scene detect\n"
                " - NO => proceed anyway (AT YOUR OWN RISK)\n"
                " - CANCEL => abort loading this video"
            )
            resp = QtW.QMessageBox.warning(
                self,
                "Long Video Detected",
                msg,
                QtW.QMessageBox.Yes | QtW.QMessageBox.No | QtW.QMessageBox.Cancel,
                QtW.QMessageBox.Yes
            )
            if resp == QtW.QMessageBox.Yes:
                # Pedir umbral
                from PyQt5.QtWidgets import QInputDialog
                threshold, ok = QInputDialog.getInt(
                    self, "Scene Threshold",
                    "Enter scene detection threshold [15..55 recommended]:",
                    35, 1, 999, 1
                )
                if ok:
                    self.do_scene_detect_with_threshold(threshold)
                else:
                    self.log("[INFO] => Scene detect threshold cancelled by user.")
            elif resp == QtW.QMessageBox.No:
                self.log("[WARNING] => Proceeding with a long video at user's own risk.")
            else:
                self.log("[INFO] => User cancelled loading this long video.")
                return

        # 8) Restaurar sliders (per-video) => si base_ está en video_params
        if base_ in self.video_params:
            vp = self.video_params[base_]
            self.disp_value = vp.get("disp_value", 20.0)
            self.conv_value = vp.get("conv_value", 0.0)
            self.brightness_value = vp.get("brightness_value", 1.0)
            self.gamma_value = vp.get("gamma_value", 1.0)
            self.dilate_h_value = vp.get("dilate_h_value", 4)
            self.dilate_v_value = vp.get("dilate_v_value", 1)
            self.blur_ksize_value = vp.get("blur_ksize_value", 3)
            self.blur_sigma_value = vp.get("blur_sigma_value", 2.0)
            self.warp_exponent_base = vp.get("warp_exponent_base", 1.414)

            self.slider_disp.setValue(int(self.disp_value))
            self.lbl_disp_val.setText(str(self.disp_value))
            self.slider_conv.setValue(int(self.conv_value))
            self.lbl_conv_val.setText(str(self.conv_value))

            self.slider_brightness.setValue(int(self.brightness_value * 100))
            self.lbl_bri_val.setText(str(self.brightness_value))

            self.slider_gamma.setValue(int(self.gamma_value * 100))
            self.lbl_gam_val.setText(str(self.gamma_value))

            self.slider_dilate_h.setValue(self.dilate_h_value)
            self.lbl_dh_val.setText(str(self.dilate_h_value))

            self.slider_dilate_v.setValue(self.dilate_v_value)
            self.lbl_dv_val.setText(str(self.dilate_v_value))

            bk_ = self.blur_ksize_value
            if bk_ > 0 and bk_ % 2 == 0:
                bk_ += 1
            self.blur_ksize_value = bk_
            self.slider_blur_ksize.setValue(bk_)
            self.lbl_bk_val.setText(str(bk_))

            self.slider_blur_sigma.setValue(int(self.blur_sigma_value * 10))
            self.lbl_bs_val.setText(str(self.blur_sigma_value))

        # 9) Actualizar combo de keyframes, etc.
        current_fr = self.current_frame_idx
        self.update_keyframes_ui(block_signals=True)
        self.slider_frames.setValue(current_fr)

        # 10) Mostrar preview primer frame
        if self.num_frames > 0:
            self.show_preview(0)


    def on_create_project_clicked(self):
        # 1) Pedir el nombre del proyecto
        from PyQt5.QtWidgets import QInputDialog, QMessageBox
        project_name, ok = QInputDialog.getText(
            self,
            "New Project",
            "Enter project name:"
        )
        if not ok or not project_name.strip():
            # Usuario canceló o dejó vacío
            return

        project_name = project_name.strip()

        # 2) Definir la carpeta raíz “projects” (al lado de tu script, por ej.)
        projects_root = os.path.abspath("projects")
        if not os.path.isdir(projects_root):
            os.makedirs(projects_root, exist_ok=True)

        # 3) Crear la carpeta del proyecto
        project_folder = os.path.join(projects_root, project_name)
        if os.path.exists(project_folder):
            resp = QMessageBox.question(
                self,
                "Project already exists",
                f"The folder '{project_folder}' already exists. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if resp == QMessageBox.No:
                return
        else:
            os.makedirs(project_folder, exist_ok=True)

        # 4) Crear subcarpetas => input_videos, pre_depth, ...
        subfolders = [
            "input_videos", "pre_depth", "pre_depth2",
            "pre_depth_final", "output_splattings",
            "inpainting_out", "completed_merged"
        ]
        for sf in subfolders:
            path_ = os.path.join(project_folder, sf)
            os.makedirs(path_, exist_ok=True)

        # 5) Actualizamos la interfaz: 
        #    - Rellenar de nuevo la combo con todos los proyectos detectados
        self.scan_projects()

        # 6) Seleccionar en la combo el que acabamos de crear (si existe):
        idx_new = self.combo_projects.findText(project_name)
        if idx_new >= 0:
            self.combo_projects.setCurrentIndex(idx_new)
        else:
            # Si no está, algo raro pasó, pero no pasa nada
            pass

        QMessageBox.information(
            self,
            "Project Created",
            f"Project '{project_name}' created at:\n{project_folder}"
        )


    def clear_current_loaded_video(self):
        """
        Limpia cualquier referencia a decord u otras variables de vídeo,
        así como el label de previsualización y la lista de videos.
        """
        # 1) Liberar referencias de decord
        self.vid_original = None
        self.vid_depth = None
        self.vid_depth2 = None
        self.vid_depth_merged = None
        self.vid_splatted = None
        self.vid_inpainted = None

        # 2) Limpiar label preview y forzar re-dibujo
        self.label_preview.clear()
        self.label_preview.update()

        # 3) Restablecer contadores de frames
        self.num_frames = 0
        self.current_frame_idx = 0

        # 4) Vaciar la lista de items (para que no queden vídeos del proyecto anterior)
        self.list_widget.clear()

        # 5) (Opcional) forzar garbage collection, útil en Windows
        import gc
        gc.collect()

    def on_project_selected(self, index):
        if not self.combo_projects.isEnabled():
            return

        project_name = self.combo_projects.currentText()
        if not project_name or project_name.startswith("<No Projects"):
            return

        # --- CASO ESPECIAL: limpiar vídeos abiertos ANTES de cambiar a las nuevas rutas
        self.clear_current_loaded_video()

        # A partir de aquí, tus rutas cambian a las del proyecto elegido:
        projects_root = os.path.abspath("projects")
        project_folder = os.path.join(projects_root, project_name)

        # Asignas subcarpetas
        self.dir_input_videos = os.path.join(project_folder, "input_videos")
        self.dir_pre_depth = os.path.join(project_folder, "pre_depth")
        self.dir_pre_depth2 = os.path.join(project_folder, "pre_depth2")
        self.dir_pre_depth_final = os.path.join(project_folder, "pre_depth_final")
        self.dir_output_splattings = os.path.join(project_folder, "output_splattings")
        self.dir_inpainting_out = os.path.join(project_folder, "inpainting_out")
        self.dir_completed_merged = os.path.join(project_folder, "completed_merged")

        # Reflejar en los QLineEdit
        self.input_videos_edit.setText(self.dir_input_videos)
        self.pre_depth_edit.setText(self.dir_pre_depth)
        self.pre_depth2_edit.setText(self.dir_pre_depth2)
        self.pre_depth_final_edit.setText(self.dir_pre_depth_final)
        self.output_splattings_edit.setText(self.dir_output_splattings)
        self.inpainting_out_edit.setText(self.dir_inpainting_out)
        self.completed_merged_edit.setText(self.dir_completed_merged)

        # Guardar config y volver a escanear => poblar self.list_widget con los 
        # vídeos de la nueva carpeta de proyecto
        self.save_paths()
        self.scan_and_list_videos()

        self.setWindowTitle(f"StereoMaster - Project: {project_name}")





    ###########################################################
    # ITEM SELECT
    ###########################################################
        
    def scan_and_list_videos(self, reselect_base=None):
        self.log("=> Scanning videos ...")
        self.save_paths()

        # [AÑADIDO] Guardar selección actual por base_name
        selected_bases = set()
        for it in self.list_widget.selectedItems():
            dd = it.data(Qt.UserRole)
            if dd:
                base_ = dd.get("base_name", "")
                if base_:
                    selected_bases.add(base_)

        # Guardar filtro actualmente seleccionado
        current_filter = self.filter_combo.currentText()

        self.all_items_data = []
        exts = ("*.mp4", "*.mov", "*.avi", "*.mkv")

        orig_dict = {}
        depth1_dict = {}
        depth2_dict = {}
        depthf_dict = {}
        splat_dict = {}
        inpaint_dict = {}

        # --- 1) Input Videos (ORIG) ---
        for e_ in exts:
            for p in glob.glob(os.path.join(self.input_videos_edit.text(), e_)):
                bn = os.path.basename(p)
                bn_noext, _ = os.path.splitext(bn)
                orig_dict[bn_noext] = p

        # --- 2) Depth1 (Depth Crafter) ---
        for e_ in exts:
            for p in glob.glob(os.path.join(self.pre_depth_edit.text(), e_)):
                bn = os.path.basename(p)
                bn_noext, _ = os.path.splitext(bn)
                if "_depth" in bn_noext:
                    base_ = bn_noext.replace("_depth", "")
                    depth1_dict[base_] = p

        # --- 3) Depth2 (VDA) ---
        for e_ in exts:
            for p in glob.glob(os.path.join(self.pre_depth2_edit.text(), e_)):
                bn = os.path.basename(p)
                bn_noext, _ = os.path.splitext(bn)
                if "_depth" in bn_noext:
                    base_ = bn_noext.replace("_depth", "")
                    depth2_dict[base_] = p

        # --- 4) Depth Merged ---
        for e_ in exts:
            for p in glob.glob(os.path.join(self.pre_depth_final_edit.text(), e_)):
                bn = os.path.basename(p)
                bn_noext, _ = os.path.splitext(bn)
                if "_depth" in bn_noext:
                    base_ = bn_noext.replace("_depth", "")
                    depthf_dict[base_] = p

        # --- 5) Splatted ---
        for e_ in exts:
            for p in glob.glob(os.path.join(self.output_splattings_edit.text(), e_)):
                bn = os.path.basename(p)
                bn_noext, _ = os.path.splitext(bn)
                if "_splatted" in bn_noext and "_inpainting" not in bn_noext:
                    # e.g.  someVideo_splatted.mp4
                    base_ = bn_noext.replace("_splatted", "")
                    splat_dict[base_] = p

        # --- 6) Inpainted ---
        #     Check both "_splatted_inpainting_results_" and "_splatted_inpainting"
        #     so it catches e.g. "myvideo_splatted_inpainting_results_fsbs_x264.mp4"
        #     or "myvideo_splatted_inpainting.mp4"
        for e_ in exts:
            for p in glob.glob(os.path.join(self.inpainting_out_edit.text(), e_)):
                bn = os.path.basename(p)
                bn_noext, _ = os.path.splitext(bn)

                if "_splatted_inpainting_results_" in bn_noext:
                    # e.g. base_ = bn_noext.split("_splatted_inpainting_results_", 1)[0]
                    base_ = bn_noext.split("_splatted_inpainting_results_", 1)[0]
                    inpaint_dict[base_] = p
                elif "_splatted_inpainting" in bn_noext:
                    # fallback for older naming
                    base_ = bn_noext.split("_splatted_inpainting", 1)[0]
                    inpaint_dict[base_] = p

        # We do NOT scan 'dir_completed_merged' to keep them out of the main list

        # --- Combine all base_ names and build self.all_items_data ---
        all_bases = set(
            list(orig_dict.keys()) +
            list(depth1_dict.keys()) +
            list(depth2_dict.keys()) +
            list(depthf_dict.keys()) +
            list(splat_dict.keys()) +
            list(inpaint_dict.keys())
        )
        all_bases = sorted(list(all_bases))

        self.all_items_data.clear()
        for base_ in all_bases:
            ddict = {
                "base_name": base_,
                "orig": orig_dict.get(base_, ""),
                "depth1": depth1_dict.get(base_, ""),
                "depth2": depth2_dict.get(base_, ""),
                "depthf": depthf_dict.get(base_, ""),
                "splatted": splat_dict.get(base_, ""),
                "inpainted": inpaint_dict.get(base_, "")
            }

            # Build a small display for the status
            st_list = []
            detail = []
            if ddict["orig"]:
                st_list.append("ORIG")
                detail.append(f"orig={ddict['orig']}")
            if ddict["depth1"] or ddict["depth2"] or ddict["depthf"]:
                st_list.append("DEPTH")
            if ddict["splatted"]:
                st_list.append("SPLATTED")
            if ddict["inpainted"]:
                st_list.append("INPAINTED")
            if not st_list:
                st_list = ["NO VIDEO?"]

            self.all_items_data.append((base_, st_list, detail, ddict))

        self.log(f"=> Scan completed => {len(self.all_items_data)} total.")

        # Rebuild the filter combo
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItems(self.all_state_combos)
        idx = self.filter_combo.findText(current_filter)
        if idx >= 0:
            self.filter_combo.setCurrentIndex(idx)
        else:
            self.filter_combo.setCurrentIndex(0)
        self.filter_combo.blockSignals(False)

        # Apply the filter to fill the QListWidget
        self.apply_filter_list()

        # Reselect previously selected items
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            dd = item.data(Qt.UserRole)
            if dd:
                base_ = dd.get("base_name", "")
                if base_ in selected_bases:
                    item.setSelected(True)

   
        if reselect_base:
            self.list_widget.clearSelection()
            item_to_select = None
            for i in range(self.list_widget.count()):
                it = self.list_widget.item(i)
                dd2 = it.data(Qt.UserRole)
                if dd2 and dd2.get("base_name", "") == reselect_base:
                    item_to_select = it
                    break
            if item_to_select:
                self.list_widget.setCurrentItem(item_to_select)
                item_to_select.setSelected(True)
          

    def on_warp_exponent_changed(self, val):
        """
        The slider value is in [100..300], meaning 1.00..3.00 factor => val/100.0
        """
        base_ = val / 100.0
        self.warp_exponent_base = base_
        self.lbl_warp_exponent_val.setText(f"{base_:.2f}")
        # Update preview if user is in "Forward Warp" preview mode
        if self.chk_anaglyph.isChecked():
            self.show_preview(self.current_frame_idx)
            
            
    def reset_warp_exponent(self):
        self.warp_exponent_base = 1.414
        self.slider_warp_exponent.setValue(141)
        self.lbl_warp_exponent_val.setText("1.41")
        if self.chk_anaglyph.isChecked():
            self.show_preview(self.current_frame_idx)


    def on_sbs_mode_changed(self, index):
        self.sbs_mode = self.combo_sbs.currentText()
        self.log(f"[OUTPUT] => SBS Mode changed to {self.sbs_mode}")

    def on_encoder_changed(self, index):
        self.log(f"[OUTPUT] => Encoder changed to {self.output_encoder}")


    def do_scene_detect_with_threshold(self, threshold, video_path=None):
        """
        Lanza SceneDetect para el video especificado:
          - Si 'video_path' está presente, lo usa.
          - Si 'video_path' es None, usa self.selected_video_path.
        """
        import sys, os

        # 1) Determinar la ruta del video que vas a usar
        if video_path is None:
            if not self.selected_video_path:
                self.log("[ERROR] => No selected_video_path for scene detect.")
                return
            path_to_use = self.selected_video_path
        else:
            path_to_use = video_path

     
        if not os.path.isfile(path_to_use):
            self.log(f"[ERROR] => Video file doesn't exist: {path_to_use}")
            return

   
        path_abs = os.path.abspath(path_to_use)

  
        cmd = [
            sys.executable, "-m", "scenedetect",
            "-i", path_abs,
            "-o", self.dir_input_videos,
            "detect-content",
            "-t", str(threshold),
            "split-video"
        ]
        self.log("[INFO] => scenedetect => " + " ".join(cmd))

  
        self.scene_worker = SubprocessWorker(cmd)
        self.scene_worker.lineReady.connect(self.log)
        
    
        self.scene_worker.finishedSignal.connect(
            lambda success: self.on_scene_detect_finished(success, path_abs)
        )

        self.current_batch_worker = self.scene_worker
        self.batch_cancelled = False
        self.scene_worker.start()




    ###########################################################
    # PREVIEW
    ###########################################################
    def on_slider_frame_changed(self, value):
        self.current_frame_idx = value
        self.pending_slider_value = value
            # reiniciamos el timer
        self.slider_timer.start()
        
    def on_slider_timer_timeout(self):
        """
        Se llama cuando pasan 150ms sin que el usuario mueva el slider.
        """
        final_value = self.pending_slider_value
        # Aplica keyframe interpolation/exact
        if not self.enable_interpolation_local:
            self.apply_keyframe_if_exists(final_value)
        else:
            self.apply_interpolation_local(final_value)
        # Y ahora sí mostramos la preview
        self.show_preview(final_value)


    def apply_interpolation_local(self, frame_idx):
        """
        Interpolate keyframes in the immediate range, for preview only.
        """
        item = self.list_widget.currentItem()
        if not item:
            return
        dd = item.data(Qt.UserRole)
        if not dd:
            return
        base_ = dd.get("base_name","")
        if not base_:
            return
        if base_ not in self.video_params:
            return
        if "keyframes" not in self.video_params[base_]:
            return

        kf_dict = self.video_params[base_]["keyframes"]
        if not kf_dict:
            return

        frames_sorted = sorted([int(k) for k in kf_dict.keys()])
        # 1) Debajo del primer keyframe
        if frame_idx <= frames_sorted[0]:
            f0 = frames_sorted[0]
            kf_ = kf_dict[str(f0)]
            self.set_sliders_from_kf(kf_)
            return
        # 2) Encima del último keyframe
        if frame_idx >= frames_sorted[-1]:
            f0 = frames_sorted[-1]
            kf_ = kf_dict[str(f0)]
            self.set_sliders_from_kf(kf_)
            return

        # 3) Buscar el par de keyframes que encierran frame_idx
        left_f = frames_sorted[0]
        right_f= frames_sorted[-1]
        for i in range(len(frames_sorted)-1):
            f0 = frames_sorted[i]
            f1 = frames_sorted[i+1]
            if f0 <= frame_idx <= f1:
                left_f = f0
                right_f= f1
                break

        if left_f == right_f:
            kf_ = kf_dict[str(left_f)]
            self.set_sliders_from_kf(kf_)
            return

        ratio = (frame_idx - left_f)/ float(right_f - left_f)

        # 4) default_val => fallback con valores actuales (por si un keyframe no define algo)
        default_val = {
            "disp_value": self.disp_value,
            "convergence": self.conv_value,
            "brightness_value": self.brightness_value,
            "gamma_value": self.gamma_value,
            "dilate_h_value": self.dilate_h_value,
            "dilate_v_value": self.dilate_v_value,
            "blur_ksize_value": self.blur_ksize_value,
            "blur_sigma_value": self.blur_sigma_value,
            # Polylines:
            "stereo_offset_exponent_value": self.poly_exponent,
            "fill_technique_value": self.poly_fill_tech,
            "warp_exponent_base": self.warp_exponent_base
        }

        left_valset  = kf_dict[str(left_f)]
        right_valset = kf_dict[str(right_f)]

        params_inter = {}
        for p_ in default_val.keys():
            lv = left_valset.get(p_, default_val[p_])
            rv = right_valset.get(p_, default_val[p_])

            # 5) Interpolar => si ambos son numéricos, se interpola;
    
            if isinstance(lv, (int,float)) and isinstance(rv, (int,float)):
                val = lv + (rv - lv)* ratio
            elif isinstance(lv, str) and isinstance(rv, str):
                val = lv  # elegimos la versión "left"
            else:
                # fallback => coge la izq
                val = lv

            params_inter[p_] = val

 
        self.set_sliders_from_kf(params_inter)


    def scan_projects(self):
        projects_root = os.path.abspath("projects")
        if not os.path.isdir(projects_root):
            os.makedirs(projects_root, exist_ok=True)

        self.combo_projects.blockSignals(True)
        self.combo_projects.clear()

        subdirs = []
        for entry in os.scandir(projects_root):
            if entry.is_dir():
                subdirs.append(entry.name)
        subdirs.sort()

        for project_name in subdirs:
            self.combo_projects.addItem(project_name)

     
        if len(subdirs) == 0:
            self.combo_projects.addItem("<No Projects Found>")
            self.combo_projects.setEnabled(False)
        else:
            self.combo_projects.setEnabled(True)

        self.combo_projects.blockSignals(False)
        if len(subdirs) == 1:
        
            self.combo_projects.setCurrentIndex(0)
         
            self.on_project_selected(0)



    def set_sliders_from_kf(self, kf_data):
        self.disp_value=float(kf_data.get("disp_value", self.disp_value))
        self.slider_disp.setValue(int(self.disp_value))
        self.lbl_disp_val.setText(str(self.disp_value))

        self.conv_value= float(kf_data.get("convergence", self.conv_value))
        self.slider_conv.setValue(int(self.conv_value))
        self.lbl_conv_val.setText(str(self.conv_value))
        
        we = kf_data.get("warp_exponent_base", self.warp_exponent_base)
        self.warp_exponent_base = we
        self.slider_warp_exponent.setValue(int(we * 100))
        self.lbl_warp_exponent_val.setText(f"{we:.2f}")

        self.brightness_value= float(kf_data.get("brightness_value", self.brightness_value))
        self.slider_brightness.setValue(int(self.brightness_value*100))
        self.lbl_bri_val.setText(f"{self.brightness_value:.2f}")

        self.gamma_value= float(kf_data.get("gamma_value", self.gamma_value))
        g_= int(self.gamma_value*100)
        if g_<1: g_=1
        self.slider_gamma.setValue(g_)
        self.lbl_gam_val.setText(f"{self.gamma_value:.2f}")

        self.dilate_h_value= int(kf_data.get("dilate_h_value", self.dilate_h_value))
        self.slider_dilate_h.setValue(self.dilate_h_value)
        self.lbl_dh_val.setText(str(self.dilate_h_value))

        self.dilate_v_value= int(kf_data.get("dilate_v_value", self.dilate_v_value))
        self.slider_dilate_v.setValue(self.dilate_v_value)
        self.lbl_dv_val.setText(str(self.dilate_v_value))

        bk_= int(kf_data.get("blur_ksize_value", self.blur_ksize_value))
        if bk_>0 and bk_%2==0: bk_+=1
        self.blur_ksize_value=bk_
        self.slider_blur_ksize.setValue(bk_)
        self.lbl_bk_val.setText(str(bk_))

        bs_= float(kf_data.get("blur_sigma_value", self.blur_sigma_value))
        self.blur_sigma_value= bs_
        self.slider_blur_sigma.setValue(int(bs_*10))
        self.lbl_bs_val.setText(f"{bs_:.2f}")

    
        
    def on_preview_bri_changed(self, val):
        """
        El slider va de 0..200 => factor 0..2.0
        """
        factor = val / 100.0
        self.preview_bri = factor
        self.lbl_preview_bri_val.setText(f"{factor:.2f}")
        self.show_preview(self.current_frame_idx)

    def on_preview_gamma_changed(self, val):
        """
        El slider va de 10..300 => factor 0.1..3.0
        """
        factor = val / 100.0
        self.preview_gamma = factor
        self.lbl_preview_gamma_val.setText(f"{factor:.2f}")
        self.show_preview(self.current_frame_idx)

    def reset_preview_bri(self):
        """
        Restaura el preview_bri a 1.0
        """
        self.preview_bri = 1.0
        self.slider_preview_bri.setValue(100)  # 100 equivale a factor 1.0
        self.lbl_preview_bri_val.setText("1.00")
        self.show_preview(self.current_frame_idx)

    def reset_preview_gamma(self):
        """
        Restaura el preview_gamma a 1.0
        """
        self.preview_gamma = 1.0
        self.slider_preview_gamma.setValue(100)  # 100 equivale a factor 1.0
        self.lbl_preview_gamma_val.setText("1.00")
        self.show_preview(self.current_frame_idx)



    def is_video_higher_than_1080p(self, video_path):
        """
        Returns True if the video exceeds 1080p (height > 1080 or width > 1920),
        or False otherwise. If there's an error opening the file, returns False.
        """
        import os
        if not os.path.isfile(video_path):
            return False

        from decord import VideoReader, cpu
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            frame0 = vr[0].asnumpy()  # shape: (height, width, channels)
            height, width = frame0.shape[:2]

            if height > 1080 or width > 1920:
                return True
            return False
        except Exception as e:
            self.log(f"[WARN] => Could not retrieve video resolution: {e}")
            return False

         
    def show_preview(self, frame_idx):
        """
        Un solo checkbox activo a la vez. Si 'orig' está marcado => muestra original
        Si 'depth' está marcado => muestra video depth
        Si 'splatted' => muestra video splatted
        Si 'inpaint' => muestra video inpaint
        Si 'polylines' => genera anaglifo con polylines + la depth elegida en combo
        Si 'anaglyph' => anaglifo con forward warp + la depth elegida en combo
        Si no existe el archivo => pantalla negra + mensaje.
        Si la combo depth es "No Depth" => pantalla negra + mensaje.
        """

        try:
            # Primero comprobamos rango de frame_idx
            if frame_idx < 0 or frame_idx >= self.num_frames:
                self.log("[WARN] => frame_idx out of range.")
                return

            # Miramos qué checkbox está activo:
            if self.chk_orig.isChecked():
                # => Mostrar video original, si existe
                if not self.vid_original:
                    self.log("[WARN] => No vid_original to show in 'Orig' mode.")
                    return

                fr_ = self.get_video_frame(self.vid_original, frame_idx)
                if fr_ is None:
                    self.log("[ERROR] => can't read original frame.")
                    return

                # Ajuste bri/gamma
                fr_ = np.clip(fr_ * self.preview_bri, 0, 1)
                fr_ = fr_ ** (1.0 / max(self.preview_gamma, 1e-5))
                self.show_image_in_label(fr_, self.label_preview)
                return

            elif self.chk_depth.isChecked():
                # => Mostrar video depth que esté en el combo (Depth1/Depth2/Merged)
                dfr = self.get_depth_video_frame(frame_idx)
                if dfr is None:
                    self.log("[WARN] => No depth video for that choice or 'No Depth' selected.")
                    return

                # Puede venir en gris => convertir a RGB
                if dfr.ndim == 2:
                    dfr = cv2.cvtColor(dfr, cv2.COLOR_GRAY2RGB)

                dfr = np.clip(dfr * self.preview_bri, 0, 1)
                dfr = dfr ** (1.0 / max(self.preview_gamma, 1e-5))

                self.show_image_in_label(dfr, self.label_preview)
                return

            elif self.chk_splatted.isChecked():
                # => Mostrar video splatted, si existe
                if not self.vid_splatted:
                    self.log("[WARN] => No splatted video loaded.")
                    return

                fr_ = self.get_video_frame(self.vid_splatted, frame_idx)
                if fr_ is None:
                    self.log("[WARN] => can't read splatted frame.")
                    return

                fr_ = np.clip(fr_ * self.preview_bri, 0, 1)
                fr_ = fr_ ** (1.0 / max(self.preview_gamma, 1e-5))

                self.show_image_in_label(fr_, self.label_preview)
                return

            elif self.chk_inpaint.isChecked():
                # => Mostrar video inpaint, si existe
                if not self.vid_inpainted:
                    self.log("[WARN] => No inpainted video loaded.")
                    return

                fr_ = self.get_video_frame(self.vid_inpainted, frame_idx)
                if fr_ is None:
                    self.log("[WARN] => can't read inpaint frame.")
                    return

                fr_ = np.clip(fr_ * self.preview_bri, 0, 1)
                fr_ = fr_ ** (1.0 / max(self.preview_gamma, 1e-5))

                self.show_image_in_label(fr_, self.label_preview)
                return

            elif self.chk_polylines.isChecked():
                # => Polylines => requiere original + depth
                if not self.vid_original:
                    self.log("[WARN] => Polylines: No vid_original for left-eye.")
                    return

                fr_orig = self.get_video_frame(self.vid_original, frame_idx)
                if fr_orig is None:
                    self.log("[WARN] => can't read original.")
                    return

                depth_ = self.get_depth_gray(frame_idx)
                if depth_ is None:
                    self.log("[WARN] => Polylines: No depth selected or no depth file.")
                    return

                # Genera “right eye” usando polylines
                right_eye = self.apply_polylines(fr_orig, depth_)

                # Anaglifo: R = original left, G/B = right
                anag = np.zeros_like(fr_orig)
                anag[..., 0] = fr_orig[..., 0]
                anag[..., 1] = right_eye[..., 1]
                anag[..., 2] = right_eye[..., 2]

                anag = np.clip(anag * self.preview_bri, 0, 1)
                anag = anag ** (1.0 / max(self.preview_gamma, 1e-5))

                self.show_image_in_label(anag, self.label_preview)
                return

            elif self.chk_anaglyph.isChecked():
                # => Anaglifo => requiere original + depth
                if not self.vid_original:
                    self.log("[WARN] => Anaglyph: No vid_original for left-eye.")
                    return

                fr_orig = self.get_video_frame(self.vid_original, frame_idx)
                if fr_orig is None:
                    self.log("[WARN] => can't read original.")
                    return

                depth_ = self.get_depth_gray(frame_idx)
                if depth_ is None:
                    self.log("[WARN] => Anaglyph: No depth selected or no depth file.")
                    return

                # Forward warp => right eye
                right_eye = self.apply_forward_warp(fr_orig, depth_)

                anag = np.zeros_like(fr_orig)
                anag[..., 0] = fr_orig[..., 0]
                anag[..., 1] = right_eye[..., 1]
                anag[..., 2] = right_eye[..., 2]

                anag = np.clip(anag * self.preview_bri, 0, 1)
                anag = anag ** (1.0 / max(self.preview_gamma, 1e-5))

                self.show_image_in_label(anag, self.label_preview)
                return

            else:
                # Si ningún checkbox está marcado => no hacemos nada
                self.log("[WARN] => No checkbox active.")
                return

        except Exception as e:
            self.log(f"[ERROR show_preview] => {e}")
            self.log("[ERROR] => Exception in show_preview")
            

    def get_video_frame(self, vid, frame_idx):
        if not vid:
            return None
        if frame_idx < 0 or frame_idx >= len(vid):
            return None
        try:
            fr_ = vid[frame_idx].asnumpy().astype(np.float32)/255.0
            return fr_
        except:
            return None

    def get_depth_video_frame(self, frame_idx):
        """
        Devuelve un frame de mapa de profundidad en RGB (o None),
        aplicando brightness/gamma/dilat/blur según los sliders de profundidad.
        Usa 'selected_preview_depth' para decidir si es Depth1, Depth2 o Merge.
        """
        choice = self.selected_preview_depth
        if choice == "No Depth":
            return None

        # --- 1) Leemos el frame de profundidad (raw) según la opción ---
        if choice == "Depth1":
            # depth1
            dframe = self.get_video_frame(self.vid_depth, frame_idx)

        elif choice == "Depth2":
            # depth2
            dframe = self.get_video_frame(self.vid_depth2, frame_idx)

        elif choice == "Merged":
            # Fusión “al vuelo” (ignorando self.vid_depth_merged):
            if not (self.vid_depth and self.vid_depth2):
                return None
            if frame_idx >= len(self.vid_depth) or frame_idx >= len(self.vid_depth2):
                return None

            try:
                d1_ = self.vid_depth[frame_idx].asnumpy().astype(np.float32)/255.0
                d2_ = self.vid_depth2[frame_idx].asnumpy().astype(np.float32)/255.0
            except:
                return None

            # Convertimos a gris
            if d1_.ndim == 3 and d1_.shape[-1] == 3:
                d1_ = d1_.mean(axis=-1)
            if d2_.ndim == 3 and d2_.shape[-1] == 3:
                d2_ = d2_.mean(axis=-1)

            # Normalizamos cada depth
            mn1, mx1 = d1_.min(), d1_.max()
            if (mx1 - mn1) > 1e-5:
                d1_ = (d1_ - mn1)/(mx1 - mn1)
            else:
                d1_[:] = 0.0

            mn2, mx2 = d2_.min(), d2_.max()
            if (mx2 - mn2) > 1e-5:
                d2_ = (d2_ - mn2)/(mx2 - mn2)
            else:
                d2_[:] = 0.0

            # Fusión
            a_ = self.fusion_alpha / 100.0
            merged_ = d1_*(1-a_) + d2_*a_
            dframe = merged_

        else:
            # Por si acaso
            return None

        # Si no hay frame leído, salimos
        if dframe is None:
            return None

        # --- 2) Convertir a gris si no lo está (por seguridad) ---
        #    (En Merged ya lo convertimos; en Depth1/Depth2 no siempre).
        if dframe.ndim == 3 and dframe.shape[-1] == 3:
            dframe = dframe.mean(axis=-1)

        # --- 3) Normalizar [0..1] ---
        d_min, d_max = dframe.min(), dframe.max()
        if (d_max - d_min) > 1e-6:
            dframe = (dframe - d_min)/(d_max - d_min)
        else:
            dframe[:] = 0.0

        # --- 4) Aplicar brillo/gamma de la profundidad (sliders “Depth Bri/Gam”) ---
        dframe = np.clip(dframe * self.brightness_value, 0, 1)
        dframe = dframe ** (1.0 / max(self.gamma_value, 1e-5))

        # --- 5) Aplicar dilate/blur (sliders “DilateH/V, BlurK/Sigma”) ---
        dframe = apply_depth_preprocess(
            dframe,
            self.dilate_h_value,
            self.dilate_v_value,
            self.blur_ksize_value,
            self.blur_sigma_value
        )

        # --- 6) Replicar a 3 canales para mostrarlo en color ---
        depth_rgb = np.stack([dframe]*3, axis=-1)  # [H,W,3]
        return depth_rgb





    def get_depth_gray(self, frame_idx):
        """
        Similar a get_depth_video_frame, pero devuelves un 2D float32 con la Depth,
        normalizada y con dilate/blur si quieres. 
        Si no existe => None
        """
        dfr = self.get_depth_video_frame(frame_idx)
        if dfr is None:
            return None

        # Pasar a gris si está en 3 canales
        if dfr.ndim == 3 and dfr.shape[-1] == 3:
            dfr = dfr.mean(axis=-1)

        # Normalización
        mn, mx = dfr.min(), dfr.max()
        if (mx - mn) > 1e-6:
            dfr = (dfr - mn)/(mx - mn)
        else:
            dfr = np.zeros_like(dfr, dtype=np.float32)

        # Optional: brightness/gamma con tus sliders de depth
        dfr = np.clip(dfr * self.brightness_value, 0,1)
        dfr = dfr ** (1.0 / max(self.gamma_value, 1e-5))

        # dilate/blur => apply_depth_preprocess
        dfr = apply_depth_preprocess(dfr, self.dilate_h_value, self.dilate_v_value,
                                     self.blur_ksize_value, self.blur_sigma_value)
        return dfr
        

    def apply_polylines(self, left_eye, depth_gray):
        """
        Genera la imagen 'right_eye' con polylines. 
        left_eye => [H,W,3], float32 [0..1]
        depth_gray => [H,W], float32 [0..1]
        Devuelve right_eye => [H,W,3], float32 [0..1].
        """
        disp_frame = (depth_gray*2.0 - 1.0)*self.disp_value + self.conv_value

        d_min, d_max = disp_frame.min(), disp_frame.max()
        if (d_max - d_min) < 1e-9:
            disp_norm = np.zeros_like(disp_frame, dtype=np.float32)
        else:
            disp_norm = ((disp_frame - d_min)/(d_max - d_min)).astype(np.float32)

        left_u8 = (left_eye*255).astype(np.uint8)

        try:
            right_u8 = apply_stereo_divergence_polylines(
                original_image= left_u8,
                normalized_depth= disp_norm,
                divergence_px= self.disp_value,
                separation_px= self.conv_value,
                stereo_offset_exponent= self.poly_exponent,
                fill_technique= self.poly_fill_tech
            )
            return right_u8.astype(np.float32)/255.0
        except Exception as e:
            self.log(f"[ERROR] polylines => {e}")
            return left_eye  # fallback
            
            
    def apply_forward_warp(self, left_eye, depth_gray):
        import torch
        try:
            disp_ = (depth_gray*2.0 - 1.0)* self.disp_value
            left_t = torch.from_numpy(left_eye).permute(2,0,1).unsqueeze(0).float().cuda()
            disp_t = torch.from_numpy(disp_).unsqueeze(0).unsqueeze(0).float().cuda()

            with torch.no_grad():
                right_t, occ_t = self.stereo_warper(left_t, disp_t, convergence=self.conv_value)
            rn = right_t.squeeze(0).permute(1,2,0).cpu().numpy()
            self.stereo_warper.warp_exponent_base = self.warp_exponent_base
            rn = np.clip(rn,0,1)
            return rn
        except Exception as e:
            self.log("[ERROR] => forward_warp =>", e)
            return left_eye  # fallback



    def show_image_in_label(self, img_rgb, label_widget):
        import cv2
        import numpy as np
        hh= label_widget.height()
        ww= label_widget.width()
        hi,wi= img_rgb.shape[:2]
        sc= min(ww/wi, hh/hi)
        newW= int(wi*sc)
        newH= int(hi*sc)
        rr= cv2.resize(img_rgb,(newW,newH), interpolation=cv2.INTER_AREA)
        u8= np.clip(rr*255,0,255).astype(np.uint8)
        qimg= QImage(u8, newW,newH, QImage.Format_RGB888)
        pix= QPixmap.fromImage(qimg)
        label_widget.setPixmap(pix)

    def on_preview_depth_changed(self, idx):
        self.selected_preview_depth= self.preview_depth_combo.currentText()
        self.log(f"[INFO] => preview depth => {self.selected_preview_depth}")
        self.show_preview(self.current_frame_idx)

    def on_fusion_alpha_changed(self, val):
        self.fusion_alpha= val
        self.show_preview(self.current_frame_idx)


        
    def do_polylines_sbs(self):
        if not self.selected_video_path:
            QMessageBox.warning(self, "Attention", "No video selected.")
            return

        # Escoger depth (Depth1, Depth2, Merged) según self.selected_splat_depth
        if self.selected_splat_depth == "Depth1":
            depth_path = self.selected_depth_path
        elif self.selected_splat_depth == "Depth2":
            depth_path = self.selected_depth_path2
        else:
            depth_path = self.selected_depth_merged

        if not depth_path or not os.path.isfile(depth_path):
            QMessageBox.warning(self, "Attention", "Depth file not found.")
            return

        base_ = os.path.splitext(os.path.basename(self.selected_video_path))[0]
        out_path = os.path.join(self.output_splattings_edit.text(), base_+"_polylines_sbs.mp4")

        # 1) Generar JSON de keyframes si existe
        keyf_json_path = ""
        if base_ in self.video_params and "keyframes" in self.video_params[base_]:
            kf_dict = self.video_params[base_]["keyframes"]
            if kf_dict:
                # igual que do_splatting:
                real_dict = {}
                for k,v in kf_dict.items():
                    try:
                        fr_int = int(k)
                        real_dict[fr_int] = v
                    except:
                        pass
                if real_dict:
                    keyf_json_path = f"temp_keyframes_{base_}.json"
                    try:
                        with open(keyf_json_path, "w", encoding="utf-8") as f:
                            json.dump({base_: real_dict}, f, indent=2)
                        self.log(f"[INFO] => Keyframes saved => {keyf_json_path}")
                    except Exception as e:
                        self.log(f"[ERROR] => Could not save keyframes => {e}")
                        keyf_json_path = ""

        # 2) Construir el comando a tu script => e.g. "depth_splatting.py" con --generate_stereo
        cmd = [
                sys.executable, "depth_splatting.py",
                "--single_video", self.selected_video_path,
                "--input_depth_maps", os.path.dirname(depth_path),
                "--output_splatted", self.inpainting_out_edit.text(),
                "--max_disp", str(self.disp_value),
                "--convergence", str(self.conv_value),
                "--depth_brightness_value", str(self.brightness_value),
                "--depth_gamma_value", str(self.gamma_value),
                "--stereo_offset_exponent", str(self.poly_exponent),
                "--dilate_h", str(self.dilate_h_value),
                "--dilate_v", str(self.dilate_v_value),
                "--blur_ksize", str(self.blur_ksize_value),
                "--blur_sigma", str(self.blur_sigma_value),
                "--generate_stereo", "True",  # Tells script to create a stereo pair via polylines
                "--fill_technique", self.poly_fill_tech,          
                "--encoder", self.output_encoder,
                # half-SBS or full-SBS
                "--sbs_mode", "half" if self.sbs_mode == "HSBS" else "full"
            ]
            
         
        cmd.append("--warp_exponent_base")
        cmd.append(str(self.warp_exponent_base))    

        if keyf_json_path:
            cmd.append("--keyframes_json")
            cmd.append(keyf_json_path)

         
        self.polylines_sbs_worker = SubprocessWorker(cmd)
        self.polylines_sbs_worker.lineReady.connect(self.log)
        self.polylines_sbs_worker.finishedSignal.connect(self.on_polylines_sbs_finished)
        self.current_batch_worker = self.polylines_sbs_worker
        self.batch_cancelled = False
        self.polylines_sbs_worker.start()

    def on_polylines_sbs_finished(self, success):
        self.current_batch_worker = None
        if success:
            self.log("[OK] => Polylines SBS => done => refresh")
            self.scan_and_list_videos()
        else:
            self.log("[ERROR] => Polylines SBS => fail")
            

    ###########################################################
    # GENERATE DEPTH1
    ###########################################################

    def on_depth_generated(self, success):
        self.current_batch_worker=None
        if success:
            self.log("[OK] => Depth => done => refresh")
            self.scan_and_list_videos()
        else:
            self.log("[ERROR] => Depth => fail")

    ###########################################################
    # BATCH
    ###########################################################
    def batch_generate_depth_selected(self):
        sel_items=self.list_widget.selectedItems()
        if not sel_items:
            QMessageBox.information(self, "Info", "No items selected for batch depth.")
            return
        self.batch_depth_queue=[]
        for it in sel_items:
            dd= it.data(Qt.UserRole)
            if dd:
                self.batch_depth_queue.append(dd)
        self.batch_depth_idx=0
        self.batch_cancelled = False
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setRange(0,len(self.batch_depth_queue))
        self.log(f"[INFO] => Starting BATCH depth => {len(self.batch_depth_queue)} items.")
        self.batch_depth_phase=0
        self.process_next_depth_in_queue()

    def process_next_depth_in_queue(self):
        if self.batch_cancelled:
            self.log("[INFO] => batch depth => cancelled => stopping queue.")
            return
        if self.batch_depth_idx >= len(self.batch_depth_queue):
            self.log("[INFO] => batch depth => done.")
            return

        dd = self.batch_depth_queue[self.batch_depth_idx]
        base_ = dd.get("base_name", "")
        orig = dd.get("orig", "")
        if not orig or not os.path.isfile(orig):
            self.log(f"[WARN] => {base_} => no orig => skip.")
            self.batch_depth_idx += 1
            self.batch_progress_bar.setValue(self.batch_depth_idx)
            self.process_next_depth_in_queue()
            return

        # ------------------------------
        # 1) Decide output path for depth
        # ------------------------------
        src_choice = self.combo_batch_depth_source.currentText()
        if src_choice == "DepthCrafter":
            out_path = os.path.join(self.pre_depth_edit.text(), base_ + "_depth.mp4")
        elif src_choice == "VDA":
            out_path = os.path.join(self.pre_depth2_edit.text(), base_ + "_depth.mp4")
        else:
            # "Both": Depth1 then Depth2 in two steps
            if self.batch_depth_phase == 0:
                out_path = os.path.join(self.pre_depth_edit.text(), base_ + "_depth.mp4")
            else:
                out_path = os.path.join(self.pre_depth2_edit.text(), base_ + "_depth.mp4")

        # ------------------------------
        # 2) Check if file already exists & checkbox not checked => ask
        # ------------------------------
        if (not self.check_overwrite_depth.isChecked()) and os.path.isfile(out_path):
            from PyQt5.QtWidgets import QMessageBox
            box = QMessageBox(self)
            box.setWindowTitle("Overwrite Depth?")
            msg = (
                f"Depth file already exists:\n{out_path}\n"
                "Overwrite?\n\n"
                "Yes => Overwrite only this item\n"
                "No => Skip this item\n"
                "Yes to all => Do not ask again and overwrite all"
            )
            box.setText(msg)
            yes_btn = box.addButton("Yes", QMessageBox.YesRole)
            no_btn  = box.addButton("No",  QMessageBox.NoRole)
            all_btn = box.addButton("Yes to all", QMessageBox.AcceptRole)
            box.exec_()

            if box.clickedButton() == no_btn:
                self.log(f"[INFO] => skipping {base_} => user chose no overwrite.")
                self.batch_depth_idx += 1
                self.batch_progress_bar.setValue(self.batch_depth_idx)
                self.process_next_depth_in_queue()
                return
            elif box.clickedButton() == all_btn:
                self.log("[INFO] => user chose 'Yes to all' => enabling Overwrite Depth checkbox.")
                self.check_overwrite_depth.setChecked(True)
            # if it's yes_btn => overwrite just this item

        # ------------------------------
        # 3) Original logic (unchanged)
        # ------------------------------
        if src_choice == "DepthCrafter":
            cmd = [
                sys.executable, "depth_splatting.py",
                "--single_video", orig,
                "--input_depth_maps", self.pre_depth_edit.text(),
                "--output_splatted", "./temp_splatted",
                "--max_disp", "0",
                "--convergence", "0",
                "--depth_only", "True",
                "--stereo_offset_exponent", str(self.poly_exponent),
                "--orig_brightness_value", str(self.orig_brightness),
                "--orig_gamma_value", str(self.orig_gamma),
                "--max_res", str(self.crafter_maxres_value),
                "--window_size", str(self.crafter_window),
                "--overlap", str(self.crafter_overlap),
                "--num_denoising_steps", str(self.crafter_denoising_steps),
                "--guidance_scale", f"{self.crafter_guidance_scale:.1f}",
                "--seed", str(self.crafter_seed),
            ]

            cmd.append("--warp_exponent_base")
            cmd.append(str(self.warp_exponent_base))             
            self.batch_depth_worker = SubprocessWorker(cmd)
            self.current_batch_worker = self.batch_depth_worker
            self.batch_depth_worker.lineReady.connect(self.log)
            self.batch_depth_worker.finishedSignal.connect(self.on_depth_generated_batch_single)
            self.batch_depth_worker.start()

        elif src_choice == "VDA":
            cmd = [
                sys.executable, "run.py",
                "--input_video", orig,
                "--output_dir", self.pre_depth2_edit.text(),
                "--max_len", "-1",
                "--target_fps", "-1",
                "--max_res", str(self.vda_maxres_value),
                "--input_size", str(self.vda_input_size),
                "--encoder", str(self.vda_encoder),
            ]
            if self.vda_use_fp16:
                cmd.append("--use_fp16")

            if self.vda_use_cudnn:
                cmd.append("--use_cudnn_benchmark")

            self.batch_depth_worker = SubprocessWorker(cmd)
            self.current_batch_worker = self.batch_depth_worker
            self.batch_depth_worker.lineReady.connect(self.log)
            self.batch_depth_worker.finishedSignal.connect(self.on_depth_generated_batch_single)
            self.batch_depth_worker.start()

        else:
            # "Both": Depth1 then Depth2 in two steps
            if self.batch_depth_phase == 0:
                cmd = [
                    sys.executable, "depth_splatting.py",
                    "--single_video", orig,
                    "--input_depth_maps", self.pre_depth_edit.text(),
                    "--output_splatted", "./temp_splatted",
                    "--max_disp", "0",
                    "--convergence", "0",
                    "--depth_only", "True",  
                    "--stereo_offset_exponent", str(self.poly_exponent),                    
                    "--orig_brightness_value", str(self.orig_brightness),
                    "--orig_gamma_value", str(self.orig_gamma),
                    "--max_res", str(self.crafter_maxres_value), 
                    "--window_size", str(self.crafter_window),
                    "--overlap", str(self.crafter_overlap),
                    "--num_denoising_steps", str(self.crafter_denoising_steps),
                    "--guidance_scale", f"{self.crafter_guidance_scale:.1f}",
                    "--seed", str(self.crafter_seed),
                ]
                
                cmd.append("--warp_exponent_base")
                cmd.append(str(self.warp_exponent_base)) 
                self.batch_depth_worker = SubprocessWorker(cmd)
                self.current_batch_worker = self.batch_depth_worker
                self.batch_depth_worker.lineReady.connect(self.log)
                self.batch_depth_worker.finishedSignal.connect(self.on_depth_generated_batch_phase1)
                self.batch_depth_worker.start()
            else:
                cmd = [
                    sys.executable, "run.py",
                    "--input_video", orig,
                    "--output_dir", self.pre_depth2_edit.text(),
                    "--max_len", "-1", "--target_fps", "-1",
                    "--max_res", str(self.vda_maxres_value),  
                    "--input_size", str(self.vda_input_size),
                    "--encoder", str(self.vda_encoder),
                ]
                if self.vda_use_fp16:
                    cmd.append("--use_fp16")

                if self.vda_use_cudnn:
                    cmd.append("--use_cudnn_benchmark")

                self.batch_depth_worker = SubprocessWorker(cmd)
                self.current_batch_worker = self.batch_depth_worker
                self.batch_depth_worker.lineReady.connect(self.log)
                self.batch_depth_worker.finishedSignal.connect(self.on_depth_generated_batch_phase2)
                self.batch_depth_worker.start()


    def on_depth_generated_batch_single(self, success):
        self.current_batch_worker=None
        if success:
            self.log("[OK] => Depth single => done.")
        else:
            self.log("[ERROR] => Depth single => fail.")
        self.batch_depth_idx+=1
        self.batch_progress_bar.setValue(self.batch_depth_idx)
        self.scan_and_list_videos()
        self.process_next_depth_in_queue()

    def on_depth_generated_batch_phase1(self, success):
        self.current_batch_worker=None
        if success:
            self.log("[OK] => Depth1 => done (batch).")
        else:
            self.log("[ERROR] => Depth1 => fail (batch).")
        self.batch_depth_phase=1
        self.scan_and_list_videos()
        self.process_next_depth_in_queue()

    def on_depth_generated_batch_phase2(self, success):
        self.current_batch_worker=None
        if success:
            self.log("[OK] => Depth2 => done (batch).")
        else:
            self.log("[ERROR] => Depth2 => fail (batch).")
        self.batch_depth_idx+=1
        self.batch_depth_phase=0
        self.batch_progress_bar.setValue(self.batch_depth_idx)
        self.scan_and_list_videos()
        self.process_next_depth_in_queue()

    def select_all_filtered(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setSelected(True)
            
    def unselect_all_filtered(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setSelected(False)


    def batch_splat_selected(self):
        sel_items= self.list_widget.selectedItems()
        if not sel_items:
            QMessageBox.information(self,"Info","No items selected for batch splat.")
            return
        self.batch_splat_queue=[]
        for it in sel_items:
            dd= it.data(Qt.UserRole)
            if dd:
                self.batch_splat_queue.append(dd)
        self.batch_splat_idx=0
        self.batch_cancelled = False
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setRange(0,len(self.batch_splat_queue))
        self.log(f"[INFO] => Starting BATCH splat => {len(self.batch_splat_queue)} items.")
        self.process_next_splat_in_queue()

    def process_next_splat_in_queue(self):
        if self.batch_cancelled:
            self.log("[INFO] => batch splat => cancelled => stopping queue.")
            return
        if self.batch_splat_idx >= len(self.batch_splat_queue):
            self.log("[INFO] => batch splat => done.")
            return

        dd = self.batch_splat_queue[self.batch_splat_idx]
        base_ = dd.get("base_name", "")
        orig_ = dd.get("orig", "")
        if not orig_ or not os.path.isfile(orig_):
            self.log(f"[WARN] => {base_} => no orig => skip.")
            self.batch_splat_idx += 1
            self.batch_progress_bar.setValue(self.batch_splat_idx)
            self.process_next_splat_in_queue()
            return

     
        if self.is_video_higher_than_1080p(orig_):
            self.log(f"[WARN] => {base_} => resolution higher than 1080p => skipping this item.")
            self.batch_splat_idx += 1
            self.batch_progress_bar.setValue(self.batch_splat_idx)
            self.process_next_splat_in_queue()
            return

        # ----------------------------------
        # 1) Output path for the splatted file
        # ----------------------------------
        out_path = os.path.join(self.output_splattings_edit.text(), base_ + "_splatted.mp4")

        # ----------------------------------
        # 2) Overwrite check
        # ----------------------------------
        if (not self.check_overwrite_splat.isChecked()) and os.path.isfile(out_path):
            from PyQt5.QtWidgets import QMessageBox
            box = QMessageBox(self)
            box.setWindowTitle("Overwrite Splat?")
            msg = (
                f"File already exists:\n{out_path}\n"
                "Overwrite?\n\n"
                "Yes => Overwrite only this item\n"
                "No => Skip this item\n"
                "Yes to all => Do not ask again and overwrite all"
            )
            box.setText(msg)
            yes_btn = box.addButton("Yes", QMessageBox.YesRole)
            no_btn  = box.addButton("No",  QMessageBox.NoRole)
            all_btn = box.addButton("Yes to all", QMessageBox.AcceptRole)
            box.exec_()

            if box.clickedButton() == no_btn:
                self.log(f"[INFO] => skipping {base_} (user chose no).")
                self.batch_splat_idx += 1
                self.batch_progress_bar.setValue(self.batch_splat_idx)
                self.process_next_splat_in_queue()
                return
            elif box.clickedButton() == all_btn:
                self.log("[INFO] => user chose 'Yes to all' => enabling Overwrite Splat checkbox.")
                self.check_overwrite_splat.setChecked(True)
            # if yes_btn => overwrite only this


        spl_src = self.combo_batch_splat_source.currentText()
        if spl_src == "depth1":
            dep = dd.get("depth1", "")
        elif spl_src == "depth2":
            dep = dd.get("depth2", "")
        else:
            dep = dd.get("depthf", "")

        if not dep or not os.path.isfile(dep):
            self.log(f"[WARN] => {base_} => no depth => skip.")
            self.batch_splat_idx += 1
            self.batch_progress_bar.setValue(self.batch_splat_idx)
            self.process_next_splat_in_queue()
            return

        # Keyframes logic (unchanged):
        keyf_json_path = ""
        if base_ in self.video_params and "keyframes" in self.video_params[base_]:
            kf_dict = self.video_params[base_]["keyframes"]
            if kf_dict:
                real_dict = {}
                for k,v in kf_dict.items():
                    try:
                        key_int = int(k)
                        real_dict[key_int] = v
                    except:
                        pass
                if real_dict:
                    keyf_json_path = f"temp_keyframes_{base_}.json"
                    try:
                        with open(keyf_json_path, "w", encoding="utf-8") as ff:
                            json.dump({base_: real_dict}, ff, indent=2)
                        self.log(f"[INFO] => Keyframes saved => {keyf_json_path}")
                    except Exception as e:
                        self.log(f"[ERROR] => Could not save keyframes => {e}")
                        keyf_json_path = ""

        cmd = [
            sys.executable, "depth_splatting.py",
            "--single_video", orig_,
            "--input_depth_maps", os.path.dirname(dep),
            "--output_splatted", self.output_splattings_edit.text(),
            "--max_disp", str(self.disp_value),
            "--convergence", str(self.conv_value),
            "--depth_brightness_value", str(self.brightness_value),
            "--depth_gamma_value", str(self.gamma_value),
            "--dilate_h", str(self.dilate_h_value),
            "--dilate_v", str(self.dilate_v_value),
            "--blur_ksize", str(self.blur_ksize_value),
            "--blur_sigma", str(self.blur_sigma_value),
            "--stereo_offset_exponent", str(self.poly_exponent),
            "--process_length", "-1",
            "--max_res", str(self.depth_maxres_value),
            "--window_size", str(self.crafter_window),
            "--overlap", str(self.crafter_overlap),
            "--enable_interpolation", "True",
            "--num_denoising_steps", str(self.crafter_denoising_steps),
            "--guidance_scale", f"{self.crafter_guidance_scale:.1f}",
            "--seed", str(self.crafter_seed),
        ]
        
        cmd.append("--warp_exponent_base")
        cmd.append(str(self.warp_exponent_base)) 
        if keyf_json_path:
            cmd.append("--keyframes_json")
            cmd.append(keyf_json_path)
         
        cmd.append("--warp_exponent_base")
        cmd.append(str(self.warp_exponent_base)) 
        self.batch_splat_worker = SubprocessWorker(cmd)
        self.current_batch_worker = self.batch_splat_worker
        self.batch_splat_worker.lineReady.connect(self.log)
        self.batch_splat_worker.finishedSignal.connect(self.on_splatted_batch)
        self.batch_splat_worker.start()


    def on_splatted_batch(self, success):
        self.current_batch_worker=None
        if success:
            self.log("[OK] => splat => done (batch item).")
        else:
            self.log("[ERROR] => splat => fail (batch item).")
        self.batch_splat_idx+=1
        self.batch_progress_bar.setValue(self.batch_splat_idx)
        self.scan_and_list_videos()
        self.process_next_splat_in_queue()

    def on_inpaint_threshold_batch_changed(self, val):
        self.inpaint_threshold_batch = val
        self.save_depth_global_params()

    def batch_inpaint_selected(self):
        sel_items= self.list_widget.selectedItems()
        if not sel_items:
            QMessageBox.information(self,"Info","No items selected for batch inpaint.")
            return
        self.batch_inpaint_queue=[]
        for it in sel_items:
            dd= it.data(Qt.UserRole)
            if dd:
                self.batch_inpaint_queue.append(dd)
        self.batch_inpaint_idx=0
        self.batch_cancelled = False
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setRange(0,len(self.batch_inpaint_queue))
        self.log(f"[INFO] => Starting BATCH inpaint => {len(self.batch_inpaint_queue)} items.")
        self.process_next_inpaint_in_queue()

    def process_next_inpaint_in_queue(self):
        if self.batch_cancelled:
            self.log("[INFO] => batch inpaint => cancelled => stopping queue.")
            return
        if self.batch_inpaint_idx >= len(self.batch_inpaint_queue):
            self.log("[INFO] => batch inpaint => done.")
            return

        dd = self.batch_inpaint_queue[self.batch_inpaint_idx]
        base_ = dd.get("base_name", "")
        spl_  = dd.get("splatted", "")
        orig_ = dd.get("orig", "")

        if not spl_ or not os.path.isfile(spl_):
            self.log(f"[WARN] => {base_} => no splatted => skip.")
            self.batch_inpaint_idx += 1
            self.batch_progress_bar.setValue(self.batch_inpaint_idx)
            self.process_next_inpaint_in_queue()
            return

        # ------------------------------------------------
        # 1) Output path for inpainted video
        # ------------------------------------------------
        out_name = f"{base_}_splatted_inpainting_results_{self.sbs_mode}_{self.output_encoder}.mp4"
        out_path = os.path.join(self.inpainting_out_edit.text(), out_name)

        # ------------------------------------------------
        # 2) Overwrite check with "Yes/No/Yes to all"
        # ------------------------------------------------
        if (not self.check_overwrite_inpaint.isChecked()) and os.path.isfile(out_path):
            from PyQt5.QtWidgets import QMessageBox
            box = QMessageBox(self)
            box.setWindowTitle("Overwrite Inpaint?")
            msg = (
                f"Inpaint file already exists:\n{out_path}\n"
                "Overwrite?\n\n"
                "Yes => Overwrite only this item\n"
                "No => Skip this item\n"
                "Yes to all => Do not ask again and overwrite all"
            )
            box.setText(msg)
            yes_btn = box.addButton("Yes", QMessageBox.YesRole)
            no_btn  = box.addButton("No",  QMessageBox.NoRole)
            all_btn = box.addButton("Yes to all", QMessageBox.AcceptRole)
            box.exec_()

            if box.clickedButton() == no_btn:
                self.log(f"[INFO] => skipping {base_} (user chose no).")
                self.batch_inpaint_idx += 1
                self.batch_progress_bar.setValue(self.batch_inpaint_idx)
                self.process_next_inpaint_in_queue()
                return
            elif box.clickedButton() == all_btn:
                self.log("[INFO] => user chose 'Yes to all' => enabling Overwrite Inpaint checkbox.")
                self.check_overwrite_inpaint.setChecked(True)
            # if yes_btn => overwrite only this


        cmd = [
            sys.executable, "inpainting.py",
            "--single_video", spl_,
            "--output_folder", self.inpainting_out_edit.text(),
            "--threshold_mask", f"{self.inpaint_threshold_batch:.3f}",
            "--orig_video", orig_
        ]
        cmd.append("--num_inference_steps")
        cmd.append(str(self.inpaint_num_inference_steps_batch))
        cmd.append(f"--tile_num={self.inpaint_tile_num_batch}")
        cmd.append("--sbs_mode")
        cmd.append(self.sbs_mode)
        cmd.append("--encoder")
        cmd.append(self.output_encoder)
        cmd.append("--overlap")
        cmd.append(str(self.inpaint_overlap_batch))

        self.batch_inpaint_worker = SubprocessWorker(cmd)
        self.current_batch_worker = self.batch_inpaint_worker
        self.batch_inpaint_worker.lineReady.connect(self.log)
        self.batch_inpaint_worker.finishedSignal.connect(self.on_inpaint_done_batch)
        self.batch_inpaint_worker.start()

        

        
    def on_inpaint_done_batch(self, success):
        self.current_batch_worker=None
        if success:
            self.log("[OK] => inpaint => done (batch item).")
        else:
            self.log("[ERROR] => inpaint => fail (batch item).")
        self.batch_inpaint_idx+=1
        self.batch_progress_bar.setValue(self.batch_inpaint_idx)
        self.scan_and_list_videos()
        self.process_next_inpaint_in_queue()
        
        
    def batch_direct_stereo(self):
        """
        Runs the entire pipeline for each selected video:
          - Generate Depth1 and/or Depth2
          - Merge if needed
          - Splat
          - Inpaint
        all in one go, without stopping until finished or canceled.
        """
        sel_items = self.list_widget.selectedItems()
        if not sel_items:
            QMessageBox.information(self, "Info", "No items selected for Direct Stereo.")
            return

        # Build the queue
        self.direct_stereo_queue = []
        for it in sel_items:
            dd = it.data(Qt.UserRole)
            if dd:
                self.direct_stereo_queue.append(dd)

        self.direct_stereo_idx = 0
        self.batch_cancelled = False
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setRange(0, len(self.direct_stereo_queue))

        self.log(f"[INFO] => Starting Direct Stereo on {len(self.direct_stereo_queue)} items.")
        self.process_next_direct_stereo_in_queue()

    def process_next_direct_stereo_in_queue(self):
        """
        Moves to the next video in 'self.direct_stereo_queue',
        resets phase to 0, and starts direct stereo from Depth => Splat => Inpaint.
        """
        if self.batch_cancelled:
            self.log("[INFO] => Direct Stereo => cancelled => stopping queue.")
            return
        if self.direct_stereo_idx >= len(self.direct_stereo_queue):
            self.log("[INFO] => Direct Stereo => all done!")
            return

        self.current_item_data = self.direct_stereo_queue[self.direct_stereo_idx]
        base_ = self.current_item_data.get("base_name", "")
        orig_ = self.current_item_data.get("orig", "")

        # Quick skip if no original
        if not orig_ or not os.path.isfile(orig_):
            self.log(f"[WARN] => {base_} => no original => skipping.")
            self.direct_stereo_idx += 1
            self.batch_progress_bar.setValue(self.direct_stereo_idx)
            self.process_next_direct_stereo_in_queue()
            return

        # We'll run up to 5 phases per item: [Depth1, Depth2, Merge, Splat, Inpaint]
        # depending on user selections.
        self.direct_stereo_phase = 0
        self.do_direct_stereo_phase()

    def do_direct_stereo_phase(self):
        """
        Executes the current phase (0..4) for the item in self.current_item_data.
        On success, calls on_direct_stereo_phase_finished(True).
        On fail, calls on_direct_stereo_phase_finished(False).
        """
        if self.batch_cancelled:
            self.log("[INFO] => user cancelled => do_direct_stereo_phase returning.")
            return

        base_ = self.current_item_data.get("base_name", "")
        orig_ = self.current_item_data.get("orig", "")

        depth_source_choice = self.combo_batch_depth_source.currentText()  
        # e.g. "DepthCrafter", "VDA", or "Both"
        splat_source_choice = self.combo_batch_splat_source.currentText() 
        # e.g. "depth1", "depth2", "Merged"

        # We'll track output paths here:
        self.depth1_path = self.current_item_data.get("depth1", "")
        self.depth2_path = self.current_item_data.get("depth2", "")
        self.merged_path = self.current_item_data.get("depthf", "")  # merged
        self.splatted_path = self.current_item_data.get("splatted", "")
        self.inpainted_path = self.current_item_data.get("inpainted", "")

        # Phase codes:
        # 0 => generate Depth1 (if needed)
        # 1 => generate Depth2 (if needed)
        # 2 => merge Depth1/Depth2 => DepthF (if needed)
        # 3 => splat
        # 4 => inpaint
        # If a phase is not needed, skip to the next.

        # --------------- Phase 0 => Depth1 ---------------
        if self.direct_stereo_phase == 0:
            if depth_source_choice in ["DepthCrafter", "Both"]:
                # If depth1 already exists => check overwrite
                out_path = os.path.join(self.pre_depth_edit.text(), base_ + "_depth.mp4")
                if not self.check_overwrite_depth.isChecked() and os.path.isfile(out_path):
                    # ask user
                    box = QMessageBox(self)
                    box.setWindowTitle("Overwrite Depth1?")
                    msg = f"Depth file already exists:\n{out_path}\nOverwrite?"
                    box.setText(msg)
                    yes_btn = box.addButton("Yes", QMessageBox.YesRole)
                    no_btn = box.addButton("No", QMessageBox.NoRole)
                    all_btn = box.addButton("Yes to all", QMessageBox.AcceptRole)
                    box.exec_()
                    if box.clickedButton() == no_btn:
                        self.log(f"[INFO] => skipping Depth1 for {base_}.")
                        self.direct_stereo_phase = 1  # go next
                        self.do_direct_stereo_phase()
                        return
                    elif box.clickedButton() == all_btn:
                        self.log("[INFO] => user chose 'Yes to all' => enabling Overwrite Depth checkbox.")
                        self.check_overwrite_depth.setChecked(True)

                # Build command for DepthCrafter
                cmd = [
                    sys.executable, "depth_splatting.py",
                    "--single_video", orig_,
                    "--input_depth_maps", self.pre_depth_edit.text(),
                    "--output_splatted", "./temp_splatted",
                    "--max_disp", "0",
                    "--convergence", "0",
                    "--depth_only", "True",
                    "--stereo_offset_exponent", str(self.poly_exponent),
                    "--orig_brightness_value", str(self.orig_brightness),
                    "--orig_gamma_value", str(self.orig_gamma),
                    "--max_res", str(self.crafter_maxres_value),
                    "--window_size", str(self.crafter_window),
                    "--overlap", str(self.crafter_overlap),
                    "--num_denoising_steps", str(self.crafter_denoising_steps),
                    "--guidance_scale", f"{self.crafter_guidance_scale:.1f}",
                    "--seed", str(self.crafter_seed),
                ]

                cmd.append("--warp_exponent_base")
                cmd.append(str(self.warp_exponent_base)) 
                self.current_batch_worker = SubprocessWorker(cmd)
                self.current_batch_worker.lineReady.connect(self.log)
                self.current_batch_worker.finishedSignal.connect(self.on_direct_stereo_phase_finished)
                self.current_batch_worker.start()
                return

            else:
                # not needed => skip
                self.direct_stereo_phase = 1
                self.do_direct_stereo_phase()
                return

        # --------------- Phase 1 => Depth2 ---------------
        if self.direct_stereo_phase == 1:
            if depth_source_choice in ["VDA", "Both"]:
                out_path = os.path.join(self.pre_depth2_edit.text(), base_ + "_depth.mp4")
                if not self.check_overwrite_depth.isChecked() and os.path.isfile(out_path):
                    box = QMessageBox(self)
                    box.setWindowTitle("Overwrite Depth2?")
                    msg = f"Depth2 file already exists:\n{out_path}\nOverwrite?"
                    box.setText(msg)
                    yes_btn = box.addButton("Yes", QMessageBox.YesRole)
                    no_btn = box.addButton("No", QMessageBox.NoRole)
                    all_btn = box.addButton("Yes to all", QMessageBox.AcceptRole)
                    box.exec_()
                    if box.clickedButton() == no_btn:
                        self.log(f"[INFO] => skipping Depth2 for {base_}.")
                        self.direct_stereo_phase = 2
                        self.do_direct_stereo_phase()
                        return
                    elif box.clickedButton() == all_btn:
                        self.log("[INFO] => user chose 'Yes to all' => enabling Overwrite Depth checkbox.")
                        self.check_overwrite_depth.setChecked(True)

                # Build command for VDA:
                cmd = [
                    sys.executable, "run.py",
                    "--input_video", orig_,
                    "--output_dir", self.pre_depth2_edit.text(),
                    "--max_len", "-1",
                    "--target_fps", "-1",
                    "--max_res", str(self.vda_maxres_value),
                    "--input_size", str(self.vda_input_size),
                    "--encoder", str(self.vda_encoder),
                ]
                if self.vda_use_fp16:
                    cmd.append("--use_fp16")
                if self.vda_use_cudnn:
                    cmd.append("--use_cudnn_benchmark")

              
                self.current_batch_worker = SubprocessWorker(cmd)
                self.current_batch_worker.lineReady.connect(self.log)
                self.current_batch_worker.finishedSignal.connect(self.on_direct_stereo_phase_finished)
                self.current_batch_worker.start()
                return

            else:
                self.direct_stereo_phase = 2
                self.do_direct_stereo_phase()
                return

        # --------------- Phase 2 => Merge if "Both" ---------------
        if self.direct_stereo_phase == 2:
            if depth_source_choice == "Both":
                # same logic as self.merge_depths() but in a subprocess
                out_ = os.path.join(self.pre_depth_final_edit.text(), base_ + "_depth.mp4")
                if not self.check_overwrite_depth.isChecked() and os.path.isfile(out_):
                    box = QMessageBox(self)
                    box.setWindowTitle("Overwrite Merged Depth?")
                    msg = f"Merged depth file already exists:\n{out_}\nOverwrite?"
                    box.setText(msg)
                    yes_btn = box.addButton("Yes", QMessageBox.YesRole)
                    no_btn = box.addButton("No", QMessageBox.NoRole)
                    all_btn = box.addButton("Yes to all", QMessageBox.AcceptRole)
                    box.exec_()
                    if box.clickedButton() == no_btn:
                        self.log(f"[INFO] => skipping Merge for {base_}.")
                        # no merged => we continue
                        self.direct_stereo_phase = 3
                        self.do_direct_stereo_phase()
                        return
                    elif box.clickedButton() == all_btn:
                        self.check_overwrite_depth.setChecked(True)

                depth1_ = os.path.join(self.pre_depth_edit.text(), base_ + "_depth.mp4")
                depth2_ = os.path.join(self.pre_depth2_edit.text(), base_ + "_depth.mp4")
                cmd = [
                    sys.executable, "fusion_depths.py",
                    "--depth1", depth1_,
                    "--depth2", depth2_,
                    "--alpha", str(self.fusion_alpha/100.0),
                    "--output", out_,
                    "--brightness_value", str(self.brightness_value),
                    "--gamma_value", str(self.gamma_value),
                    "--dilate_h", str(self.dilate_h_value),
                    "--dilate_v", str(self.dilate_v_value),
                    "--blur_ksize", str(self.blur_ksize_value),
                    "--blur_sigma", str(self.blur_sigma_value)
                ]
               
                self.current_batch_worker = SubprocessWorker(cmd)
                self.current_batch_worker.lineReady.connect(self.log)
                self.current_batch_worker.finishedSignal.connect(self.on_direct_stereo_phase_finished)
                self.current_batch_worker.start()
                return
            else:
                self.direct_stereo_phase = 3
                self.do_direct_stereo_phase()
                return

        # --------------- Phase 3 => Splat ---------------
        if self.direct_stereo_phase == 3:
            # same approach as process_next_splat_in_queue
            spl_src = self.combo_batch_splat_source.currentText()  # "depth1","depth2","Merged"
            out_path = os.path.join(self.output_splattings_edit.text(), base_ + "_splatted.mp4")
            if not self.check_overwrite_splat.isChecked() and os.path.isfile(out_path):
                box = QMessageBox(self)
                box.setWindowTitle("Overwrite Splat?")
                msg = f"File already exists:\n{out_path}\nOverwrite?"
                box.setText(msg)
                yes_btn = box.addButton("Yes", QMessageBox.YesRole)
                no_btn  = box.addButton("No", QMessageBox.NoRole)
                all_btn = box.addButton("Yes to all", QMessageBox.AcceptRole)
                box.exec_()
                if box.clickedButton() == no_btn:
                    self.log(f"[INFO] => skipping splat for {base_}.")
                    self.direct_stereo_phase = 4
                    self.do_direct_stereo_phase()
                    return
                elif box.clickedButton() == all_btn:
                    self.check_overwrite_splat.setChecked(True)

            # pick depth path:
            if spl_src == "depth1":
                dep = os.path.join(self.pre_depth_edit.text(), base_ + "_depth.mp4")
            elif spl_src == "depth2":
                dep = os.path.join(self.pre_depth2_edit.text(), base_ + "_depth.mp4")
            else:
                # "Merged"
                dep = os.path.join(self.pre_depth_final_edit.text(), base_ + "_depth.mp4")

            if not dep or not os.path.isfile(dep):
                self.log(f"[WARN] => {base_} => no depth => skipping splat.")
                self.direct_stereo_phase = 4
                self.do_direct_stereo_phase()
                return

            # check for keyframes
            keyf_json_path = ""
            if base_ in self.video_params and "keyframes" in self.video_params[base_]:
                kf_dict = self.video_params[base_]["keyframes"]
                if kf_dict:
                    real_dict = {}
                    for k,v in kf_dict.items():
                        try:
                            key_int = int(k)
                            real_dict[key_int] = v
                        except:
                            pass
                    if real_dict:
                        keyf_json_path = f"temp_keyframes_{base_}.json"
                        try:
                            with open(keyf_json_path, "w", encoding="utf-8") as ff:
                                json.dump({base_: real_dict}, ff, indent=2)
                            self.log(f"[INFO] => Keyframes saved => {keyf_json_path}")
                        except Exception as e:
                            self.log(f"[ERROR] => Could not save keyframes => {e}")
                            keyf_json_path = ""

            cmd = [
                sys.executable, "depth_splatting.py",
                "--single_video", self.current_item_data["orig"],
                "--input_depth_maps", os.path.dirname(dep),
                "--output_splatted", self.output_splattings_edit.text(),
                "--max_disp", str(self.disp_value),
                "--convergence", str(self.conv_value),
                "--depth_brightness_value", str(self.brightness_value),
                "--depth_gamma_value", str(self.gamma_value),
                "--dilate_h", str(self.dilate_h_value),
                "--stereo_offset_exponent", str(self.poly_exponent),
                "--dilate_v", str(self.dilate_v_value),
                "--blur_ksize", str(self.blur_ksize_value),
                "--blur_sigma", str(self.blur_sigma_value),
                "--process_length", "-1",
                "--max_res", str(self.depth_maxres_value),
                "--window_size", str(self.crafter_window),
                "--overlap", str(self.crafter_overlap),
                "--enable_interpolation", "True",
                "--num_denoising_steps", str(self.crafter_denoising_steps),
                "--guidance_scale", f"{self.crafter_guidance_scale:.1f}",
                "--seed", str(self.crafter_seed),
            ]
            
            cmd.append("--warp_exponent_base")
            cmd.append(str(self.warp_exponent_base)) 
            if keyf_json_path:
                cmd.append("--keyframes_json")
                cmd.append(keyf_json_path)

            
            cmd.append("--warp_exponent_base")
            cmd.append(str(self.warp_exponent_base)) 
            self.current_batch_worker = SubprocessWorker(cmd)
            self.current_batch_worker.lineReady.connect(self.log)
            self.current_batch_worker.finishedSignal.connect(self.on_direct_stereo_phase_finished)
            self.current_batch_worker.start()
            return

        # --------------- Phase 4 => Inpaint ---------------
        if self.direct_stereo_phase == 4:
            splatted_ = os.path.join(self.output_splattings_edit.text(), base_ + "_splatted.mp4")
            if not splatted_ or not os.path.isfile(splatted_):
                self.log(f"[WARN] => {base_} => no splatted => skipping inpaint.")
                self.end_direct_stereo_item()
                return

            out_name = f"{base_}_splatted_inpainting_results_{self.sbs_mode}_{self.output_encoder}.mp4"
            out_path = os.path.join(self.inpainting_out_edit.text(), out_name)
            if not self.check_overwrite_inpaint.isChecked() and os.path.isfile(out_path):
                box = QMessageBox(self)
                box.setWindowTitle("Overwrite Inpaint?")
                msg = f"Inpaint file already exists:\n{out_path}\nOverwrite?"
                box.setText(msg)
                yes_btn = box.addButton("Yes", QMessageBox.YesRole)
                no_btn  = box.addButton("No",  QMessageBox.NoRole)
                all_btn = box.addButton("Yes to all", QMessageBox.AcceptRole)
                box.exec_()
                if box.clickedButton() == no_btn:
                    self.log(f"[INFO] => skipping inpaint for {base_}.")
                    self.end_direct_stereo_item()
                    return
                elif box.clickedButton() == all_btn:
                    self.check_overwrite_inpaint.setChecked(True)

            orig_ = self.current_item_data.get("orig","")

            cmd = [
                sys.executable, "inpainting.py",
                "--single_video", splatted_,
                "--output_folder", self.inpainting_out_edit.text(),
                "--threshold_mask", f"{self.inpaint_threshold_batch:.3f}",
                "--orig_video", orig_,
                "--num_inference_steps", str(self.inpaint_num_inference_steps_batch),
                f"--tile_num={self.inpaint_tile_num_batch}",
                "--sbs_mode", self.sbs_mode,
                "--encoder", self.output_encoder
            ]
            
      
            cmd.append("--overlap")
            cmd.append(str(self.inpaint_overlap_batch))

        
            self.current_batch_worker = SubprocessWorker(cmd)
            self.current_batch_worker.lineReady.connect(self.log)
            self.current_batch_worker.finishedSignal.connect(self.on_direct_stereo_phase_finished)
            self.current_batch_worker.start()
            return

    def on_direct_stereo_phase_finished(self, success: bool):
        """
        Called when the current phase’s SubprocessWorker finishes.
        If success => move to the next phase.
        Otherwise => skip the rest for this item.
        """
        self.current_batch_worker = None
        if not success:
            self.log("[ERROR] => Phase failed => skipping the rest for this item.")
            self.end_direct_stereo_item()
            return

        # success => go next phase
        self.direct_stereo_phase += 1
        if self.direct_stereo_phase > 4:
            # done all phases
            self.end_direct_stereo_item()
        else:
            # proceed
            self.do_direct_stereo_phase()
               
        
    def end_direct_stereo_item(self):
        self.direct_stereo_idx += 1
        self.batch_progress_bar.setValue(self.direct_stereo_idx)
        self.process_next_direct_stereo_in_queue()
            
    def cancel_current_batch(self):
        self.batch_cancelled = True
        if self.current_batch_worker is not None:
            self.log("[INFO] => Canceling current batch worker...")
            self.current_batch_worker.request_cancel()
        else:
            QMessageBox.information(self, "Info", "No batch worker is running currently.")
                
            

    ###########################################################
    # MERGE SELECTED
    ###########################################################
    def merge_selected_videos(self):
        sel_items= self.list_widget.selectedItems()
        if not sel_items:
            QMessageBox.information(self,"Info","No items to merge.")
            return
        merged_dir= self.completed_merged_edit.text().strip()
        if not os.path.exists(merged_dir):
            os.makedirs(merged_dir,exist_ok=True)

        source_choice= self.combo_merge_source.currentText()
        choice_to_key={
            "ORIG":"orig",
            "DEPTH1":"depth1",
            "DEPTH2":"depth2",
            "MERGED_DEPTH":"depthf",
            "SPLATTED":"splatted",
            "INPAINTED":"inpainted"
        }
        chosen_key= choice_to_key.get(source_choice,"orig")

        merge_list=[]
        for it in sel_items:
            dd= it.data(Qt.UserRole)
            if not dd: continue
            base_= dd.get("base_name","")
            path_= dd.get(chosen_key,"")
            if path_ and os.path.isfile(path_):
                merge_list.append((base_,path_))
        if not merge_list:
            QMessageBox.information(self,"Info",f"No valid {source_choice} in selected items.")
            return
        merge_list.sort(key=lambda x: x[0])

        temp_txt=os.path.join(merged_dir,"temp_merge_list.txt")
        try:
            with open(temp_txt,"w",encoding="utf-8") as f:
                for _,p_ in merge_list:
                    f.write(f"file '{os.path.abspath(p_)}'\n")
        except Exception as e:
            self.log("[ERROR] => can't create list => "+str(e))
            return

        out_name=f"merged_{len(merge_list)}_videos_{source_choice.lower()}.mp4"
        out_path=os.path.join(merged_dir,out_name)
        if os.path.isfile(out_path):
            resp= QMessageBox.question(self,"File exist",
                                       f"{out_path} exist. Overwrite?",
                                       QMessageBox.Yes|QMessageBox.No,
                                       QMessageBox.No)
            if resp== QMessageBox.No:
                return
        cmd= [
            "ffmpeg","-y","-f","concat","-safe","0",
            "-i", temp_txt,
            "-c","copy",
            out_path
        ]
        self.log("[INFO] => merging => "+" ".join(cmd))
        self.merge_videos_worker= SubprocessWorker(cmd)
        self.current_batch_worker=self.merge_videos_worker
        self.batch_cancelled = False
        self.merge_videos_worker.lineReady.connect(self.log)
        self.merge_videos_worker.finishedSignal.connect(lambda s:self.on_merge_videos_finished(s,out_path))
        self.merge_videos_worker.start()

    def on_merge_videos_finished(self, success, out_path):
        self.current_batch_worker=None
        if success:
            self.log(f"[OK] => merged => {out_path}")
            QMessageBox.information(self,"Merge done",f"Merged => {out_path}")
        else:
            self.log("[ERROR] => merge fail.")

    ###########################################################
    # UTILS
    ###########################################################
    def open_folder(self, path_dir):
        path_abs= os.path.abspath(path_dir)
        if not os.path.exists(path_abs):
            os.makedirs(path_abs, exist_ok=True)
        import platform
        pf= platform.system()
        try:
            if pf=="Windows":
                os.startfile(path_abs)
            elif pf=="Darwin":
                subprocess.Popen(["open", path_abs])
            else:
                subprocess.Popen(["xdg-open", path_abs])
        except Exception as e:
            QMessageBox.warning(self,"Attention", f"Could not open folder:\n{path_abs}\nError: {e}")

    def browse_and_set_folder(self, line_edit):
        selected_dir = QFileDialog.getExistingDirectory(self, "Select a directory")
        if selected_dir:
            line_edit.setText(selected_dir)
            self.save_paths()
            self.scan_and_list_videos()

    def log(self, *args):
        # Unir todo en una sola cadena
        line = " ".join(str(a) for a in args)
        self.log_box.appendPlainText(line)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )
        print(line)

    def store_current_params(self):
        item= self.list_widget.currentItem()
        if not item: return
        dd= item.data(Qt.UserRole)
        if not dd: return
        base_= dd.get("base_name","")
        if not base_: return
        old_kf= self.video_params.get(base_,{}).get("keyframes",{})
        self.video_params[base_]={
            "disp_value": self.disp_value,
            "conv_value": self.conv_value,
            "brightness_value": self.brightness_value,
            "gamma_value": self.gamma_value,
            "dilate_h_value": self.dilate_h_value,
            "dilate_v_value": self.dilate_v_value,
            "blur_ksize_value": self.blur_ksize_value,
            "blur_sigma_value": self.blur_sigma_value,
            "warp_exponent_base": self.warp_exponent_base,
            "keyframes": old_kf
        }
        self.save_params()
        

 
    def load_new_video_to_input(self):
        # 1) Seleccionar path
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a video",
            "",
            "Videos (*.mp4 *.mov *.avi *.mkv)"
        )
        if not path:
            return  # Usuario canceló el diálogo

        bn = os.path.basename(path)
        target = os.path.join(self.input_videos_edit.text(), bn)

        from PyQt5.QtWidgets import QMessageBox

        # Si ya existe => preguntar overwrite
        if os.path.isfile(target):
            resp = QMessageBox.question(
                self,
                "File Already Exists",
                f"File {bn} already in input folder. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if resp == QMessageBox.No:
                return

        # 2) Verificar frames con decord
        from decord import VideoReader, cpu
        try:
            vr = VideoReader(path, ctx=cpu(0))
            n_frames = len(vr)
            self.log(f"[INFO] => The video '{bn}' has {n_frames} frames.")
        except Exception as e:
            QMessageBox.warning(self, "Attention", f"Could not open the video:\n{e}")
            self.log(f"[ERROR] => opening {path} => {e}")
            return

        # 3) Si > 400 frames => advertir
        if n_frames > 400:
            from PyQt5.QtWidgets import QMessageBox, QInputDialog
            import PyQt5.QtWidgets as QtW

            msg = (
                f"You have selected a video with {n_frames} frames.\n\n"
                "StereoMaster is NOT optimized for long videos.\n"
                "It is strongly recommended to SPLIT it using Scene Detect.\n\n"
                "Do you want to run scene detect now?\n"
                " - YES => enter threshold + do scene detect (no copy)\n"
                " - NO => proceed anyway (copy to input)\n"
                " - CANCEL => do nothing"
            )
            r2 = QtW.QMessageBox.warning(
                self,
                "Large Video Imported",
                msg,
                QtW.QMessageBox.Yes | QtW.QMessageBox.No | QtW.QMessageBox.Cancel,
                QtW.QMessageBox.Yes
            )
            if r2 == QtW.QMessageBox.Yes:
                # => Pedir threshold => Hacer scene detect => RETORNAR (no copiamos)
                thresh, ok = QInputDialog.getInt(
                    self,
                    "Scene Threshold",
                    "Enter scene detection threshold [15..30 recommended]:",
                    35, 1, 999, 1
                )
                if ok:
                    # Llamamos do_scene_detect con path
                    self.log("[INFO] => launching scene detect w/ threshold => " + str(thresh))
                    self.do_scene_detect(threshold=str(thresh), video_path=path)
                else:
                    self.log("[INFO] => Scene detect threshold cancelled by user.")
                return  

            elif r2 == QtW.QMessageBox.No:
                # => “proceed anyway” => Hacemos la copia
                self.log("[WARN] => proceeding with large video at user's own risk.")
                # Continuamos abajo
            else:
                # CANCEL => no copiamos, no hacemos nada
                self.log("[INFO] => user cancelled importing video.")
                return

        # 4) Hacer la copia (solo si no se salió antes)
        try:
            self.log(f"[INFO] => Copying '{path}' => '{target}' with progress bar...")

            self.batch_progress_bar.setValue(0)
            self.batch_progress_bar.setMaximum(100)
            self.batch_progress_bar.show()

            src_size = os.path.getsize(path)
            copied_bytes = 0
            chunk_size = 1024 * 1024  # 1MB

            with open(path, "rb") as fsrc, open(target, "wb") as fdst:
                while True:
                    chunk = fsrc.read(chunk_size)
                    if not chunk:
                        break
                    fdst.write(chunk)
                    copied_bytes += len(chunk)
                    percent = int((copied_bytes / src_size) * 100)
                    self.batch_progress_bar.setValue(percent)
                    QApplication.processEvents()

            self.batch_progress_bar.setValue(100)
            self.log(f"[INFO] => Copied => {target}")

        except Exception as e:
            QMessageBox.warning(self, "Attention", f"Could not copy:\n{e}")
            self.log(f"[ERROR] => copying {path} => {e}")
            return

        # 5) Al final => refrescar
        self.scan_and_list_videos()




    ###########################################################
    # SLIDERS => store + updates
    ###########################################################
    def on_disp_slider_changed(self, value):
        self.disp_value= float(value)
        self.lbl_disp_val.setText(str(value))
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def on_conv_slider_changed(self, value):
        self.conv_value= float(value)
        self.lbl_conv_val.setText(str(value))
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def on_brightness_slider_changed(self, value):
        vf= value/100.0
        self.brightness_value= vf
        self.lbl_bri_val.setText(f"{vf:.2f}")
        self.store_current_params()
        self.show_preview(self.current_frame_idx)
        
    def on_poly_exp_changed(self, value):
        # slider de 1..300 => factor= value/100 => 0.01..3.0
        factor = value / 100.0
        self.poly_exponent = factor
        self.lbl_poly_exp_val.setText(f"{factor:.2f}")

        # guardamos en la current params
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def on_poly_fill_changed(self, idx):
        fill_str = self.combo_poly_fill.currentText()
        self.poly_fill_tech = fill_str
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def reset_poly_exp(self):
        self.poly_exponent = 1.0
        self.slider_poly_exp.setValue(100)  # 1.00
        self.lbl_poly_exp_val.setText("1.00")
        self.store_current_params()
        self.show_preview(self.current_frame_idx)


    def on_gamma_slider_changed(self, value):
        vf= value/100.0
        if vf<0.01: vf=0.01
        self.gamma_value= vf
        self.lbl_gam_val.setText(f"{vf:.2f}")
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def on_dilate_h_changed(self, value):
        self.dilate_h_value= int(value)
        self.lbl_dh_val.setText(str(value))
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def on_dilate_v_changed(self, value):
        self.dilate_v_value= int(value)
        self.lbl_dv_val.setText(str(value))
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def on_blur_ksize_changed(self, value):
        if value>0 and value%2==0:
            value+=1
        self.blur_ksize_value= value
        self.slider_blur_ksize.setValue(value)
        self.lbl_bk_val.setText(str(value))
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def on_blur_sigma_changed(self, value):
        vf= value/10.0
        self.blur_sigma_value= vf
        self.lbl_bs_val.setText(f"{vf:.2f}")
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def on_orig_brightness_changed(self, value):
        vf= value/100.0
        self.orig_brightness= vf
        self.lbl_orig_bri_val.setText(str(vf))
        self.show_preview(self.current_frame_idx)

    def on_orig_gamma_changed(self, value):
        vf= value/100.0
        if vf<0.01: vf=0.01
        self.orig_gamma= vf
        self.lbl_orig_gam_val.setText(str(vf))
        self.show_preview(self.current_frame_idx)

    def on_maxres_changed(self, index):
        val_str = self.max_res_combo.currentText()
        val_i   = int(val_str)
        self.crafter_maxres_value = val_i  
        self.save_depth_global_params()
        if val_i >= 768:
            QMessageBox.warning(
                self,
                "High resolution!",
                "Warning: from 768 and above, you may need ~16GB VRAM.\nSystem could freeze or crash."
            )
        self.log(f"[INFO] => DepthCrafter max_res => {val_i}")
        
    def on_inpaint_steps_single_changed(self, val):
        self.inpaint_num_inference_steps_single = val
        self.log(f"[INFO] => Single Inpaint num_inference_steps = {val}")
        
    def open_kofi_link(self):
        QDesktopServices.openUrl(QUrl("https://ko-fi.com/3dultraenhancer"))    
        
    def on_inpaint_steps_batch_changed(self, val):
        self.inpaint_num_inference_steps_batch = val
        self.log(f"[INFO] => Batch Inpaint num_inference_steps = {val}")
    
    ###########################################################
    # RESETS
    ###########################################################
    def reset_disp(self):
        self.disp_value=20.0
        self.slider_disp.setValue(20)
        self.lbl_disp_val.setText("20")
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def reset_conv(self):
        self.conv_value=0.0
        self.slider_conv.setValue(0)
        self.lbl_conv_val.setText("0")
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def reset_brightness_depth(self):
        self.brightness_value=1.0
        self.slider_brightness.setValue(100)
        self.lbl_bri_val.setText("1.0")
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def reset_gamma_depth(self):
        self.gamma_value=1.0
        self.slider_gamma.setValue(100)
        self.lbl_gam_val.setText("1.0")
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def reset_dilate_h(self):
        self.dilate_h_value=4
        self.slider_dilate_h.setValue(4)
        self.lbl_dh_val.setText("4")
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def reset_dilate_v(self):
        self.dilate_v_value=1
        self.slider_dilate_v.setValue(1)
        self.lbl_dv_val.setText("1")
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def reset_blur_ksize(self):
        self.blur_ksize_value=3
        self.slider_blur_ksize.setValue(3)
        self.lbl_bk_val.setText("3")
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def reset_blur_sigma(self):
        self.blur_sigma_value=2.0
        self.slider_blur_sigma.setValue(int(2.0*10))
        self.lbl_bs_val.setText("2.0")
        self.store_current_params()
        self.show_preview(self.current_frame_idx)

    def reset_orig_bri(self):
        self.orig_brightness=1.0
        self.slider_orig_bri.setValue(100)
        self.lbl_orig_bri_val.setText("1.0")
        self.show_preview(self.current_frame_idx)

    def reset_orig_gamma(self):
        self.orig_gamma=1.0
        self.slider_orig_gam.setValue(100)
        self.lbl_orig_gam_val.setText("1.0")
        self.show_preview(self.current_frame_idx)
        



def main():
    app= QApplication(sys.argv)
    gui= StereoMasterGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()
