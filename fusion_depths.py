import argparse
import os
import sys
import cv2
import numpy as np
from decord import VideoReader, cpu


def apply_depth_preprocess(depth_frame_01, 
                           brightness=1.0, gamma=1.0,
                           dilate_h=0, dilate_v=0,
                           blur_ksize=0, blur_sigma=0.0):


    # 1) brightness/gamma
    depth_ = np.clip(depth_frame_01 * brightness, 0, 1)
    depth_ = depth_ ** (1.0 / gamma)

    # 2) dilate
    if dilate_h>0 or dilate_v>0:
        depth_u8 = (depth_*255).astype(np.uint8)
        kernel = np.ones((dilate_v,dilate_h), np.uint8)
        depth_u8 = cv2.dilate(depth_u8, kernel, iterations=1)
        depth_ = depth_u8.astype(np.float32)/255.0

    # 3) blur
    if blur_ksize>0:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        depth_255 = (depth_*255).astype(np.float32)
        depth_255 = cv2.GaussianBlur(depth_255, (blur_ksize,blur_ksize), sigmaX=blur_sigma)
        depth_ = np.clip(depth_255 / 255.0, 0,1)
    return depth_


def main(depth1, depth2, alpha=0.5,
         output="merged_depth.mp4",
         brightness_value=1.0,
         gamma_value=1.0,
         dilate_h=0,
         dilate_v=0,
         blur_ksize=0,
         blur_sigma=0.0

         ):
    """
    fusion_depths.py 
    USO:
      python fusion_depths.py --depth1=path1.mp4 --depth2=path2.mp4 --alpha=0.4 
                              --output=merged_depth.mp4
                              --brightness_value=1.0
                              --gamma_value=1.0
                              --dilate_h=4
                              --dilate_v=1
                              --blur_ksize=3
                              --blur_sigma=2.0

    """
    # 1) Abrir con decord
    vr1 = VideoReader(depth1, ctx=cpu(0))
    vr2 = VideoReader(depth2, ctx=cpu(0))

    # 2) nframes
    n1 = len(vr1)
    n2 = len(vr2)
    nmin = min(n1, n2)
    print(f"[INFO] => Depth1={depth1} frames={n1}, Depth2={depth2} frames={n2}, using nmin={nmin}")
    if nmin==0:
        print("[WARN] => one video has 0 frames => no output.")
        return

    # 3) fps => supón que tomamos la del primer video
    fps1 = vr1.get_avg_fps()
    fps2 = vr2.get_avg_fps()
    if fps2<1: 
        fps2= fps1
    fps_out= fps1 if fps1>0 else 25.0

    # 4) Leer shape
    fr0 = vr1[0].asnumpy()
    H,W = fr0.shape[:2]
    # supondremos que depth2 coincide en resolución...
    # De lo contrario, habría que reescalar.

    # 5) Preparar VideoWriter
    out_dir = os.path.dirname(output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(output, fourcc, fps_out, (W, H))
    if not vw.isOpened():
        print(f"[ERROR] => Could not open VideoWriter => {output}")
        return


    # 6) Fusionar
    a_ = float(alpha)  # 0..1
    for i in range(nmin):
        d1_ = vr1[i].asnumpy().astype(np.float32)/255.0
        d2_ = vr2[i].asnumpy().astype(np.float32)/255.0

        # Convertir a 1 canal
        if d1_.shape[-1]==3:
            d1_ = d1_.mean(axis=-1)
        if d2_.shape[-1]==3:
            d2_ = d2_.mean(axis=-1)

        # Normalizar c/u
        mn1, mx1 = d1_.min(), d1_.max()
        if (mx1-mn1)>1e-6:
            d1_ = (d1_-mn1)/(mx1-mn1)
        mn2, mx2 = d2_.min(), d2_.max()
        if (mx2-mn2)>1e-6:
            d2_ = (d2_-mn2)/(mx2-mn2)

        # 7) apply DepthPreprocess => brightness_value, gamma_value, ...
        d1_ = apply_depth_preprocess(d1_,
            brightness=brightness_value,
            gamma=gamma_value,
            dilate_h=dilate_h,
            dilate_v=dilate_v,
            blur_ksize=blur_ksize,
            blur_sigma=blur_sigma
        )
        d2_ = apply_depth_preprocess(d2_,
            brightness=brightness_value,
            gamma=gamma_value,
            dilate_h=dilate_h,
            dilate_v=dilate_v,
            blur_ksize=blur_ksize,
            blur_sigma=blur_sigma
        )

        # 8) alpha blend => dmerge = d1_ * (1-a_) + d2_ * a_
        dmerge = d1_*(1.0 - a_) + d2_*(a_)

        # 9) Guardar => .mp4 (grayscale en 3 canales)
        d255 = np.clip(dmerge*255, 0,255).astype(np.uint8)
        d3 = cv2.cvtColor(d255, cv2.COLOR_GRAY2BGR)
        vw.write(d3)
    
    vw.release()
    print(f"[OK] => saved => {output}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Fusion of two depth videos (alpha + brightness/gamma/dilate/blur).")
    parser.add_argument("--depth1", type=str, required=True)
    parser.add_argument("--depth2", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="merged_depth.mp4")

    parser.add_argument("--brightness_value", type=float, default=1.0)
    parser.add_argument("--gamma_value", type=float, default=1.0)
    parser.add_argument("--dilate_h", type=int, default=0)
    parser.add_argument("--dilate_v", type=int, default=0)
    parser.add_argument("--blur_ksize", type=int, default=0)
    parser.add_argument("--blur_sigma", type=float, default=0.0)

    args = parser.parse_args()

    main(depth1=args.depth1,
         depth2=args.depth2,
         alpha=args.alpha,
         output=args.output,
         brightness_value=args.brightness_value,
         gamma_value=args.gamma_value,
         dilate_h=args.dilate_h,
         dilate_v=args.dilate_v,
         blur_ksize=args.blur_ksize,
         blur_sigma=args.blur_sigma,
         window_size=args.window_size,
         overlap=args.overlap)
