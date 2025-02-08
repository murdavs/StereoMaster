import torch
import torch.nn.functional as F
from torch import nn

# Importa tu forward_warp original:
# Ajusta la ruta si tu "forward_warp" está en otro lugar
from .forward_warp import forward_warp

class forward_warp_padded(nn.Module):
    """
    Este wrapper añade padding a la imagen y al flow antes de llamar a forward_warp
    para minimizar la pérdida de pixeles en los bordes.
    """
    def __init__(self, pad=20, interpolation_mode="Bilinear"):
        """
        Args:
            pad (int): Número de pixeles de padding en cada borde.
            interpolation_mode (str): "Bilinear" o "Nearest", igual que en forward_warp.
        """
        super().__init__()
        self.pad = pad
        # Reutiliza tu forward_warp original:
        self.forward_warp = forward_warp(interpolation_mode=interpolation_mode)

    def forward(self, im0, flow):
        """
        im0:  [B, C, H, W]
        flow: [B, H, W, 2]
        """
        B, C, H, W = im0.shape
        p = self.pad

        # 1) Añadimos padding a la imagen de entrada
        #    Utiliza 'reflect' o 'replicate' en lugar de 'constant' para evitar bordes negros
        im0_pad = F.pad(im0, (p, p, p, p), mode='reflect')

        # 2) Preparamos el flow para aplicar el mismo padding
        #    Reordenamos a [B, 2, H, W] para usar F.pad fácilmente.
        flow_pad = flow.permute(0, 3, 1, 2)  # [B, 2, H, W]
        flow_pad = F.pad(flow_pad, (p, p, p, p), mode='constant', value=0)

        # 3) Ajustamos el offset de flow, porque hemos desplazado la imagen p píxeles
        flow_pad[:, 0, :, :] += p  # Eje X
        flow_pad[:, 1, :, :] += p  # Eje Y

        # 4) Regresamos el flow a [B, H, W, 2]
        flow_pad = flow_pad.permute(0, 2, 3, 1)

        # 5) Llamamos a tu forward_warp original con la imagen y flow padded
        warped_pad = self.forward_warp(im0_pad, flow_pad)

        # 6) Recortamos al tamaño original
        #    Eliminamos los p pixeles agregados en cada borde
        warped = warped_pad[:, :, p:-p, p:-p]

        return warped
