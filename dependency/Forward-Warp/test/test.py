import cv2
import time
import torch
import pickle
import numpy as np

from Forward_Warp import forward_warp, forward_warp_rescaled


def get_mask(img, flow):
    ones = torch.ones_like(im0)
    mask = fw(ones, flow)
    return mask


def get_color_mask(mask, value=1, margin=0.1):

    def get_map(mask, value, margin, bigger_than=False, smaller_than=False):
        assert bool(smaller_than) or bool(bigger_than)  # select either
        mask1 = mask[:,0,:,:]
        map1 = torch.zeros_like(mask1)
        if smaller_than and bigger_than:
            map1[mask1 < value + margin] += 1
            map1[mask1 > value - margin] += 1
            map1 -= 1
            map1[map1 < 0] = 0
        elif smaller_than:
            map1[mask1 < value - margin] = 1
        else: # bigger_than
            map1[mask1 > value + margin] = 1
        return map1
    
    mask_colored = torch.zeros_like(mask)
    mask_colored[:,0,:,:] = get_map(mask, value, margin, smaller_than=True)
    mask_colored[:,1,:,:] = get_map(mask, value, margin, smaller_than=True, bigger_than=True)
    map_bigger = get_map(mask, value, margin, bigger_than=True)
    mask_colored[:,2,:,:] = map_bigger
    count_bigger = map_bigger.sum().cpu().detach().numpy()
    count_pixels = map_bigger.numel()
    print(margin, count_bigger / count_pixels, count_bigger, count_pixels)

    return mask_colored


def log_image(image, name):
    image = image.clone().permute(0, 2, 3, 1)[0]
    cv2.imwrite("{}.png".format(name), image.cpu().detach().numpy().astype(np.uint8))
    print("{}: min {}, max {}, mean {}".format(name, image.min(), image.max(), image.mean()))


if __name__ == "__main__":

    im0 = cv2.imread("im0.png")[np.newaxis, :, :, :]
    im1 = cv2.imread("im1.png")[np.newaxis, :, :, :]
    with open("flow.pkl", "rb+") as f:
        flow = pickle.load(f)
    im0 = torch.tensor(im0, dtype=torch.float32, requires_grad=True).permute(0, 3, 1, 2)
    im1 = torch.tensor(im1, dtype=torch.float32, requires_grad=True).permute(0, 3, 1, 2)
    flow = torch.tensor(flow, dtype=torch.float32, requires_grad=True)

    fw = forward_warp()
    fw_rescaled = forward_warp_rescaled()

    # since = time.time()
    # im1_python = fw(im0, flow)
    # print("python version forward cost time: {}".format(time.time()-since))

    im0 = im0.cuda()
    flow = flow.cuda()
    torch.cuda.synchronize()
    since = time.time()
    im1_cuda = fw(im0, flow)
    torch.cuda.synchronize()
    print("cuda version forward cost time: {}".format(time.time()-since))
    
    torch.cuda.synchronize()
    since = time.time()
    im1_cuda_rescaled = fw_rescaled(im0, flow)
    torch.cuda.synchronize()
    print("cuda rescaled version forward cost time: {}".format(time.time()-since))
    
    loss_fn = torch.nn.MSELoss()
    # python_loss = loss_fn(im1_python, im1)
    # print("python loss: {}".format(python_loss))
    cuda_loss = loss_fn(im1_cuda, im1.cuda())
    print("cuda loss: {}".format(cuda_loss))
    cuda_rescaled_loss = loss_fn(im1_cuda_rescaled, im1.cuda())
    print("cuda rescaled loss: {}".format(cuda_rescaled_loss))

    previous_loss = loss_fn(im0.cuda(), im1.cuda())
    print("previous loss: {}".format(previous_loss))

    # log_image(im1_python, "im1_python")
    log_image(im1_cuda, "im1_cuda")
    log_image(im1_cuda_rescaled, "im1_cuda_rescaled")
    
    mask_cuda = get_mask(im0, flow)
    for margin in [0.5, 0.1, 0.01, 0.001]:
        mask_colored = get_color_mask(mask_cuda, margin=margin)
        log_image(mask_colored * 255, "mask_colored_{}".format(str(margin)))
    # margin 0.5 or 0.1 should be good

    # test backward
    im1_cuda.mean().backward()
