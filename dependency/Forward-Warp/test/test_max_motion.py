import torch


def test_forward_warp_max_motion_forward():
    from Forward_Warp import forward_warp_max_motion

    B, H, W = 2, 5, 10

    img = torch.ones(B, 3, H, W).cuda()
    flow = torch.zeros(B, H, W, 2).cuda()

    fw = forward_warp_max_motion()

    warped, disocclusions, im1_buffer, d_buffer, wght_buffer = fw(img, flow, debug=True)

    assert warped.shape == img.shape, f"{warped.shape, img.shape}"
    assert torch.allclose(disocclusions, torch.zeros(B, 1, H, W).cuda()), f"{disocclusions}"
    assert torch.allclose(im1_buffer, torch.ones(B, 3, H, W).cuda()), f"{im1_buffer}"
    assert torch.allclose(d_buffer, torch.zeros(B, 1, H, W, dtype=torch.int32).cuda()), f"{d_buffer}"
    assert torch.allclose(wght_buffer, torch.ones(B, 1, H, W).cuda()), f"{wght_buffer}"
    assert torch.allclose(warped, torch.ones(B, 3, H, W).cuda()), f"{warped}"

    flow[0,0,0] = 1

    warped, disocclusions, im1_buffer, d_buffer, wght_buffer = fw(img, flow, debug=True)

    disocclusions_target = torch.zeros(B, 1, H, W).cuda()
    disocclusions_target[0,0,0,0] = 1
    im1_buffer_target = torch.ones(B, 3, H, W).cuda()
    im1_buffer_target[0,:,0,0] = 0
    d_buffer_target = torch.zeros(B, 1, H, W, dtype=torch.int32).cuda()
    d_buffer_target[0,0,1,1] = 141
    wght_buffer_target = torch.ones(B, 1, H, W).cuda()
    wght_buffer_target[0,0,0,0] = 0
    warped_target = torch.ones(B, 3, H, W).cuda()
    warped_target[0,:,0,0] = 0

    assert warped.shape == img.shape, f"{warped.shape, img.shape}"
    assert torch.allclose(disocclusions, disocclusions_target), f"{disocclusions, disocclusions_target}"
    assert torch.allclose(im1_buffer, im1_buffer_target), f"{im1_buffer, im1_buffer_target}"
    assert torch.allclose(d_buffer, d_buffer_target), f"{d_buffer, d_buffer_target}"
    assert torch.allclose(wght_buffer, wght_buffer_target), f"{wght_buffer, wght_buffer_target}"
    assert torch.allclose(warped, warped_target), f"{warped, warped_target}"

    flow[0,0,0] = 0.5

    warped, disocclusions, im1_buffer, d_buffer, wght_buffer = fw(img, flow, debug=True)

    im1_buffer_target = torch.ones(B, 3, H, W).cuda()
    im1_buffer_target[0,:,:2,:2] = 0.25
    d_buffer_target = torch.zeros(B, 1, H, W, dtype=torch.int32).cuda()
    d_buffer_target[0,0,:2,:2] = 70
    wght_buffer_target = torch.ones(B, 1, H, W).cuda()
    wght_buffer_target[0,0,:2,:2] = 0.25  # gets spread equally onto 4 neighbors

    assert warped.shape == img.shape, f"{warped.shape, img.shape}"
    assert torch.allclose(disocclusions, torch.zeros(B, 1, H, W).cuda()), f"{disocclusions}"
    assert torch.allclose(im1_buffer, im1_buffer_target), f"{im1_buffer, im1_buffer_target}"
    assert torch.allclose(d_buffer, d_buffer_target), f"{d_buffer, d_buffer_target}"
    assert torch.allclose(wght_buffer, wght_buffer_target), f"{wght_buffer, wght_buffer_target}"
    assert torch.allclose(warped, torch.ones(B, 3, H, W).cuda()), f"{warped}"


def test_forward_warp_max_motion_arange():
    from Forward_Warp import forward_warp_max_motion

    B, H, W = 1, 1, 6

    img = torch.arange(W, dtype=torch.float).view(B, 1, H, W).cuda()
    flow = torch.zeros(B, H, W, 2).cuda()
    flow[0,0,:,0] = torch.arange(W)

    fw = forward_warp_max_motion()

    warped, disocclusions, im1_buffer, d_buffer, wght_buffer = fw(img, flow, debug=True)

    warped_target = torch.tensor([0, 0, 1, 0, 2, 0], dtype=torch.float).view(B, 1, H, W).cuda()
    disocclusions_target = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float).view(B, 1, H, W).cuda()
    d_buffer_target = torch.tensor([0, 0, 100, 0, 200, 0], dtype=torch.int32).view(B, 1, H, W).cuda()

    assert torch.allclose(warped, warped_target), f"{warped, warped_target}"
    assert torch.allclose(disocclusions, disocclusions_target), f"{disocclusions, disocclusions_target}"
    assert torch.allclose(d_buffer, d_buffer_target), f"{d_buffer, d_buffer_target}"


def test_forward_warp_max_motion_occlusions():
    from Forward_Warp import forward_warp_max_motion

    B, H, W = 1, 1, 6

    img = torch.arange(W, dtype=torch.float).view(B, 1, H, W).cuda()
    flow = torch.zeros(B, H, W, 2).cuda()
    flow[0,0,:,0] = torch.tensor([1, 0, 1, 1, 2, 0])

    fw = forward_warp_max_motion()

    warped, disocclusions, im1_buffer, d_buffer, wght_buffer = fw(img, flow, debug=True)

    warped_target = torch.tensor([0, 0, 0, 2, 3, 5], dtype=torch.float).view(B, 1, H, W).cuda()
    disocclusions_target = torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.float).view(B, 1, H, W).cuda()

    assert torch.allclose(warped, warped_target), f"{warped, warped_target}"
    assert torch.allclose(disocclusions, disocclusions_target), f"{disocclusions, disocclusions_target}"


def test_forward_warp_max_motion_multiple_occlusions():
    from Forward_Warp import forward_warp_max_motion

    B, H, W = 1, 1, 4

    img = torch.arange(W, dtype=torch.float).view(B, 1, H, W).cuda()
    flow = torch.zeros(B, H, W, 2).cuda()
    flow[0,0,:,0] = torch.tensor([0, 2, 1, 0])

    fw = forward_warp_max_motion()

    warped, disocclusions, im1_buffer, d_buffer, wght_buffer = fw(img, flow, debug=True)

    warped_target = torch.tensor([0, 0, 0, 1], dtype=torch.float).view(B, 1, H, W).cuda()
    disocclusions_target = torch.tensor([0, 1, 1, 0], dtype=torch.float).view(B, 1, H, W).cuda()
    d_buffer_target = torch.tensor([0,0,0,200], dtype=torch.int32).view(B, 1, H, W).cuda()

    assert torch.allclose(d_buffer, d_buffer_target), f"{d_buffer, d_buffer_target}"
    assert torch.allclose(warped, warped_target), f"{warped, warped_target}"
    assert torch.allclose(disocclusions, disocclusions_target), f"{disocclusions, disocclusions_target}"


def test_forward_warp_max_motion_expansion():
    from Forward_Warp import forward_warp_max_motion

    B, H, W = 1, 1, 16

    img = torch.arange(W, dtype=torch.float).view(B, 1, H, W).cuda()
    flow = torch.zeros(B, H, W, 2).cuda()
    flow[0,0,:,0] = torch.arange(W) * 0.5

    fw = forward_warp_max_motion()

    warped, disocclusions, im1_buffer, d_buffer, wght_buffer = fw(img, flow, debug=True)

    warped_target = torch.tensor(
        [0.,  1.,  1.,  2.,  3.,  3.,  4.,  5.,  5.,  6.,  7.,  7.,  8.,  9., 9., 10.],
        dtype=torch.float
    ).view(B, 1, H, W).cuda()
    disocclusions_target = torch.zeros(B, 1, H, W, dtype=torch.float).cuda()

    assert torch.allclose(warped, warped_target), f"{warped, warped_target}"
    assert torch.allclose(disocclusions, disocclusions_target), f"{disocclusions, disocclusions_target}"

    flow[0,0,:,0] = torch.arange(W) * 0.25

    warped, disocclusions, im1_buffer, d_buffer, wght_buffer = fw(img, flow, debug=True)

    warped_target = torch.tensor(
        [ 0.0000,  1.0000,  1.6667,  2.3333,  3.0000,  4.0000,  5.0000, 5.6667,
          6.3333,  7.0000,  8.0000,  9.0000,  9.6667, 10.3333, 11.0000, 12.0000]
        , dtype=torch.float
    ).view(B, 1, H, W).cuda()
    disocclusions_target = torch.zeros(B, 1, H, W, dtype=torch.float).cuda()

    assert torch.allclose(warped, warped_target, atol=1e-4), f"{warped, warped_target, warped - warped_target}"
    assert torch.allclose(disocclusions, disocclusions_target), f"{disocclusions, disocclusions_target}"

    flow[0,0,:,0] = torch.arange(W) * 0.24999

    warped, disocclusions, im1_buffer, d_buffer, wght_buffer = fw(img, flow, debug=True)

    # Note the differences to arange * 0.25
    warped_target = torch.tensor(
        [ 0.0000,  1.0000,  2.0000,  2.3334,  3.0000,  4.0000,  5.0000, 6.0000,
          6.3334,  7.0000,  8.0000,  9.0000, 10.0000, 10.3335, 11.0000, 12.0000]
        , dtype=torch.float
    ).view(B, 1, H, W).cuda()

    assert torch.allclose(warped, warped_target, atol=1e-4), f"{warped, warped_target, warped - warped_target}"
    assert torch.allclose(disocclusions, disocclusions_target), f"{disocclusions, disocclusions_target}"

    flow[0,0,:,0] = torch.arange(W) * 0.25001

    warped, disocclusions, im1_buffer, d_buffer, wght_buffer = fw(img, flow, debug=True)

    # Note the differences to arange * 0.25
    warped_target = torch.tensor(
        [ 0.0000,  1.0000,  1.6666,  2.0000,  3.0000,  4.0000,  5.0000,  5.6666,
          6.0000,  7.0000,  8.0000,  9.0000,  9.6665, 10.0000,  11.0000, 12.000]
        , dtype=torch.float
    ).view(B, 1, H, W).cuda()

    assert torch.allclose(warped, warped_target, atol=1e-4), f"{warped, warped_target, warped - warped_target}"
    assert torch.allclose(disocclusions, disocclusions_target), f"{disocclusions, disocclusions_target}"

    # # assert False, '\n'.join([str(val) for val in [img, flow, warped, disocclusions, im1_buffer, d_buffer, wght_buffer]])
