import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import math
import random
from tqdm import tqdm

def create_circular_mask(h, w, center=None, radius=None):
    '''
        https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    '''
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def extract_contours(img, device = 'cpu', reverse = False, expand = False, max_iterations = 256):
    img_max = img > 0.900
    img_mask = img_max
    any_pixel = True
    image_contours = []
    cindx = 1  # countour index
    ks = 3 #contour thickness
    #weights_np = create_circular_mask(ks, ks)
    weights_np = np.ones([ks,ks])
    weights = torch.tensor(weights_np, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(0)  # cho=1, chi=1, kH, kW
    sumw = torch.sum(weights)
    weights = weights/sumw

    while(any_pixel or cindx < max_iterations):
        img_padded = F.pad(img_mask, [ks // 2] * 4, mode='constant', value=0)
        img_contour = F.conv2d(img_padded.float(), weights) #b=1, c=1, H, W
        if expand:
            img_contour = (img_contour > 0.001) ^ img_mask
            img_mask = img_mask + img_contour
            any_pixel = torch.any(~img_mask)
        else:
            img_contour = (img_contour < 0.999) * img_mask
            img_mask = img_mask ^ img_contour
            any_pixel = torch.any(img_mask)
        if any_pixel:
            image_contours.append(img_contour)
            cindx += 1
        else:
            break
    if reverse:
        image_contours.reverse()
    image_contours = torch.cat(image_contours, dim=1)
    flat_bool = torch.sum(image_contours, dim=1) < 0.999
    if expand:
        flat_bool = flat_bool*(~img_max)
    else:
        flat_bool = flat_bool*img_max
    if reverse:
        image_contours[:, 0] += flat_bool[:, 0]
    else:
        image_contours[:, -1] += flat_bool[:, 0]

    return image_contours

def interpolate_contours(contour1, contour0):
    contour1 = contour1.float()
    weight = torch.eye(contour0.size()[1], device=contour0.device) #ctr0, ctr0
    weight = F.interpolate(weight[None, None], size=(contour0.size()[1], contour1.size()[1]), mode='bicubic')
    weight = weight.permute(2,3,0,1)
    contour1 = F.conv2d(contour1, weight)
    #contour1 = contour1 > 0.501 # 1, ctr(c0), H, W
    return contour1

def write_tensor(tensor):
    f = open('./debug/tensor.txt', 'w')
    f.write(tensor.cpu().__str__())
    f.close()

def save_img_tensor(tensor, out_path = './debug/test.png', resolution = -1):
    '''
    tensor: ch, H, W
    '''
    img = transforms.ToPILImage()(tensor)
    if args.resolution > 0:
        img = img.resize(res0, resample=Image.BILINEAR)
    img.save(out_path, format='PNG')

def get_position_matrix(res, device, blue_channel = False):
    x = torch.arange(0, res, device=device, dtype=torch.long)
    x = x.unsqueeze(0).expand(res,res)
    to_stack = [x.t(), x]
    if blue_channel:
        to_stack.append( torch.zeros_like(x) )
    return torch.stack(to_stack, dim=2) #res,res,2

class DirectionalTransform(nn.Module):
    def __init__(self, c0, grid_smoothing):
        super().__init__()
        self.smooth_factor = grid_smoothing
        #transform = torch.ones((*c0.size()[1:4], 2), device=c0.device)
        #for ctr in range(c0.size()[1]):
        #    ctr_bool = (c0[0, ctr] < 0.5).nonzero(as_tuple=True)
        #    transform[ctr][ctr_bool] = 0
        h, w = [ c0.size()[2] ]*2
        x, y = torch.arange(h) / (h - 1), torch.arange(w) / (w - 1)
        grid = torch.dstack(torch.meshgrid(x, y)) * 2 - 1
        self.transform = nn.Parameter(grid.unsqueeze(0))
        self.base_grid = grid.unsqueeze(0).clone().to(c0.device)

    def smooth_grid(self):
        ks = 3
        sf = self.smooth_factor
        uv = self.transform.permute(0,3,1,2)

        uv_padded = F.pad(uv, [ks // 2] * 4, mode='reflect')
        smoothed_uv = F.avg_pool2d(uv_padded, ks, stride=1)
        self.transform.data = self.transform*(1-sf) + smoothed_uv.permute(0,2,3,1)*sf

    def forward(self, x):
        to_return = F.grid_sample(x, self.transform, mode='bicubic')
        return to_return

def semi_flatten_contours(c0, border_ctr):
    ks = 5
    n_contours = c0.size()[1]
    ctr_range = torch.arange(0, 1, 1 / n_contours, device=c0.device)
    kernel = torch.ones([n_contours, n_contours, ks, ks], device=c0.device)/ks**2

    ctr_range = torch.arange(0, 1, 1 / n_contours, device=c0.device)
    ctr_loss = (1 - torch.abs(ctr_range - border_ctr/n_contours))**2
    ctr_loss = ctr_loss.view(1, -1, 1, 1)

    c0 = c0

    for ctr in range(n_contours):
        ctr_bleed = (1 - torch.abs(ctr_range - ctr/n_contours))**1 #ctr
        ctr_bleed = ctr_bleed.unsqueeze(-1).unsqueeze(-1) #ctr, 1, 1
        kernel[:, ctr] *= ctr_bleed
    img_padded = F.pad(c0.float(), [ks // 2] * 4, mode='reflect')
    semi_flattened = F.conv2d(img_padded, kernel)
    return semi_flattened

def contour_with_position(c0):
    contour_position = torch.zeros([*c0.size(), 2], device= c0.device)
    for ctr in range(c0.size()[1]):
        position = get_position_matrix(c0.size()[2], device=c0.device) / c0.size()[2] #H, W, 2
        ctr_bool = (c0[0, ctr] < 0.5).nonzero(as_tuple=True) #H, W        position[ctr_bool] = 0
        contour_position[0, ctr] = position
    return contour_position

def grid_to_UV(grid, rotation = 90):
    transform = grid.data.detach()  # 1, H, W, 2
    uv_map = (torch.sum(transform, dim=0) + 1) / 2
    blue_channel = torch.zeros((*uv_map.size()[:-1], 1), device=grid.device)
    uv_map = torch.cat([uv_map, blue_channel], dim=2)
    uv_map = uv_map.permute(2, 0, 1)
    uv_map = transforms.functional.rotate(uv_map, rotation)
    return uv_map

def mesh_optim(alpha0, alpha1, orig0=None, orig1=None, steps=1000, lr=0.001, ctr_l = 0.999, grid_smoothing=0.1, device='cpu'):
    '''
    Parameters
    ----------
    img0: [num_masks], (b)1, (C)1, H, W
    img1: [num_masks], (b)1, (C)1, H, W
    lr: learning rate
    steps: steps to perform

    Returns
    -------
        pixel mapping x,y of alpha to x',y' of alpha 1
    '''
    with torch.no_grad():
        p_c1 = extract_contours(alpha1, device, reverse=True) #1, ctr/2, H, W
        n_c1 = extract_contours(alpha1, device, expand=True)

        p_c0 = extract_contours(alpha0, device, reverse=True)
        n_c0 = extract_contours(alpha0, device, expand=True)

        p_c1 = interpolate_contours(p_c1, p_c0) #1, ctr(c0)/2, H, W
        n_c1 = interpolate_contours(n_c1, n_c0)
        border_ctr = p_c0.size()[1] - 1

        c0 = torch.cat([p_c0, n_c0], dim=1)
        c1 = torch.cat([p_c1, n_c1], dim=1)

        sm_c0 = semi_flatten_contours(c0, border_ctr)
        sm_c1 = semi_flatten_contours(c1, border_ctr)


    model = DirectionalTransform(c0, grid_smoothing)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)
    c0f = c0.float()
    c1f = c1.float()
    with torch.no_grad():
        mod1 = F.grid_sample(sm_c1, model.base_grid, mode='bicubic')
        if ctr_l < 1:
            mod1_orig = F.grid_sample(orig1, model.base_grid, mode='bicubic')
    n_contours = c0.size()[1]

    for i in tqdm(range(steps)):
        opt.zero_grad()
        mod0 = model(sm_c0)
        loss = F.mse_loss(mod0, mod1)
        if ctr_l < 1:
            mod0_orig = model(orig0)
            loss_orig = F.mse_loss(mod0_orig, mod1_orig)
        else:
            loss_orig = 0
        loss = loss*ctr_l + loss_orig*(1-ctr_l)
        loss.backward()
        opt.step()
        model.smooth_grid()
        if (i % steps/10) == 0:
            print(loss)
    uv_map = grid_to_UV(model.transform)
    return uv_map

def flatten_move(contT, flat_move_bool):
    flat_move = contT + flat_move_bool.float()
    flat_move = torch.round(torch.sum(flat_move, dim=1)).long()  # B, H, W, 2
    return flat_move

def remove_transparency(im, bg_colour=(0, 0, 0)):
    im = im.convert('RGBA')
    bg = Image.new('RGBA', im.size, bg_colour)
    im = Image.alpha_composite(bg, im)
    return im.convert('L')
def load_image(folder_path, partial_path, resolution, device):
    file_path = os.path.join(folder_path, partial_path)
    img = Image.open(file_path)
    img = remove_transparency(img)
    orig_resolution = img.size[0:2]
    if resolution > 0:
        img = img.resize((resolution, resolution), resample=Image.LANCZOS)
    img = transforms.ToTensor()(img)
    #img = transforms.functional.hflip(img)
    img = transforms.functional.vflip(img)
    img = img.unsqueeze(0).to(device)
    return img, orig_resolution

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_alpha0", type=str, default="./in_0")
    parser.add_argument("--in_alpha1", type=str, default="./in_1")
    parser.add_argument("--in_orig0", type=str, default="./orig_0")
    parser.add_argument("--in_orig1", type=str, default="./orig_1")
    parser.add_argument("--output_folder", type=str, default="./out")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--contour_lambda", type=float, default=1)
    parser.add_argument("--grid_smoothing", type=float, default=1)
    parser.add_argument("--lr", type=float, default= 0.002)
    #parser.add_argument("--kernel_size", type=int, default=5)
    args = parser.parse_args()
    to_iter = []
    if not os.path.exists(args.in_alpha0) and not os.path.isdir(args.in_alpha0):
        to_iter = []
    else:
        to_iter = os.listdir(args.in_alpha0)
    if not os.path.exists(args.in_alpha1) and not os.path.isdir(args.in_alpha1):
        to_iter = []
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for pfp in os.listdir(args.in_alpha0):
        basefp, ext = os.path.splitext(pfp)
        if not ext.lower() in ['.png', '.jpg', '.jpeg', '.exr']:
            continue
        alpha0, res0 = load_image(args.in_alpha0, pfp, args.resolution, args.device)
        alpha1, res1 = load_image(args.in_alpha1, pfp, args.resolution, args.device)
        assert res0 == res1
        if args.contour_lambda < 1:
            orig0, res0 = load_image(args.in_orig0, pfp, args.resolution, args.device)
            orig1, res0 = load_image(args.in_orig1, pfp, args.resolution, args.device)
        else:
            orig0 = None
            orig1 = None
        uv_map = mesh_optim(alpha0, alpha1, orig0, orig1,
                        ctr_l=args.contour_lambda,steps=args.steps, grid_smoothing=args.grid_smoothing,
                        device=args.device, lr = args.lr)
        uv_map = uv_map #.permute(1, 2, 0)
        save_img_tensor(uv_map, os.path.join(args.output_folder, pfp), res0)
