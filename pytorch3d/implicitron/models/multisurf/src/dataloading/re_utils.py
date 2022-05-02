import os
from glob import glob
import torch
import re
import numpy as np

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def split_input(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 10000
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        # data['intrinsics'] = torch.index_select(model_input['intrinsics'], 1, indx)
        # data['pose'] = torch.index_select(model_input['pose'], 1, indx)
        # data['imgs'] = torch.index_select(model_input['imgs'], 1, indx)
        # data['proj_mats'] = torch.index_select(model_input['proj_mats'], 1, indx)
        # data['imgs'] = torch.index_select(model_input['imgs'], 1, indx)
        
        split.append(data)
    return split

def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def get_xyz(depth_root, intrinsics_extrinsics_root, N):
    depth_h = np.array(read_pfm(depth_root)[0], dtype=np.float32)  # (128, 160)
    h, w = depth_h.shape
    with open(intrinsics_extrinsics_root) as f:
        lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        # depth_min = float(lines[11].split()[0]) * self.scale_factor
        # depth_max = depth_min + float(lines[11].split()[1]) * 192 * self.scale_factor
    u = np.tile(np.arange(w).reshape(1, 1, w), (1, h, 1)) #[1,h,w]
    v = np.tile(np.arange(h).reshape(1, h, 1), (1, 1, w)) #[1,h,w]
    d = depth_h.reshape(-1)
    u = u.reshape(-1)
    v = v.reshape(-1)
    mask = np.where(d > 0)[0]
    #pos_h_num = mask.shape[0]
    sample = np.random.choice(mask, N)
    d = d[sample][None, :]
    u = u[sample][None, :]
    v = v[sample][None, :]
    ones = np.ones_like(u)
    uv1 = np.concatenate((u,v,ones), axis=0) # [3, N]
    intrinsics_inv = np.linalg.inv(intrinsics)
    extrinsics_inv = np.linalg.inv(extrinsics)
    cam_coord_1 = np.matmul(intrinsics_inv, uv1) * d #[3,N]
    cam_coord = np.concatenate((cam_coord_1, ones), axis=0) #[4,N]
    world_coord = np.matmul(extrinsics_inv, cam_coord)[:3, :].T#[N,3]
    return world_coord, cam_coord_1, uv1, d
