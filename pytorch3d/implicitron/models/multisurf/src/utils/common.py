from ipdb import set_trace
import torch
import numpy as np
import logging
from copy import deepcopy


logger_py = logging.getLogger(__name__)


def get_depth_mask_gt(sampled_pixels_origin, depth_map_origin, depth_mask_origin):
    # set_trace()
    sampled_pixels = sampled_pixels_origin.clone().squeeze()
    depth_map = depth_map_origin.clone().squeeze(0)
    sampled_depth = depth_map[sampled_pixels[:, 1].long(), sampled_pixels[:, 0].long()]
    depth_mask = depth_mask_origin.clone().bool().squeeze()
    depth_gt = torch.where(depth_mask[sampled_pixels[:, 1].long(), sampled_pixels[:, 0].long()]==True, sampled_depth.float(), torch.tensor(0.0).cuda())
    return depth_gt.unsqueeze(0)


def sample_patch_points_with_mask_neighbor(n_points, image_resolution, origin_mask):
    # set_trace()
    # assert(list(mask.shape) == image_resolution)
    # set_trace()
    n_center = int(n_points/4)
    n_center_object = int(n_center*3/4)
    n_center_background = int(n_center/4)
    if origin_mask is None:
        origin_mask = torch.ones(size=image_resolution)
    mask = torch.ones(size=image_resolution)
    mask_object = origin_mask.squeeze()
    mask_background = ~mask_object
    
    assert(mask_object.shape[0] == image_resolution[0] and mask_object.shape[1] == image_resolution[1])
    h, w = image_resolution
    # index = torch.stack([torch.arange(h).reshape(-1, 1).repeat(1, w), torch.arange(w).reshape(1, -1).repeat(h, 1)], dim=2)
    index = torch.stack([torch.arange(w).reshape(1, -1).repeat(h, 1), torch.arange(h).reshape(-1, 1).repeat(1, w)], dim=2)
    index_masked = index[mask.bool()]
    index_background = index[mask_background.bool()]
    sampled_pixels_object = index_masked[torch.multinomial(torch.ones(index_masked.shape[0]), n_center_object, replacement=True)]
    sampled_pixels_background = index_background[torch.multinomial(torch.ones(index_background.shape[0]), n_center_background, replacement=True)]
    sampled_pixels = torch.cat((sampled_pixels_object, sampled_pixels_background),dim=0)
    # sampled_p_ndc = -(sampled_pixels / torch.tensor(image_resolution)) * 2 + 1
    neighbor = sampled_pixels_object.repeat(3,1) + torch.randint(-3,4,sampled_pixels_object.repeat(3,1).shape)
    neighbor = neighbor.clamp(0,199)
    # aaa = mask.clone().detach()
    # set_trace()
    try:
        neighbor_masked = torch.where(mask_object[neighbor[:,1],neighbor[:,0]].unsqueeze(1).repeat(1,2)==False, sampled_pixels_object.repeat(3,1).cuda(), neighbor.cuda())
    except:
        set_trace()
    neighbor_masked_1 =torch.cat((neighbor_masked[:n_center_object,:], sampled_pixels_background.cuda()),dim=0)
    neighbor_masked_2 =torch.cat((neighbor_masked[n_center_object:2*n_center_object,:], sampled_pixels_background.cuda()),dim=0)
    neighbor_masked_3 =torch.cat((neighbor_masked[2*n_center_object:3*n_center_object,:], sampled_pixels_background.cuda()),dim=0)
    neighbor_masked_all = torch.cat((neighbor_masked_1,neighbor_masked_2,neighbor_masked_3),dim=0)
    sampled_pixels_all = torch.cat((sampled_pixels.cuda(), neighbor_masked_all),dim=0)
    sampled_p_ndc_all = -(sampled_pixels_all.cpu() / torch.tensor(image_resolution)) * 2 + 1
    return sampled_p_ndc_all.unsqueeze(0), sampled_pixels_all.float().unsqueeze(0)


def sample_patch_points_with_mask(n_points, image_resolution, origin_mask):
    # set_trace()
    # assert(list(mask.shape) == image_resolution)
    if origin_mask is None:
        origin_mask = torch.ones(size=image_resolution)
    mask = origin_mask.squeeze()
    
    assert(mask.shape[0] == image_resolution[0] and mask.shape[1] == image_resolution[1])
    h, w = image_resolution
    # index = torch.stack([torch.arange(h).reshape(-1, 1).repeat(1, w), torch.arange(w).reshape(1, -1).repeat(h, 1)], dim=2)
    index = torch.stack([torch.arange(w).reshape(1, -1).repeat(h, 1), torch.arange(h).reshape(-1, 1).repeat(1, w)], dim=2)
    index_masked = index[mask.bool()]
    sampled_pixels = index_masked[torch.multinomial(torch.ones(index_masked.shape[0]), n_points, replacement=True)]
    sampled_p_ndc = -(sampled_pixels / torch.tensor(image_resolution)) * 2 + 1
    return sampled_p_ndc.unsqueeze(0), sampled_pixels.float().unsqueeze(0)


def arange_pixels_with_mask(image_resolution, origin_mask):
    # set_trace()
    if origin_mask is None:
        origin_mask = torch.ones(size=image_resolution)
    mask = origin_mask.squeeze()
    
    assert(mask.shape[0] == image_resolution[0] and mask.shape[1] == image_resolution[1])
    h, w = image_resolution
    # index = torch.stack([torch.arange(h).reshape(-1, 1).repeat(1, w), torch.arange(w).reshape(1, -1).repeat(h, 1)], dim=2)
    index = torch.stack([torch.arange(w).reshape(1, -1).repeat(h, 1), torch.arange(h).reshape(-1, 1).repeat(1, w)], dim=2)
    sampled_pixels = index[mask.bool()]
    sampled_p_ndc = -(sampled_pixels / torch.tensor(image_resolution)) * 2 + 1
    return sampled_pixels.float().unsqueeze(0), sampled_p_ndc.unsqueeze(0)
    

def sample_patch_points(n_points,
                        image_resolution, 
                        sensor_size=np.array([[-1, 1],[-1, 1]]), 
                        continuous=True,patch_size=1):
    ''' Returns sampled points in the range of sensorsize.

    Args:
        batch_size (int): required batch size
        n_points (int): number of points to sample
        patch_size (int): size of patch; if > 1, patches of size patch_size
            are sampled instead of individual points
        image_resolution (tuple): image resolution (required for calculating
            the pixel distances)
        continuous (bool): whether to sample continuously or only on pixel
            locations
    '''
    assert(patch_size > 0)
    batch_size = 1

    n_patches = int(n_points)

    if continuous:
        p = torch.rand(batch_size, n_patches, 2)  # [0, 1]
    else:
        px = torch.randint(0, image_resolution[1], size=(
            batch_size, n_patches, 1)).float() 
        py = torch.randint(0, image_resolution[0], size=(
            batch_size, n_patches, 1)).float() 
        p = torch.cat([px, py], dim=-1)

    p = p.view(batch_size, n_patches, 1, 2) 
    p = p.view(batch_size, -1, 2)
    pix = p.clone()

    p[:,:,0] *= (sensor_size[1,1] - sensor_size[1,0]) / (image_resolution[1] - 1)
    p[:,:,1] *= (sensor_size[0,1] - sensor_size[0,0])  / (image_resolution[0] - 1)
    p[:,:,0] += sensor_size[1,0] 
    p[:,:,1] += sensor_size[0,0]
    p[:,:,0] = -p[:,:,0]
    p[:,:,1] = -p[:,:,1]

    assert(p.max() <= sensor_size.max())
    assert(p.min() >= sensor_size.min())
    assert(pix.max() < max(image_resolution))
    assert(pix.min() >= 0)

    return p, pix


def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                  subsample_to=None):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    '''
    h, w = resolution
    n_points = resolution[0] * resolution[1]

    # Arrange pixel location in scale resolution
    pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
    pixel_locations = torch.stack(
        [pixel_locations[0], pixel_locations[1]],
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = (image_range[1] - image_range[0])/ 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc
    pixel_scaled[:, :, 0] = -pixel_scaled[:, :, 0]
    pixel_scaled[:, :, 1] = -pixel_scaled[:, :, 1]
    # Subsample points if subsample_to is not None and > 0
    if (subsample_to is not None and subsample_to > 0 and
            subsample_to < n_points):
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,),
                               replace=False)
        pixel_scaled = pixel_scaled[:, idx]
        pixel_locations = pixel_locations[:, idx]

    # set_trace()
    return pixel_locations, pixel_scaled


def to_pytorch(tensor, return_type=False):
    ''' Converts input tensor to pytorch.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    '''
    is_numpy = False
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor)
        is_numpy = True

    tensor = tensor.clone()
    if return_type:
        return tensor, is_numpy
    return tensor


def get_mask(tensor):
    ''' Returns mask of non-illegal values for tensor.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
    '''
    tensor, is_numpy = to_pytorch(tensor, True)
    mask = ((abs(tensor) != np.inf) & (torch.isnan(tensor) == False))
    mask = mask.bool()
    if is_numpy:
        mask = mask.numpy()

    return mask


def transform_mesh(mesh, transform):
    ''' Transforms a mesh with given transformation.

    Args:
        mesh (trimesh mesh): mesh
        transform (tensor): transformation matrix of size 4 x 4
    '''
    mesh = deepcopy(mesh)
    v = np.asarray(mesh.vertices).astype(np.float32)
    v_transformed = transform_pointcloud(v, transform)
    mesh.vertices = v_transformed
    return mesh


def transform_pointcloud(pointcloud, transform):
    ''' Transforms a point cloud with given transformation.

    Args:
        pointcloud (tensor): tensor of size N x 3
        transform (tensor): transformation of size 4 x 4
    '''

    assert(transform.shape == (4, 4) and pointcloud.shape[-1] == 3)

    pcl, is_numpy = to_pytorch(pointcloud, True)
    transform = to_pytorch(transform)

    # Transform point cloud to homogen coordinate system
    pcl_hom = torch.cat([
        pcl, torch.ones(pcl.shape[0], 1)
    ], dim=-1).transpose(1, 0)

    # Apply transformation to point cloud
    pcl_hom_transformed = transform @ pcl_hom

    # Transform back to 3D coordinates
    pcl_out = pcl_hom_transformed[:3].transpose(1, 0)
    if is_numpy:
        pcl_out = pcl_out.numpy()

    return pcl_out


def get_tensor_values(tensor, pe, grid_sample=True, mode='nearest',
                      with_mask=False, squeeze_channel_dim=False):
    '''
    Returns values from tensor at given location p.

    Args:
        tensor (tensor): tensor of size B x C x H x W
        p (tensor): position values scaled between [-1, 1] and
            of size B x N x 2
        grid_sample (boolean): whether to use grid sampling
        mode (string): what mode to perform grid sampling in
        with_mask (bool): whether to return the mask for invalid values
        squeeze_channel_dim (bool): whether to squeeze the channel dimension
            (only applicable to 1D data)
    '''

    batch_size, _, h, w = tensor.shape

    p = pe.clone().detach()
    p[:, :, 0] = 2.*p[:, :, 0]/w - 1
    p[:, :, 1] = 2.*p[:, :, 1]/h - 1
    p = p.unsqueeze(1)
    # tensor = tensor.permute(0, 1, 3, 2)
    values = torch.nn.functional.grid_sample(tensor, p, mode=mode)
    values = values.squeeze(2).detach()
    values = values.permute(0, 2, 1)

    if squeeze_channel_dim:
        values = values.squeeze(-1)

    return values


def transform_to_world(pixels, depth, camera_mat, world_mat, scale_mat,
                       invert=True):
    ''' Transforms pixel positions p with given depth value d to world coordinates.

    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    assert(pixels.shape[-1] == 2)

    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    scale_mat = to_pytorch(scale_mat)

    # Invert camera matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

    # Project pixels into camera space
    pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # Transform pixels to world space
    #p_world = scale_mat @ world_mat @ camera_mat @ pixels
    p_world = scale_mat @ world_mat @ pixels
    # Transform p_world back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)

    if is_numpy:
        p_world = p_world.numpy()
    return p_world


def transform_to_camera_space(p_world, camera_mat, world_mat, scale_mat):
    ''' Transforms world points to camera space.
        Args:
        p_world (tensor): world points tensor of size B x N x 3
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
    '''
    batch_size, n_p, _ = p_world.shape
    device = p_world.device

    # Transform world points to homogen coordinates
    p_world = torch.cat([p_world, torch.ones(
        batch_size, n_p, 1).to(device)], dim=-1).permute(0, 2, 1)

    # Apply matrices to transform p_world to camera space
    #p_cam = camera_mat @ world_mat @ scale_mat @ p_world
    p_cam = world_mat @ scale_mat @ p_world
    # Transform points back to 3D coordinates
    p_cam = p_cam[:, :3].permute(0, 2, 1)
    return p_cam


def transform_to_camera_space_new(p_world, camera):
    # set_trace()
    # print(camera.transform_points_screen(p_world).shape)
    # print(camera.transform_points_screen(p_world).max())
    # print(camera.transform_points_screen(p_world).min())
    # pixel_coord = camera.transform_points_screen(p_world, eps=0.01)
    # bias = torch.ones_like(pixel_coord) * 200 
    # pixel_coord = bias - pixel_coord
    return camera.transform_points_screen(p_world, eps=0.01)


def origin_to_world(camera_mat, world_mat, scale_mat):
    ''' Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: true)
    '''
    batch_size = camera_mat.shape[0]
    device = camera_mat.device

    # Create origin in homogen coordinates
    p = torch.zeros(batch_size, 4, 1).to(device)
    p[:, -1] = 1.

    # Invert matrices

    camera_mat = torch.inverse(camera_mat)
    world_mat = torch.inverse(world_mat)
    scale_mat = torch.inverse(scale_mat)

    # Apply transformation
    #p_world = scale_mat @ world_mat @ camera_mat @ p
    p_world = scale_mat @ world_mat @ p
    # Transform points back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1).squeeze(0)
    return p_world


def image_points_to_world(image_points, camera_mat, world_mat, scale_mat,
                          invert=True):
    ''' Transforms points on image plane to world coordinates.

    In contrast to transform_to_world, no depth value is needed as points on
    the image plane have a fixed depth of 1.

    Args:
        image_points (tensor): image points tensor of size B x N x 2
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    batch_size, n_pts, dim = image_points.shape
    assert(dim == 2)
    device = image_points.device
    d_image = torch.ones(batch_size, n_pts, 1).to(device)
    return transform_to_world(image_points, d_image, camera_mat, world_mat,
                              scale_mat, invert=invert)


def check_weights(params):
    ''' Checks weights for illegal values.

    Args:
        params (tensor): parameter tensor
    '''
    for k, v in params.items():
        if torch.isnan(v).any():
            logger_py.warn('NaN Values detected in model weight %s.' % k)


def check_tensor(tensor, tensorname='', input_tensor=None):
    ''' Checks tensor for illegal values.

    Args:
        tensor (tensor): tensor
        tensorname (string): name of tensor
        input_tensor (tensor): previous input
    '''
    if torch.isnan(tensor).any():
        logger_py.warn('Tensor %s contains nan values.' % tensorname)
        if input_tensor is not None:
            logger_py.warn('Input was:', input_tensor)


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def normalize_tensor(tensor, min_norm=1e-5, feat_dim=-1):
    ''' Normalizes the tensor.

    Args:
        tensor (tensor): tensor
        min_norm (float): minimum norm for numerical stability
        feat_dim (int): feature dimension in tensor (default: -1)
    '''
    norm_tensor = torch.clamp(torch.norm(tensor, dim=feat_dim, keepdim=True),
                              min=min_norm)
    normed_tensor = tensor / norm_tensor
    return normed_tensor
