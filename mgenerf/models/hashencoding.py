import megengine as mge
import numpy as np
from .registry import BACKBONES
import megengine.module as M
import megengine.functional as F
from megengine.module import init


def total_variation_loss(embeddings, min_resolution, max_resolution, level, log2_hashmap_size, n_levels):
    # Get resolution
    b = F.exp((F.log(max_resolution) - F.log(min_resolution))/(n_levels-1))
    resolution = F.floor(min_resolution * (b ** level))

    # Cube size to apply TV loss
    min_cube_size = min_resolution - 1
    max_cube_size = 50 # can be tuned
    
    if min_cube_size > max_cube_size:
        raise RuntimeError("")

    cube_size = F.floor(F.clip(resolution/10.0, min_cube_size, max_cube_size)).astype(np.int32)

    # Sample cuboid
    min_vertex = np.random.randint(0, resolution-cube_size, (3,), dtype=np.int32)
    idx = min_vertex + np.stack([np.arange(cube_size+1, dtype=np.int32) for _ in range(3)], axis=-1)
    np_mesh = np.meshgrid(idx[:,0], idx[:,1], idx[:,2])
    mesh = mge.tensor(np_mesh, dtype=np.int32)
    cube_indices = F.stack(mesh, axis=-1)

    hashed_indices = hash(cube_indices, log2_hashmap_size)
    cube_embeddings = embeddings(hashed_indices)
    #hashed_idx_offset_x = hash(idx+torch.tensor([1,0,0]), log2_hashmap_size)
    #hashed_idx_offset_y = hash(idx+torch.tensor([0,1,0]), log2_hashmap_size)
    #hashed_idx_offset_z = hash(idx+torch.tensor([0,0,1]), log2_hashmap_size)

    # Compute loss
    #tv_x = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_x), 2).sum()
    #tv_y = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_y), 2).sum()
    #tv_z = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_z), 2).sum()
    tv_x = F.pow(cube_embeddings[1:,:,:,:]-cube_embeddings[:-1,:,:,:], 2).sum()
    tv_y = F.pow(cube_embeddings[:,1:,:,:]-cube_embeddings[:,:-1,:,:], 2).sum()
    tv_z = F.pow(cube_embeddings[:,:,1:,:]-cube_embeddings[:,:,:-1,:], 2).sum()
    return (tv_x + tv_y + tv_z) / cube_size


BOX_OFFSETS = mge.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]], dtype = np.int32) # [1,8,3]


def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates  [b,8,3]
    log2T:  logarithm of T w.r.t 2
    '''
    coords = coords.numpy()
    assert coords.dtype == np.int32
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = np.zeros_like(coords[..., 0]) # [b,8]

    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return mge.tensor((((1<<log2_hashmap_size)-1) & xor_result).astype(np.int32)) # 消除高位


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box

    if F.max(xyz > box_max) > 0 or F.max(xyz < box_min) > 0:
        print("ALERT: some points are outside bounding box. Clipping them!")
        # pdb.set_trace()
        for i in range(3):
            xyz[:, i] = F.clip(xyz[:, i], lower=box_min[i], upper=box_max[i])

    grid_size = (box_max - box_min) / resolution # 每个voxel的实际大小
    
    bottom_left_idx = F.floor((xyz-box_min)/grid_size).astype(np.int32) # [B, 3]
    voxel_min_vertex = bottom_left_idx * grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + grid_size

    voxel_indices = F.expand_dims(bottom_left_idx, axis=1) + BOX_OFFSETS # [B, 1, 3] + [1, 8, 3]
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices


@BACKBONES.register_module
class HashEncoding(M.Module):
    def __init__(self, bounding_box, 
                       finest_resolution,
                       log2_hashmap_size,
                       n_levels=16,
                       n_features_per_level=2,
                       base_resolution=16
                       ):
        super(HashEncoding, self).__init__()

        self.bounding_box = (mge.tensor(bounding_box[0]) , mge.tensor(bounding_box[1]))  # need to sure
        self.n_levels = n_levels

        assert n_levels > 10

        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = mge.tensor(base_resolution)
        self.finest_resolution = mge.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level
        
        self.b = F.exp((F.log(self.finest_resolution) - F.log(self.base_resolution))/(n_levels-1))
        # e((ln(f) - ln(b)) / 15)
        self.embeddings = [M.Embedding(2**self.log2_hashmap_size, self.n_features_per_level) for i in range(n_levels)]

        # custom uniform initialization
        for i in range(n_levels):
            init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)


    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0:1]) + voxel_embedds[:,4]*weights[:,0:1]
        c01 = voxel_embedds[:,1]*(1-weights[:,0:1]) + voxel_embedds[:,5]*weights[:,0:1]
        c10 = voxel_embedds[:,2]*(1-weights[:,0:1]) + voxel_embedds[:,6]*weights[:,0:1]
        c11 = voxel_embedds[:,3]*(1-weights[:,0:1]) + voxel_embedds[:,7]*weights[:,0:1]

        # step 2
        c0 = c00*(1-weights[:,1:2]) + c10*weights[:,1:2]
        c1 = c01*(1-weights[:,1:2]) + c11*weights[:,1:2]

        # step 3
        c = c0*(1-weights[:,2:3]) + c1*weights[:,2:3]

        return c # [b,2]

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = F.floor(self.base_resolution * (self.b **i))
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = get_voxel_vertices(x, self.bounding_box,
                                                resolution, self.log2_hashmap_size)
            
            voxel_embedds = self.embeddings[i](hashed_voxel_indices) # input: [b, 8]  output: [b,8,2]

            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        return F.concat(x_embedded_all, axis=-1)

    def get_tv_loss(self):
        n_levels = self.n_levels
        min_res = self.base_resolution
        max_res = self.finest_resolution
        log2_hashmap_size = self.log2_hashmap_size
        TV_loss = sum(total_variation_loss(self.embeddings[i], 
                                        min_res, max_res,
                                        i, log2_hashmap_size,
                                        n_levels=n_levels) for i in range(n_levels))
        return TV_loss