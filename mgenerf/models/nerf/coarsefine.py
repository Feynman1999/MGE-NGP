from ..registry import NERFS
from ..builder import build_backbone
import megengine as mge
import megengine.module as M
import megengine.functional as F
from .basenerf import Base_Nerf
from .utils import sample_pdf, cumprod
import numpy as np
import megengine.distributed as dist
from progress.bar import Bar
import cv2

img2mse = lambda x, y : F.mean((x-y)**2)

mse2psnr = lambda x : -10. * F.log(x) / F.log(mge.tensor([10.]))

def clamp_probs(probs):
    eps = np.finfo(probs.dtype).eps
    eps = mge.tensor(eps)
    return F.clip(probs, lower=eps, upper= 1 - eps)

def probs_to_logits(probs):
    r"""
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    ps_clamped = clamp_probs(probs)
    return F.log(ps_clamped)

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    def raw2alpha(raw, dists, act_fn=F.relu):
        before_exp = -act_fn(raw) * dists
        return 1.0 - F.exp(before_exp), before_exp

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = F.concat([dists, F.full(dists[..., :1].shape, 1e10)], -1)  # [N_rays, N_samples]
    # 距离间隔系数（不是真正距离）

    dists = dists * F.norm(F.expand_dims(rays_d, axis=-2), axis=-1) # [N_rays, N_samples]

    rgb = F.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    
    noise = 0.
    if raw_noise_std > 0.:
        noise = mge.random.normal(size=raw[..., 3].shape) * raw_noise_std

    alpha, before_exp = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    N_rays = before_exp.shape[0]
    # weights = alpha * F.exp(
    #     F.cumsum(F.concat([F.zeros((N_rays, 1)), before_exp], -1), axis=1)[:, :-1]
    # )
    weights = alpha * cumprod(
            F.concat([F.ones((N_rays, 1)), 1.0 - alpha + 1e-10], -1), -1
        )[:, :-1]
    
    rgb_map = F.sum(F.expand_dims(weights, axis=-1) * rgb, -2)  # [N_rays, 3]

    depth_map = F.sum(weights * z_vals, -1) # [N, ]
    acc_map = weights.sum(axis = 1) # [N, ]

    disp_map = 1.0 / F.maximum(
        1e-10 * F.ones_like(depth_map), depth_map / acc_map
    )
    
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - F.expand_dims(acc_map, axis=-1))

    # Calculate weights sparsity loss
    mask = acc_map > 0.5
    probs = weights + 1e-5
    probs = probs / probs.sum(-1, keepdims=True)

    logits = probs_to_logits(probs)

    min_real = np.finfo(logits.dtype).min 
    logits = F.clip(logits, lower = min_real)
    p_log_p = logits * probs
    entropy =  -p_log_p.sum(-1)

    sparsity_loss = entropy * mask

    return rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss

def render_rays(rays_o, rays_d, near, far, viewdirs,
                N_samples,
                N_importance,
                coarse_net,
                fine_net,
                lindisp=False,
                perturb=True,
                white_bkgd=False,
                raw_noise_std=0.,
                **kwargs):
    #t_vals = F.linspace(0., 1., num=N_samples)
    t_vals = mge.tensor(np.logspace(0., 1., num=N_samples, base=2.0), dtype = np.float32) - 1.0

    if not lindisp:
        z_vals = near  + (far  - near ) * t_vals # [B, N_samples]
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    if perturb:
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1]) # N_samples- 1 个中点
        upper = F.concat([mids, z_vals[...,-1:]], -1)
        lower = F.concat([z_vals[...,:1], mids], -1)
        t_rand = mge.random.uniform(low=0, high=1, size=z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # [N_rays, N_samples, 3]
    pts = F.expand_dims(rays_o, axis=-2) + F.expand_dims(rays_d, axis=-2) * F.expand_dims(z_vals, axis=-1)
    
    raw = coarse_net(pts, viewdirs)

    # [b, 64, 4]

    rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)
    # # [b, 3]  [b, ] [b, ] [b, 64] [b, ] [b, ]

    # sample 
    rgb_map_0, disp_map_0, acc_map_0, sparsity_loss_0 = rgb_map, disp_map, acc_map, sparsity_loss

    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, perturb=perturb)
    z_samples = z_samples.detach()

    z_vals, _ = F.sort(F.concat([z_vals, z_samples], -1), descending=False)
    
    pts = F.expand_dims(rays_o, axis=-2) + F.expand_dims(rays_d, axis=-2) * F.expand_dims(z_vals, axis=-1) # [N_rays, N_samples + N_importance, 3]

    raw = fine_net(pts, viewdirs)

    rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'sparsity_loss': sparsity_loss}

    ret['rgb0'] = rgb_map_0
    ret['disp0'] = disp_map_0
    ret['acc0'] = acc_map_0
    ret['sparsity_loss0'] = sparsity_loss_0
    ret['z_std'] = F.std(z_samples, axis=-1)  # [N_rays]

    return ret

def render(rays_o, rays_d, near = 0., far = 1., **kwargs):
    viewdirs = rays_d / F.norm(rays_d, axis=1, keepdims=True) # [x, 3]
    
    if 'bounding_box' in kwargs.keys():
        bounding_box = kwargs['bounding_box']
        # get near and far according to bounding_box
        # [N, 1]
        bounding_box = mge.tensor(bounding_box, dtype=np.float32) # [2,3]
        tmin = (bounding_box[0:1, :] - rays_o) / rays_d
        tmax = (bounding_box[1:2, :] - rays_o) / rays_d
        near = near * F.ones_like(rays_d[..., :1])
        far = F.where(tmin > tmax, tmin, tmax).min(axis=-1, keepdims=True)  - 1e-6
        
        if F.max(near > far) > 0:
            raise RuntimeError("bounding box is too small!")

        if F.max(near + 1 > far) > 0:
            print(" sampled ray < 1 meters ! you can change box to larger")

    else:
        near, far = near * F.ones_like(rays_d[..., :1]), far * F.ones_like(rays_d[..., :1])
    
    all_ret = render_rays(rays_o, rays_d, near, far, viewdirs, **kwargs)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


@NERFS.register_module
class Coarse_Fine_Nerf(Base_Nerf):
    def __init__(self, coarse_net, fine_net, train_cfg, test_cfg):
        super(Coarse_Fine_Nerf, self).__init__(train_cfg=train_cfg, test_cfg=test_cfg)
        
        self.coarse_net = build_backbone(coarse_net)
        self.fine_net = build_backbone(fine_net)

        self.train_kwargs  = {
            'near' : train_cfg.near,
            'far' : train_cfg.far,
            'bounding_box' : train_cfg.bounding_box,
            'N_samples' : train_cfg.N_samples,
            'N_importance' : train_cfg.N_importance,
            'lindisp' : train_cfg.lindisp,
            'perturb' : train_cfg.perturb,
            'sparse_loss_weight' : train_cfg.sparse_loss_weight,
            'tv_loss_weight' : train_cfg.tv_loss_weight
        }

        self.test_kwargs  = {
            'near' : test_cfg.near,
            'far' : test_cfg.far,
            'bounding_box' : test_cfg.bounding_box,
            'N_samples' : test_cfg.N_samples,
            'N_importance' : test_cfg.N_importance,
            'lindisp' : test_cfg.lindisp,
            'perturb' : False,
            'num_cols' : test_cfg.num_cols,
        }

    def train_step(self, batchdata, now_epoch, gm, optim):
        rays = mge.tensor(batchdata['rays'], dtype=np.float32) # [batch, 2, N, 3]
        target = mge.tensor(batchdata['target'], dtype=np.float32)
        rays_o = rays[:, 0, ...].reshape(-1, 3)
        rays_d = rays[:, 1, ...].reshape(-1, 3)
        target = target.reshape(-1, 3)

        with gm:
            rgb, disp, acc, extras = render(rays_o, rays_d, coarse_net = self.coarse_net,
            fine_net = self.fine_net, **self.train_kwargs)
            
            # cal loss
            img_loss = img2mse(rgb, target)
            loss = img_loss
            psnr = mse2psnr(img_loss)
            # psnr = dist.functional.all_reduce_sum(psnr) / dist.get_world_size()

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            
            
            sparsity_loss = self.train_kwargs['sparse_loss_weight']*(extras["sparsity_loss"].sum() + extras["sparsity_loss0"].sum())
            loss = loss + sparsity_loss

            # add Total Variation loss
            # tv_loss = self.train_kwargs['tv_loss_weight'] * self.fine_net.hash_net.get_tv_loss()

            print(img_loss, img_loss0, psnr, sparsity_loss)

            if now_epoch > 1:
                self.train_kwargs['tv_loss_weight'] = 0.0

            gm.backward(loss)
            optim.step().clear_grad()

        loss_dict = {
            'img_loss': img_loss,
            # 'img_loss0': img_loss0,
            # 'psnr': psnr,
            # 'psnr0': psnr0,
            # 'sparsity_loss': sparsity_loss,
            # 'tv_loss': tv_loss
        }
        return loss_dict

    def test_step(self, batchdata):
        # given rays, return img
        rays_o = mge.tensor(batchdata['rays_o'][0], dtype=np.float32) # [H, W, 3]
        rays_d = mge.tensor(batchdata['rays_d'][0], dtype=np.float32) # [H, W, 3]

        res_img = F.zeros_like(rays_o)
        
        H,W,_ = res_img.shape

        num_cols = self.test_kwargs['num_cols']

        with Bar('rendering', max=H // num_cols + (H % num_cols > 0)) as bar:
            for i in range(H // num_cols):
                rays_o_block = rays_o[i*num_cols : i*num_cols + num_cols, :, :].reshape(-1, 3)
                rays_d_block = rays_d[i*num_cols : i*num_cols + num_cols, :, :].reshape(-1, 3)

                rgb, disp, acc, extras = render(rays_o_block, rays_d_block, coarse_net = self.coarse_net,
                fine_net = self.fine_net, **self.test_kwargs)

                res_img [i*num_cols : i*num_cols + num_cols, :, :] = rgb.reshape(num_cols, W, 3)
                bar.next()

            if H % num_cols > 0:
                rays_o_block = rays_o[H - H % num_cols : , :, :].reshape(-1, 3)
                rays_d_block = rays_d[H - H % num_cols : , :, :].reshape(-1, 3)

                rgb, disp, acc, extras = render(rays_o_block, rays_d_block, coarse_net = self.coarse_net,
                fine_net = self.fine_net, **self.test_kwargs)

                res_img [H - H % num_cols, :, :] = rgb.reshape(H % num_cols, W, 3)

        idx = batchdata['idx'][0]
        cv2.imwrite("{}.png".format(idx), (res_img.numpy()*255).astype(np.uint8))

        output_dict = {
            'rgb': res_img,
        }

        return output_dict

    def cal_for_eval(self):
        pass
