import megengine as mge
import megengine.functional as F
import megengine.module as M


class Lie:

    @classmethod
    def se3_exp_map(cls, log_transform: mge.Tensor, eps: float = 1e-4):
        """
        se3 -> SE3

        Args:
            log_transform: shape of (N, 6)
        Returns:
            transform: shape of (N, 4, 4)
        """
        assert (
            len(log_transform.shape) == 2 and log_transform.shape[-1] == 6
        ), "Input tensor shape has to be Nx6."
        N = log_transform.shape[0]

        log_trans, log_rot =  F.split(log_transform, 2, axis=1)
        
        skews = cls.hat(log_rot)
        skews_square = F.matmul(skews, skews)
        rot_angles = F.sqrt(F.clip((log_rot**2).sum(-1), lower=eps)) # b
        rot_angles_inv = 1.0 / rot_angles
        I = F.eye(3)[None] # [1,3,3]
        
        A = (F.sin(rot_angles) * rot_angles_inv)[:, None, None] # [B,1,1]
        B = ((1.0 - F.cos(rot_angles)) * (rot_angles_inv**2))[:, None, None]
        C = ((rot_angles - F.sin(rot_angles)) * (rot_angles_inv**3))[:, None, None]

        R = I + A * skews + B * skews_square
        V = I + B * skews + C * skews_square
        T = V @ log_trans[..., None] # [B,3,3] @ [B,3,1]

        transform = F.broadcast_to(F.eye(4), (N,4,4))
        transform[..., :3, :4] = F.concat([R, T], axis=-1)

        return transform

    @classmethod
    def so3_exp_map(cls, log_rotation: mge.Tensor, eps: float = 1e-4):
        """
        so3 -> SO3

        Args:
            log_rotation: shape of (N, 3)
        Returns:
            R: shape of (N, 3, 3)
        """
        assert (
            len(log_rotation.shape) == 2 and log_rotation.shape[-1] == 3
        ), "Input tensor shape has to be Nx3."

        skews = cls.hat(log_rotation)
        skews_square = F.matmul(skews, skews)
        
        rot_angles = F.sqrt(F.clip((log_rotation**2).sum(-1), lower=eps)) 
        
        rot_angles_inv = 1.0 / rot_angles
        I = F.eye(3)[None] # [1,3,3]

        A = (F.sin(rot_angles) * rot_angles_inv)[:, None, None]
        B = ((1.0 - F.cos(rot_angles)) * (rot_angles_inv**2))[:, None, None]

        R = I + A * skews + B * skews_square

        return R

    @classmethod
    def se3_log_map(
        cls, transform: mge.Tensor, eps: float = 1e-4, cos_bound: float = 1e-4
    ):
        """
        SE3 -> se3

        Args:
            transform: shape of (N, 4, 4)
        Returns:
            : shape of (N, 6)
        """
        assert (
            len(transform.shape) == 3
            and transform.shape[1] == 4
            and transform.shape[2] == 4
        ), "Input tensor shape has to be Nx4x4."
        N = transform.shape[0]

        assert transform[:, 3, :3].sum() == 0, "All elements of `transform[:, :3, 3]` should be 0."

        R = transform[:, :3, :3]
        log_rot = cls.so3_log_map(R, eps=eps, cos_bound=cos_bound)

        T = transform[:, :3, 3]

        rot_angles = F.sqrt(F.clip((log_rot**2).sum(-1), lower=eps)) 

        log_rot_hat = cls.hat(log_rot)
        log_rot_hat_square = F.matmul(log_rot_hat, log_rot_hat)

        I = F.eye(3)[None]

        A = ((1 - F.cos(rot_angles)) / (rot_angles**2))[:, None, None]
        B = ((rot_angles - F.sin(rot_angles)) / (rot_angles**3))[:, None, None]

        V = I + A * log_rot_hat + B * log_rot_hat_square
        log_trans = F.matmul(F.matinv(V), T[:, :, None])
        log_trans = log_trans[:, :, 0]
        return F.concat([log_trans, log_rot], axis=1)

    @classmethod
    def so3_log_map(cls, R: mge.Tensor, eps: float = 0.0001, cos_bound: float = 1e-4):
        """
        SO3 -> so3

        Args:
            R: shape of (N, 3, 3)
        Returns:
            log_rotation: shape of (N, 3)
        """
        assert len(R.shape) == 3 and R.shape[1] == 3 and R.shape[2] == 3
        N = R.shape[0]

        rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        assert F.sum((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)) == 0, "A matrix has trace outside valid range [-1-eps,3+eps]."

        phi_cos = (rot_trace - 1.0) * 0.5
        phi = F.acos(F.clip(phi_cos, -1 + cos_bound, 1 - cos_bound))
        phi_sin = F.sin(phi)

        # We want to avoid a tiny denominator of phi_factor = phi / (2.0 * phi_sin).
        # Hence, for phi_sin.abs() <= 0.5 * eps, we approximate phi_factor with
        # 2nd order Taylor expansion: phi_factor = 0.5 + (1.0 / 12) * phi**2
        phi_factor = F.zeros_like(phi)
        ok_denom = phi_sin.abs() > (0.5 * eps)
        phi_factor[~ok_denom] = 0.5 + (phi[~ok_denom] ** 2) * (1.0 / 12)
        phi_factor[ok_denom] = phi[ok_denom] / (2.0 * phi_sin[ok_denom])

        log_rotation_hat = phi_factor[:, None, None] * (R - R.transpose(0, 2, 1))

        log_rotation = cls.hat_inv(log_rotation_hat)

        return log_rotation

    @classmethod
    def hat(cls, v: mge.Tensor):
        assert len(v.shape) == 2 and v.shape[-1] == 3
        N, dim = v.shape
        h = F.zeros((N, 3, 3))

        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        h[:, 0, 1] = -z
        h[:, 0, 2] = y
        h[:, 1, 0] = z
        h[:, 1, 2] = -x
        h[:, 2, 0] = -y
        h[:, 2, 1] = x

        return h

    @classmethod
    def hat_inv(cls, h: mge.Tensor, hat_inv_skew_symmetric_tol: float = 1e-5):
        assert (
            len(h.shape) == 3 and h.shape[1] == 3 and h.shape[2] == 3
        ), "Input has to be a batch of 3x3 Tensors."

        ss_diff = F.abs(h + h.transpose(0, 2, 1)).max()

        assert (
            float(ss_diff) < hat_inv_skew_symmetric_tol
        ), "One of input matrices is not skew-symmetric."

        x = h[:, 2, 1]
        y = h[:, 0, 2]
        z = h[:, 1, 0]
        v = F.stack((x,y,z), axis=1)

        return v


class ExtrinsicOptimizer(M.Module):
    def __init__(self, n_views):
        super().__init__()

        self.se3_refinement = M.Embedding(n_views, 6, initial_weight=F.zeros((n_views, 6)))
        
    def forward(self, extrinsics: mge.Tensor, indices: mge.tensor):
        """
        Optimize the camera extrinsic of the given index.

        Args:
            extrinsics: Tensor, RT matrix of shape (B, 4, 4).
            indices: Tensor of shape (B, )
        """
        refinement = Lie.se3_exp_map(self.se3_refinement(indices))
        return refinement @ extrinsics
