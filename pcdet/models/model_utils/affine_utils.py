import torch


def invert_affine_4x4(mats, eps=1e-8):
    """Invert batched 4x4 affine matrices without calling torch.inverse."""
    rot = mats[..., :3, :3]
    trans = mats[..., :3, 3]

    a00, a01, a02 = rot[..., 0, 0], rot[..., 0, 1], rot[..., 0, 2]
    a10, a11, a12 = rot[..., 1, 0], rot[..., 1, 1], rot[..., 1, 2]
    a20, a21, a22 = rot[..., 2, 0], rot[..., 2, 1], rot[..., 2, 2]

    c00 = a11 * a22 - a12 * a21
    c01 = -(a10 * a22 - a12 * a20)
    c02 = a10 * a21 - a11 * a20
    c10 = -(a01 * a22 - a02 * a21)
    c11 = a00 * a22 - a02 * a20
    c12 = -(a00 * a21 - a01 * a20)
    c20 = a01 * a12 - a02 * a11
    c21 = -(a00 * a12 - a02 * a10)
    c22 = a00 * a11 - a01 * a10

    det = a00 * c00 + a01 * c01 + a02 * c02
    det_safe = torch.where(
        det.abs() < eps,
        torch.where(det < 0, det.new_full(det.shape, -eps), det.new_full(det.shape, eps)),
        det,
    )

    inv_rot = torch.stack([
        torch.stack([c00, c10, c20], dim=-1),
        torch.stack([c01, c11, c21], dim=-1),
        torch.stack([c02, c12, c22], dim=-1),
    ], dim=-2) / det_safe[..., None, None]

    inv_trans = -(inv_rot @ trans.unsqueeze(-1)).squeeze(-1)
    inv = torch.zeros_like(mats)
    inv[..., :3, :3] = inv_rot
    inv[..., :3, 3] = inv_trans
    inv[..., 3, 3] = 1
    return inv
