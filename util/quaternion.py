import torch as pt

# necessary functions
def quaternion_derivative(q, omega):
    qw, qx, qy, qz = q[...,0], q[...,1], q[...,2], q[...,3]
    ox, oy, oz = omega[...,0], omega[...,1], omega[...,2]
    dq = 0.5 * pt.stack([
        -qx*ox - qy*oy - qz*oz,
         qw*ox + qy*oz - qz*oy,
         qw*oy + qz*ox - qx*oz,
         qw*oz + qx*oy - qy*ox
    ], dim=-1)
    return dq

# grad tracking method to get the rotation matrix from quat
def quaternion_to_matrix(q):
    # q: (..., 4) tensor with [w, x, y, z]
    w, x, y, z = q.unbind(-1)

    B = q.shape[:-1]  # batch shape if any

    # Elements of the rotation matrix
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rot = pt.stack([
        ww + xx - yy - zz, 2*(xy - wz),     2*(xz + wy),
        2*(xy + wz),     ww - xx + yy - zz, 2*(yz - wx),
        2*(xz - wy),     2*(yz + wx),     ww - xx - yy + zz
    ], dim=-1).reshape(*B, 3, 3)
    return rot