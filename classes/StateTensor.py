import torch as pt
from util.functions import add_default_arg
from util.quaternion import quaternion_to_matrix

class StateTensor(pt.Tensor):
    @classmethod
    def from_tensor(cls, tensor: pt.Tensor) -> "StateTensor":
        if not isinstance(tensor, pt.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if tensor.shape[-1] != 13:
            raise ValueError("StateTensor requires last dimension = 13")
        
        tensor.__class__ = StateTensor
        return tensor

        return pt.Tensor._make_subclass(cls, tensor, require_grad=tensor.requires_grad)
    
    
    def __new__(cls, state_vec=None, pos=None, vel=None, quat=None, angvel=None, **kwargs):
        # Position and velocity are world relative
        # Angular velocity is body relative

        add_default_arg(kwargs, 'dtype', pt.float32)
        add_default_arg(kwargs, 'device', None)
        requires_grad = kwargs.get('requires_grad', False)
        kwargs.pop('requires_grad', None)
        
        if state_vec is None:
            arg_not_none = [_ for _ in (pos, vel, quat, angvel) if _ is not None]
            if len(arg_not_none) == 0:
                state_vec = pt.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], **kwargs)
            else:
                batch = pt.as_tensor(arg_not_none[0]).shape[:-1]
                state_vec = pt.zeros((*batch, 13), **kwargs)
                state_vec[..., 6] = 1
        else:
            state_vec = pt.as_tensor(state_vec, **kwargs)

        obj = pt.Tensor._make_subclass(cls, state_vec, require_grad=requires_grad)
        if pos is not None:
            obj[..., 0:3] = pt.as_tensor(pos, **kwargs)
        if vel is not None:
            obj[..., 3:6] = pt.as_tensor(vel, **kwargs)
        if quat is not None:
            obj[..., 6:10] = pt.as_tensor(quat, **kwargs)
        if angvel is not None:
            obj[..., 10:13] = pt.as_tensor(angvel, **kwargs)
        return obj

    @property
    def pos(self) -> pt.Tensor:
        return self[..., 0:3]

    @property
    def vel(self) -> pt.Tensor:
        return self[..., 3:6]

    @property
    def quat(self) -> pt.Tensor:
        return self[..., 6:10]

    @property
    def angvel(self) -> pt.Tensor:
        return self[..., 10:13]
    
    @property
    def angvel_mat(self) -> pt.Tensor:
        wx, wy, wz      = self[..., 10], self[..., 11], self[..., 12]
        mat             = pt.zeros((*self.shape[:-1], 3, 3), device=self.device, dtype=self.dtype)
        mat[..., 0, 1]  = -wz
        mat[..., 0, 2]  =  wy
        mat[..., 1, 0]  =  wz
        mat[..., 1, 2]  = -wx
        mat[..., 2, 0]  = -wy
        mat[..., 2, 1]  =  wx
        return mat
    
    @property
    def rot_mat(self) -> pt.Tensor:
        return quaternion_to_matrix(self[..., 6:10])
    
    @property
    def batch_size(self) -> tuple:
        return self.shape[:-1]