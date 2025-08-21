import torch as pt

def arclen(X: pt.tensor):
    return pt.cumsum(pt.norm((X[1:,0:2] - X[:-1,0:2]), dim=-1), dim=-1)