import numpy as np
from numpy import int64


def bilinear_interp_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This is the vectorized implementation of bilinear interpolation.
    - a is a ND array with shape [N, H1, W1, C], dtype = int64
    - b is a ND array with shape [N, H2, W2, 2], dtype = float64
    - return a ND array with shape [N, H2, W2, C], dtype = int64
    """
    # get axis size from ndarray shape
    N, H1, W1, C = a.shape
    N1, H2, W2, _ = b.shape
    assert N == N1

    # TODO: Implement vectorized bilinear interpolation
    # res = np.empty((N, H2, W2, C), dtype=int64)
    posit0=np.arange(C).repeat(N*H2*W2).reshape(C,N*H2*W2).transpose(1,0).reshape(C*N*H2*W2)
    posit1=np.arange(N).repeat(C*W2*H2).reshape(C*N*H2*W2)
    x=b[:,:,:,0].repeat(C).reshape(C*N*H2*W2)
    y=b[:,:,:,1].repeat(C).reshape(C*N*H2*W2)
    x_down=np.floor(x).astype(int)
    y_down=np.floor(y).astype(int)
    x_up=np.ceil(x).astype(int)
    y_up=np.ceil(y).astype(int)
    # x, y = b[n, i, j]
    # x_idx, y_idx = int(np.floor(x)), int(np.floor(y))
    # _x, _y = x - x_idx, y - y_idx
    res = a[posit1, x_down, y_down,posit0] * ((x_up- x) * (y_up - y)) + a[posit1, x_up, y_down,posit0] * ((x-x_down) * (y_up - y)) + \
          a[posit1, x_down, y_up,posit0] * ((x_up - x) * (y-y_down)) + a[posit1, x_up, y_up,posit0] * ((x-x_down) * (y-y_down))
    res=res.astype(int).reshape(N, H2, W2, C)
    return res
        
    
    
    