"""Experimental two-pass backward; deliberately not selected by autograd dispatch."""
import torch


def softplus_split_backward(q,k,v,out,dout,*,window=-1,alpha=1.,scale=None,
                            qk_threads=256,qk_stages=1,dv_threads=256):
    """Return complete dQ/dK/dV using separate CuTe kernels.

    Includes each component's setup and output conversion. This exploratory
    implementation retains unused zero-gradient outputs/workspaces from the
    common driver, so timings must include both complete component calls.
    """
    from flash_attn_4.interface import _flash_attn_bwd
    if q.ndim!=4 or k.ndim!=4 or v.ndim!=4 or q.shape[2]!=k.shape[2] or k.shape[2]!=v.shape[2]:
        raise ValueError('split backward requires dense MHA inputs')
    lse=torch.empty(q.shape[0],q.shape[2],q.shape[1],device=q.device,dtype=torch.float32)
    kw=dict(causal=window<0,window_size_left=None if window<0 else window,
            window_size_right=None if window<0 else 0,attn_kind='softplus',
            softplus_alpha=alpha,softmax_scale=q.shape[-1]**-.5 if scale is None else scale)
    dq,dk,_=_flash_attn_bwd(q,k,v,out,dout,lse,**kw,softplus_bwd_component='qk',
        sm120_bwd_tile=(64,64,qk_stages,1),sm120_bwd_num_threads=qk_threads)
    _,_,dv=_flash_attn_bwd(q,k,v,out,dout,lse,**kw,softplus_bwd_component='v',
        sm120_bwd_tile=(64,64,1,1),sm120_bwd_num_threads=dv_threads)
    return dq,dk,dv
