from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from typing import Tuple


import math
import torch as tc

########################
# from https://github.com/MWod/DeeperHistReg/blob/eb92fed1b4ce22ca718b0d2d4cecbdd76b39486b/deeperhistreg/dhr_registration/dhr_building_blocks/bsplines.py
########################

### Linear Splines ###
def b_01(x):
    return 1 - x

def b_11(x):
    return x

linear_splines = {0: b_01, 1: b_11}

### Cubic Splines ###

def b_03(x):
    return (1-x)**3 / 6

def b_13(x):
    return (3*x**3 - 6*x**2 + 4) / 6

def b_23(x):
    return (-3*x**3 + 3*x**2 + 3*x + 1) / 6

def b_33(x):
    return x**3 / 6

cubic_splines = {0: b_03, 1: b_13, 2: b_23, 3: b_33}

######################


def b_splines(
    template : tc.Tensor,
    control_points : tc.Tensor, 
    spacing : tuple,
    splines_type : str='cubic',
    normalize : bool=True) -> tc.Tensor:
    """
    TODO
    """
    if splines_type == "cubic":
        splines = cubic_splines
        splines_length = 4
        adder = 0
    elif splines_type == 'linear':
        splines = linear_splines
        splines_length = 2
        adder = 1
    else:
        raise ValueError("Unsupported splines type.")

    batch_size, _, y_size, x_size = template.size()
    s_x, s_y = spacing
    grid_x, grid_y = tc.meshgrid(tc.arange(x_size), tc.arange(y_size), indexing='xy')
    grid_x, grid_y = grid_x.to(template.device), grid_y.to(template.device)
    s_x, s_y = s_x * x_size, s_y * y_size
    xss, yss = grid_x / s_x, grid_y / s_y
    f_xs, f_ys = tc.floor(xss), tc.floor(yss)
    diff_xs, diff_ys = xss - f_xs, yss - f_ys
    i, j = f_xs.type(tc.int64), f_ys.type(tc.int64)
    cached_splines = tc.zeros((batch_size, y_size, x_size, 2, splines_length), device=template.device, dtype=template.dtype)
    for z in range(splines_length):
        cached_splines[:, :, :, 0, z] = splines[z](diff_xs)
        cached_splines[:, :, :, 1, z] = splines[z](diff_ys)

    if normalize:
        n_control_points = control_points.clone()
        n_control_points[:, :, :, 0] = (control_points[:, :, :, 0] / 2.0 + 0.5) * x_size
        n_control_points[:, :, :, 1] = (control_points[:, :, :, 1] / 2.0 + 0.5) * y_size

    deformation_field = tc.zeros((batch_size, y_size, x_size, 2), device=template.device, dtype=template.dtype)
    for m in range(splines_length):
        inner_result = tc.zeros((batch_size, y_size, x_size, 2), device=template.device, dtype=template.dtype)
        for n in range(splines_length):
            diff = cached_splines[:, :, :, 1, n].unsqueeze(3).repeat(1, 1, 1, 2)
            y_index, x_index = j+n+adder, i+m+adder
            t_inner = n_control_points[:, y_index, x_index, :]
            t = diff*t_inner
            inner_result += t
        diff = cached_splines[:, :, :, 0, m].unsqueeze(3).repeat(1, 1, 1, 2)
        t = diff*inner_result
        deformation_field += t
        
    displacement_field = deformation_field - tc.stack((grid_x, grid_y), dim=2)
    displacement_field[:, :, :, 0] = displacement_field[:, :, :, 0] / x_size * 2.0
    displacement_field[:, :, :, 1] = displacement_field[:, :, :, 1] / y_size * 2.0
    return displacement_field

def create_control_points(
    template : tc.Tensor,
    spacing : tuple,
    splines_type : str='cubic',
    normalize : bool=True) -> tc.Tensor:
    """
    TODO
    """
    if splines_type == 'cubic':
        adder = 3
    elif splines_type == 'linear':
        adder = 2
    else:
        raise ValueError("Unsupported splines type.")
    batch_size, _, y_size, x_size = template.size()
    s_x, s_y = spacing
    s_x, s_y = s_x * x_size, s_y * y_size
    no_px = math.ceil((x_size+1) / s_x) + adder
    no_py = math.ceil((y_size+1) / s_y) + adder
    xs = tc.arange(-s_x, (no_px-1)*s_x, s_x, device=template.device)
    ys = tc.arange(-s_y, (no_py-1)*s_y, s_y, device=template.device)
    cp_x, cp_y = tc.meshgrid(xs, ys, indexing='xy')   
    control_points = tc.zeros(batch_size, cp_x.size(0), cp_y.size(1), 2).type_as(template)
    control_points[:, :, :, 0] = cp_x
    control_points[:, :, :, 1] = cp_y
    if normalize:
        control_points[:, :, :, 0] = ((control_points[:, :, :, 0] / x_size) - 0.5) * 2.0
        control_points[:, :, :, 1] = ((control_points[:, :, :, 1] / y_size) - 0.5) * 2.0
    return control_points

def tfrs_to_grid_M(tensor_tfrs: Tensor_tfrs|Tensor_trs, device: torch.device=None) -> Tensor_grid_M:
    """Converts TFRS parameters to an affine transformation matrix.

    Combines Translation, Flip, Rotation, and Scale parameters into a single
    affine transformation matrix in the order T @ F @ R @ S.

    Args:
        tensor_tfrs: A tensor containing either:
            - [angle, tx, ty, sx, sy] for TRS transform
            - [angle, tx, ty, sx, sy, fx, fy] for TFRS transform
        device: Optional torch device to place the output tensor on.

    Returns:
        A tensor of shape [2, 3] representing the affine transformation matrix.
    """
    
    theta,tx,ty,sx,sy = tensor_tfrs[0:5]
    fx, fy = torch.ones(1)[0], torch.ones(1)[0]
    if device is None:
        device = tensor_tfrs.device

    if tensor_tfrs.size(0) == 7:
        fx, fy = tensor_tfrs[5:]
        
    fx, fy = fx.to(device), fy.to(device)
    zeros = torch.zeros(1)[0].to(device)
    ones = torch.ones(1)[0].to(device)
        

    TF = torch.stack([
        torch.stack([fx, zeros, tx], dim=-1),
        torch.stack([zeros, fy, ty], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ], dim=0).to(device)

    R = torch.stack([
        torch.stack([torch.cos(theta), -torch.sin(theta), zeros], dim=-1),
        torch.stack([torch.sin(theta), torch.cos(theta), zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ]).to(device)
    
    S = torch.stack([
        torch.stack([1.0/torch.exp(sx), zeros, zeros], dim=-1),
        torch.stack([zeros, 1.0/torch.exp(sy), zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ]).to(device)
    
    grid_M = TF @ R @ S
    return grid_M[0:2, :]

def resample_displacement_field_to_size(
        displacement_field: Tensor_disp_2d,
        new_size: torch.Tensor, 
        mode: str='bilinear',
        align_corners: bool=False
    ) -> Tensor_disp_2d:
    """Resamples a displacement field to a new size.

    Args:
        displacement_field: A tensor of shape (N, H, W, 2) representing the displacement field.
        new_size: Target size for resampling.
        mode: Interpolation mode ('bilinear' by default).

    Returns:
        A tensor of shape (N, new_H, new_W, 2) representing the resampled displacement field.
    """
    return F.interpolate(displacement_field.permute(0, 3, 1, 2), size=new_size, mode=mode, align_corners=align_corners).permute(0, 2, 3, 1)

def smooth_displacement_with_kernel(disp, kernel):
    """Smooths a displacement field using a pre-computed kernel.

    Args:
        disp: A tensor of shape (N, H, W, 2) representing the displacement field.
        kernel: A tensor of shape (1, 1, k, k) representing the pre-computed smoothing kernel.

    Returns:
        A tensor of shape (N, H, W, 2) representing the smoothed displacement field.
    """
    disp = disp.permute(0, 3, 1, 2)
    disp_x = F.conv2d(disp[:, 0:1], kernel, padding=kernel.size(-1) // 2)
    disp_y = F.conv2d(disp[:, 1:2], kernel, padding=kernel.size(-1) // 2)
    smoothed_disp = torch.cat([disp_x, disp_y], dim=1).permute(0, 2, 3, 1)
    return smoothed_disp

class InverseD14Error(RuntimeError):
    """Continuous point geometry failed; diagnostics are retained, never clipped."""
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        codes, counts = torch.unique(diagnostics['status'].detach().cpu(), return_counts=True)
        super().__init__('continuous_d14 inverse rejected: ' + str(dict(zip(codes.tolist(), counts.tolist()))))

def sample_d14(d14, v):
    """Real bilinear value and analytic local Jacobian at normalized target v."""
    if d14.ndim != 4 or d14.shape[0] != 1 or d14.shape[-1] != 2 or min(d14.shape[1:3]) < 2:
        raise ValueError('D14 must be [1,H>=2,W>=2,2]')
    eps = torch.finfo(v.dtype).eps * 8
    if not torch.isfinite(v).all() or (v.abs() > 1 + eps).any():
        raise ValueError('Cannot sample D14 outside its certified domain')
    h, w = d14.shape[1:3]
    t = (v + 1) * v.new_tensor([(w-1)/2, (h-1)/2])
    # Only cell indices are bounded (last node uses the last cell); v is not clipped.
    ix = t[:, 0].floor().long().clamp(0, w-2)
    iy = t[:, 1].floor().long().clamp(0, h-2)
    x = (t[:, 0] - ix).unsqueeze(1); y = (t[:, 1] - iy).unsqueeze(1)
    a, b = d14[0, iy, ix], d14[0, iy, ix+1]
    c, d = d14[0, iy+1, ix], d14[0, iy+1, ix+1]
    value = a*(1-x)*(1-y) + b*x*(1-y) + c*(1-x)*y + d*x*y
    dx = ((b-a)*(1-y) + (d-c)*y) * ((w-1)/2)
    dy = ((c-a)*(1-x) + (d-b)*x) * ((h-1)/2)
    return value, torch.stack([dx, dy], dim=-1)

def certify_d14(d14, margin=1e-6, condition_limit=1e6):
    """Sufficient injectivity certificate on [-1,1]^2, not a scientific penalty.

    D(v)=v+E(v); checking all four one-sided cell-corner Jacobians bounds
    Lip(E)<1 over the convex domain, hence D is globally one-to-one there.
    This deliberately rejects fields that might be invertible by a weaker test.
    """
    with torch.no_grad():
        if d14.ndim != 4 or d14.shape[0] != 1 or d14.shape[-1] != 2 or min(d14.shape[1:3]) < 2:
            raise ValueError('D14 must be [1,H>=2,W>=2,2]')
        if not torch.isfinite(d14).all():
            return dict(accepted=False, reason='NONFINITE_FIELD', max_norm_J_minus_I=float('inf'), max_condition=float('inf'))
        h,w=d14.shape[1:3];g=d14[0]
        a,b,c,d=g[:-1,:-1],g[:-1,1:],g[1:,:-1],g[1:,1:]
        dx0,dx1=(b-a)*((w-1)/2),(d-c)*((w-1)/2)
        dy0,dy1=(c-a)*((h-1)/2),(d-b)*((h-1)/2)
        js=torch.stack([torch.stack([dx,dy],-1) for dx,dy in [(dx0,dy0),(dx0,dy1),(dx1,dy0),(dx1,dy1)]])
        bound=float(torch.linalg.matrix_norm(js-torch.eye(2,device=g.device,dtype=g.dtype),ord=2).max())
        condition=float(torch.linalg.cond(js).max())
        return dict(accepted=bound < 1-margin and condition <= condition_limit,
                    reason='CERTIFIED_CONTRACTION_PERTURBATION' if bound < 1-margin and condition <= condition_limit else 'UNCERTIFIED_OR_NEAR_SINGULAR_FIELD',
                    max_norm_J_minus_I=bound,max_condition=condition)

class _ImplicitD14Inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, points, roots, valid):
        ctx.save_for_backward(grid, roots, valid)
        return torch.where(valid[:,None], (roots+1)/2, torch.full_like(roots,float('nan')))

    @staticmethod
    def backward(ctx, grad_output):
        grid, roots, valid = ctx.saved_tensors
        grad_grid=torch.zeros_like(grid);grad_points=torch.zeros_like(roots)
        if valid.any():
            with torch.enable_grad():
                local_grid=grid.detach().requires_grad_(True)
                values,j=sample_d14(local_grid, roots[valid].detach())
                # y=(v+1)/2, D(v,theta)=2u-1: dy/du=J^-1; dy/dtheta=-.5 J^-1 D_theta.
                adj=torch.linalg.solve(j.detach().transpose(-1,-2),grad_output[valid].unsqueeze(-1)).squeeze(-1)
                grad_points[valid]=adj
                grad_grid=torch.autograd.grad(values,local_grid,grad_outputs=-.5*adj)[0]
        return grad_grid, grad_points, None, None

def inverse_d14(d14, points, *, strict=True, max_iterations=40, tolerance=None,
                certificate_margin=1e-6, condition_limit=1e6):
    """Invert certified D14, returning [0,1] points and per-point diagnostics.

    Status: 0 OK; 1 nonfinite input; 2 uncertified field; 3 outside domain;
    4 ill-conditioned iterate; 5 no convergence. Failed outputs are NaN.
    No nearest-grid fallback, border clipping, omitted rows or root selection.
    """
    if points.ndim != 2 or points.shape[1] != 2 or points.dtype != d14.dtype or points.device != d14.device:
        raise ValueError('Points must be Nx2 on the same device/dtype as D14')
    tolerance = tolerance if tolerance is not None else (1e-9 if points.dtype==torch.float64 else 2e-5)
    certificate=certify_d14(d14,certificate_margin,condition_limit)
    with torch.no_grad():
        n=len(points);status=torch.full((n,),5,dtype=torch.int64,device=points.device)
        residual=points.new_full((n,),float('inf'));condition=residual.clone()
        iterations=torch.zeros(n,dtype=torch.int64,device=points.device)
        target=2*points.detach()-1;roots=target.clone()
        finite=torch.isfinite(target).all(1);inside=(target.abs()<=1).all(1)
        status[~finite]=1;status[finite & ~inside]=3
        active=finite & inside
        if not certificate['accepted']:
            status[active]=2;active[:]=False
        for iteration in range(max_iterations):
            if not active.any():break
            ids=torch.nonzero(active).flatten()
            value,j=sample_d14(d14,roots[ids]);res=value-target[ids]
            residual[ids]=res.abs().amax(1);condition[ids]=torch.linalg.cond(j);iterations[ids]=iteration+1
            bad=~torch.isfinite(condition[ids]) | (condition[ids]>condition_limit)
            status[ids[bad]]=4;active[ids[bad]]=False
            done=(residual[ids]<=tolerance)&~bad
            status[ids[done]]=0;active[ids[done]]=False
            todo=~bad&~done
            if todo.any():
                chosen=ids[todo];step=torch.linalg.solve(j[todo],res[todo].unsqueeze(-1)).squeeze(-1)
                nxt=roots[chosen]-step
                out=~torch.isfinite(nxt).all(1)|(nxt.abs()>1).any(1)
                status[chosen[out]]=3;active[chosen[out]]=False
                roots[chosen[~out]]=nxt[~out]
        valid=status==0
        safe_roots=torch.where(valid[:,None],roots,torch.zeros_like(roots))
    info=dict(status=status,residual=residual,condition=condition,iterations=iterations,certificate=certificate)
    if strict and not valid.all():raise InverseD14Error(info)
    return _ImplicitD14Inverse.apply(d14,points,safe_roots,valid),info


def load_weights(path):
    return torch.load(path, map_location="cpu", weights_only=True)

def align_raw_coordinates(weights, layer, raw_xy):
    import numpy as np
    state = weights["modules"][layer]
    raw = np.asarray(raw_xy, dtype=np.float64)
    coarse = np.c_[raw, np.ones(len(raw))] @ weights["raw_to_coarse"][layer].numpy()
    control = state["displacement_field"]
    if state["disp_type"] == "bspline":
        disp = b_splines(control.permute(0,3,1,2), control, state["cp_spacing"], splines_type=state["splines_type"], normalize=True)
    else:
        disp = smooth_displacement_with_kernel(control, state["kernel"])
    matrix = tfrs_to_grid_M(state["tensor_trs"], device=torch.device("cpu"))
    h,w = state["tensor_size"]
    grid = F.affine_grid(matrix.unsqueeze(0), [1,1,h,w], align_corners=True)
    field = resample_displacement_field_to_size(grid, disp.shape[1:3], align_corners=True) + disp
    points = torch.as_tensor(coarse, dtype=control.dtype) / weights["max_size"]
    with torch.no_grad():
        aligned, diagnostics = inverse_d14(field, points)
    assert diagnostics["certificate"]["accepted"] and bool((diagnostics["status"] == 0).all())
    return aligned.numpy() * weights["max_size"], diagnostics
