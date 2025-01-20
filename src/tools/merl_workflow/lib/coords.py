import numpy as np
import torch

# --- built in ---
import sys
from pathlib import Path

# setting the repo directory to sys path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from tools.merl_workflow.lib.utils import inner_product


def rotate_vector(v, axis, angle):
    return v * np.cos(angle) + axis * dot(axis, v) * (1 - np.cos(angle)) + cross(axis, v) * np.sin(angle)


def rotate_vector_pt(v, axis, angle):
    return v * angle.cos() + axis * inner_product(axis, v, keepdim=True) * (1 - angle.cos()) + torch.cross(axis, v, dim=-1) * angle.sin()


def io_to_hd(wi, wo):
    # compute halfway vector
    half = normalize(*(wi + wo))
    r_h, theta_h, phi_h = xyz2sph(*half)

    # compute diff vector
    bi_normal = np.tile([0.0, 1.0, 0.0], (wi.shape[1], 1)).T
    normal = np.tile([0.0, 0.0, 1.0], (wi.shape[1], 1)).T
    tmp = rotate_vector(wi, normal, -phi_h)
    diff = rotate_vector(tmp, bi_normal, -theta_h)
    return half, diff


def io_to_hd_pt(wi, wo):
    assert wi.shape == wo.shape
    # compute halfway vector
    half = torch.nn.functional.normalize(wi + wo, dim=-1)
    _, theta_h, phi_h = xyz2sph_pt(half)

    # compute diff vector
    bi_normal = torch.tile(torch.tensor([0.0, 1.0, 0.0]), (*wi.shape[:-1], 1)).to(wi.device)
    normal = torch.tile(torch.tensor([0.0, 0.0, 1.0]), (*wi.shape[:-1], 1)).to(wi.device)
    tmp = rotate_vector_pt(wi, normal, -phi_h.unsqueeze(-1))
    diff = rotate_vector_pt(tmp, bi_normal, -theta_h.unsqueeze(-1))
    return half, diff


def hd_to_io(half, diff):
    r_h, theta_h, phi_h = xyz2sph(*half)

    y_axis = np.tile([0.0, 1.0, 0.0], (half.shape[1], 1)).T
    z_axis = np.tile([0.0, 0.0, 1.0], (half.shape[1], 1)).T

    tmp = rotate_vector(diff, y_axis, theta_h)
    wi = normalize(*rotate_vector(tmp, z_axis, phi_h))
    wo = normalize(*(2 * dot(wi, half) * half - wi))
    return wi, wo


def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def cross(v1, v2):
    return np.cross(v1.T, v2.T).T


def xyz2sph(x, y, z):
    r2_xy = x ** 2 + y ** 2
    r = np.sqrt(r2_xy + z ** 2)
    theta = np.arctan2(np.sqrt(r2_xy), z)
    phi = np.arctan2(y, x)
    return np.array([r, theta, phi])


def xyz2sph_pt(v):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    r2_xy = x ** 2 + y ** 2
    r = (r2_xy + z ** 2).sqrt()
    theta = torch.atan2(r2_xy.sqrt(), z)
    phi = torch.atan2(y, x)
    return r, theta, phi


def normalize(x, y, z):
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    norm = np.where(norm == 0, np.inf, norm)
    return np.array([x, y, z]) / norm


def sph2xyz(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


# isotropic material, so phi_h does not matter
# assumes phi_h=0 (therefore hy is fixed) and both norms=1
def rangles_to_rvectors(theta_h, theta_d, phi_d):
    hx = np.sin(theta_h) * np.cos(0.0)
    hy = np.sin(theta_h) * np.sin(0.0)
    hz = np.cos(theta_h)
    dx = np.sin(theta_d) * np.cos(phi_d)
    dy = np.sin(theta_d) * np.sin(phi_d)
    dz = np.cos(theta_d)
    return np.array([hx, hy, hz, dx, dy, dz])

# x is of shape [B, 3] for theta_h, theta_d and phi_d
def rangles_to_rvectors_pt(x):
    theta_h = x[:, 0, None]
    theta_d = x[:, 1, None]
    phi_d = x[:, 2, None]
    hx = torch.sin(theta_h) * torch.cos(torch.tensor(0.0))
    hy = torch.sin(theta_h) * torch.sin(torch.tensor(0.0))
    hz = torch.cos(theta_h)
    dx = torch.sin(theta_d) * torch.cos(phi_d)
    dy = torch.sin(theta_d) * torch.sin(phi_d)
    dz = torch.cos(theta_d)
    return torch.cat([hx, hy, hz, dx, dy, dz], dim=-1)


def rsph_to_rvectors(half_sph, diff_sph):
    hx, hy, hz = sph2xyz(*half_sph)
    dx, dy, dz = sph2xyz(*diff_sph)
    return np.array([hx, hy, hz, dx, dy, dz])


def rvectors_to_rsph(hx, hy, hz, dx, dy, dz):
    half_sph = xyz2sph(hx, hy, hz)
    diff_sph = xyz2sph(dx, dy, dz)
    return half_sph, diff_sph


def rvectors_to_rangles(hx, hy, hz, dx, dy, dz):
    theta_h = np.arctan2(np.sqrt(hx ** 2 + hy ** 2), hz)
    theta_d = np.arctan2(np.sqrt(dx ** 2 + dy ** 2), dz)
    phi_d = np.arctan2(dy, dx)
    return np.array([theta_h, theta_d, phi_d])

# x is of shape [B, 6] for hx, hy, hz, dx, dy and dz
def rvectors_to_rangles_pt(x):
    hx = x[:, 0, None]
    hy = x[:, 1, None]
    hz = x[:, 2, None]
    dx = x[:, 3, None]
    dy = x[:, 4, None]
    dz = x[:, 5, None]
    theta_h = torch.atan2((hx ** 2 + hy ** 2).sqrt(), hz)
    theta_d = torch.atan2((dx ** 2 + dy ** 2).sqrt(), dz)
    phi_d = torch.atan2(dy, dx)
    return torch.cat([theta_h, theta_d, phi_d], dim=-1)


if __name__ == '__main__':
    a, b, c = xyz2sph(*np.zeros((3, 10)))
    print(a.shape)
