"""
3D → 2D 投影 & 遮挡检测。

坐标约定（Habitat-Sim / OpenGL）：
  - 相机局部坐标：X 向右、Y 向上、Z 朝屏幕外（相机看向 -Z）。
  - 旋转矩阵 R 表示 local-to-world；投影时取 R^T 做 world-to-local。
  - 深度传感器返回 z-depth（沿观看方向距离，单位米）。
"""
import math
from typing import Tuple

import numpy as np
import quaternion as quat_mod


def quat_to_rotmat(q) -> np.ndarray:
    """numpy-quaternion / 四元数 → 3×3 旋转矩阵 (local → world)。"""
    if isinstance(q, quat_mod.quaternion):
        w, x, y, z = q.w, q.x, q.y, q.z
    elif hasattr(q, "scalar"):
        w = float(q.scalar)
        x, y, z = (float(q.vector[i]) for i in range(3))
    else:
        x, y, z, w = q[0], q[1], q[2], q[3]
    n = math.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compute_intrinsics(
    hfov_deg: float, width: int, height: int
) -> Tuple[float, float, float, float]:
    """返回 (fx, fy, cx, cy)。"""
    hfov_rad = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def project_points(
    pts_world: np.ndarray,
    cam_pos: np.ndarray,
    R_local2world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将世界坐标点投影到像素平面。
    为了便于理解和debug, 用图的方式描述一下世界坐标系和相机坐标系
        
        Y (上)
         |
         |
         +------ X (右)
        /
       /
      Z (从屏幕朝你戳出来，指向你背后)
      
    """
    pts = np.asarray(pts_world, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]

    R_w2l = R_local2world.T     # 世界坐标系到相机坐标系的旋转矩阵
    diff = pts - cam_pos[np.newaxis, :]     # 从相机到点的向量（世界坐标系下）
    pts_cam = (R_w2l @ diff.T).T    # (N, 3) 将世界坐标系中的点转换到相机坐标系中，得到相机坐标系中的点

    z = -pts_cam[:, 2]  # 相机坐标系中的z坐标，负号的原因是相机坐标系中的z轴指向屏幕外，因此正前方是-z方向
    in_front = z > 1e-4  # 判断点是否在相机前方，还是上面坐标系的缘故

    z_safe = np.where(in_front, z, 1.0)
    u = fx * (pts_cam[:, 0] / z_safe) + cx
    v = -fy * (pts_cam[:, 1] / z_safe) + cy

    uv = np.stack([u, v], axis=-1)
    return uv, z, in_front


def check_visibility(
    uv: np.ndarray,
    depths: np.ndarray,
    in_front: np.ndarray,
    depth_buffer: np.ndarray,
    width: int,
    height: int,
    tau: float,
) -> np.ndarray:
    """
    对每个投影点判断是否可见（在图像内 + 在相机前方 + 未被遮挡）。

    返回 bool mask (N,)。
    """
    N = uv.shape[0]
    vis = np.zeros(N, dtype=bool)

    u_int = np.round(uv[:, 0]).astype(np.int64)
    v_int = np.round(uv[:, 1]).astype(np.int64)

    in_bounds = (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height)
    candidate = in_front & in_bounds

    idx = np.where(candidate)[0]
    if idx.size == 0:
        return vis

    d_buf = depth_buffer[v_int[idx], u_int[idx]]
    valid_depth = (d_buf > 0) & np.isfinite(d_buf) # 一些corner case比如说渲染出现漏的地方可能 distance=inf
    not_occluded = depths[idx] <= (d_buf + tau)
    vis[idx] = valid_depth & not_occluded

    return vis
