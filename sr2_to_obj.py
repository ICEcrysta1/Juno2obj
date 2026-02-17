#!/usr/bin/env python3
"""
SimpleRockets 2 Fuselage to OBJ Converter
支持椭圆截面圆柱体、offset、长度变化、尖锐化和正方体变形
"""

import xml.etree.ElementTree as ET
import numpy as np
import math
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FuselageParams:
    """Fuselage参数类"""
    length: float = 5.0           # 长度
    radius_x: float = 2.0         # 椭圆短边半径 (X轴)
    radius_z: float = 4.0         # 椭圆长边半径 (Z轴，朝上)
    offset_x: float = 0.0         # X方向偏移
    offset_y: float = 0.0         # Y方向偏移 (沿圆柱长度方向)
    offset_z: float = 0.0         # Z方向偏移
    top_scale_x: float = 1.0      # 顶部X缩放
    top_scale_z: float = 1.0      # 顶部Z缩放
    bottom_scale_x: float = 1.0   # 底部X缩放
    bottom_scale_z: float = 1.0   # 底部Z缩放
    deformations: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 形变参数 (正方体化等)
    corner_radiuses: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # 圆角半径 (0=方, 1=圆)
    # clampDistances: 顶部和底部的挤压参数，格式: tx_neg,tx_pos,tz_neg,tz_pos,bx_neg,bx_pos,bz_neg,bz_pos
    # -1 = 不挤压(保持原状), 0 = 挤压到中心, 1 = 不挤压
    clamp_distances: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    # partScale: 整体额外缩放，格式: x,y,z，在旋转前应用
    part_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    @property
    def top_radius_x(self) -> float:
        return self.radius_x * self.top_scale_x
    
    @property
    def top_radius_z(self) -> float:
        return self.radius_z * self.top_scale_z
    
    @property
    def bottom_radius_x(self) -> float:
        return self.radius_x * self.bottom_scale_x
    
    @property
    def bottom_radius_z(self) -> float:
        return self.radius_z * self.bottom_scale_z


@dataclass
class Material:
    """材质类"""
    name: str
    color: Tuple[float, float, float]  # RGB 0-1
    metallic: float  # 0-1
    roughness: float  # 0-1 (游戏内0=粗糙,1=光滑,已转换)


class Mesh:
    """网格类，用于构建OBJ数据"""
    def __init__(self):
        self.vertices: List[Tuple[float, float, float]] = []
        self.normals: List[Tuple[float, float, float]] = []
        self.uvs: List[Tuple[float, float]] = []
        self.faces: List[Tuple[int, int, int, int, int, int, int, int, int, str]] = []  # v1,v2,v3, vt1,vt2,vt3, vn1,vn2,vn3, material_name
        self.current_material: str = ""
    
    def add_vertex(self, x: float, y: float, z: float) -> int:
        """添加顶点，返回顶点索引（从1开始）"""
        self.vertices.append((x, y, z))
        return len(self.vertices)
    
    def add_normal(self, nx: float, ny: float, nz: float) -> int:
        """添加法线，返回法线索引（从1开始）"""
        self.normals.append((nx, ny, nz))
        return len(self.normals)
    
    def add_uv(self, u: float, v: float) -> int:
        """添加UV坐标，返回UV索引（从1开始）"""
        self.uvs.append((u, v))
        return len(self.uvs)
    
    def set_material(self, material_name: str):
        """设置当前材质"""
        self.current_material = material_name
    
    def add_face(self, v1: int, v2: int, v3: int, 
                 vt1: int = 0, vt2: int = 0, vt3: int = 0,
                 vn1: int = 0, vn2: int = 0, vn3: int = 0,
                 material_name: str = ""):
        """添加三角面"""
        mat = material_name if material_name else self.current_material
        self.faces.append((v1, v2, v3, vt1, vt2, vt3, vn1, vn2, vn3, mat))
    
    def add_quad(self, v1: int, v2: int, v3: int, v4: int,
                 vt1: int = 0, vt2: int = 0, vt3: int = 0, vt4: int = 0,
                 vn1: int = 0, vn2: int = 0, vn3: int = 0, vn4: int = 0,
                 material_name: str = ""):
        """添加四边面（分解为两个三角面）"""
        # 四边形 v1-v2-v3-v4 -> 三角面 (v1,v2,v3) 和 (v1,v3,v4)
        mat = material_name if material_name else self.current_material
        self.add_face(v1, v2, v3, vt1, vt2, vt3, vn1, vn2, vn3, mat)
        self.add_face(v1, v3, v4, vt1, vt3, vt4, vn1, vn3, vn4, mat)
    
    def write_obj(self, filename: str, mtl_filename: str = ""):
        """写入OBJ文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# SimpleRockets 2 Fuselage to OBJ Export\n")
            f.write(f"# Vertices: {len(self.vertices)}\n")
            f.write(f"# Faces: {len(self.faces)}\n\n")
            
            # 引用材质文件
            if mtl_filename:
                f.write(f"mtllib {mtl_filename}\n\n")
            
            # 写入顶点
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # 写入UV坐标
            if self.uvs:
                f.write("\n")
                for vt in self.uvs:
                    f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
            
            # 写入法线
            if self.normals:
                f.write("\n")
                for vn in self.normals:
                    f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
            
            # 写入面，按材质分组
            f.write("\n")
            current_mat = None
            for face in self.faces:
                v1, v2, v3, vt1, vt2, vt3, vn1, vn2, vn3, mat_name = face
                
                # 材质变化时输出usemtl
                if mat_name and mat_name != current_mat:
                    f.write(f"\nusemtl {mat_name}\n")
                    current_mat = mat_name
                
                # 构建面的字符串
                if vn1 > 0 and vt1 > 0:
                    # 有顶点、纹理、法线
                    f.write(f"f {v1}/{vt1}/{vn1} {v2}/{vt2}/{vn2} {v3}/{vt3}/{vn3}\n")
                elif vn1 > 0:
                    # 有顶点、法线，无纹理
                    f.write(f"f {v1}//{vn1} {v2}//{vn2} {v3}//{vn3}\n")
                elif vt1 > 0:
                    # 有顶点、纹理，无法线
                    f.write(f"f {v1}/{vt1} {v2}/{vt2} {v3}/{vt3}\n")
                else:
                    # 只有顶点
                    f.write(f"f {v1} {v2} {v3}\n")


def parse_vector(text: str) -> Tuple[float, float, float]:
    """解析逗号分隔的向量"""
    parts = text.split(',')
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def parse_scale(text: str) -> Tuple[float, float]:
    """解析Fuselage的scale参数 (x,z)"""
    parts = text.split(',')
    return (float(parts[0]), float(parts[1]))


def rotation_matrix(euler_angles: Tuple[float, float, float]) -> np.ndarray:
    """
    从Unity欧拉角(度数)创建旋转矩阵
    Unity使用ZXY顺序: 先Z，再X，再Y
    """
    rx, ry, rz = [math.radians(a) for a in euler_angles]
    
    # ZXY旋转顺序 (Unity)
    cz, sz = math.cos(rz), math.sin(rz)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    
    # Z旋转
    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])
    
    # X旋转
    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])
    
    # Y旋转
    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])
    
    # ZXY顺序: Ry * Rx * Rz
    R = Ry @ Rx @ Rz
    return R


def apply_clamp_to_ring(coords: List[Tuple[float, float, float]],
                       clamp_x_neg: float, clamp_x_pos: float,
                       clamp_z_neg: float, clamp_z_pos: float) -> List[Tuple[float, float, float]]:
    """
    对一圈顶点应用层级挤压效果
    
    核心算法（以X方向为例）：
    1. 按原始|X|值分层（从大到小）
    2. 最外层先被挤压到 max_ratio * 原始值
    3. 当外层被压后的位置 < 内层原始位置时，内层也被带动
    4. 被带动的内层 = 外层被压后的位置（保持层级间的相对关系）
    
    注意：X和Z方向独立处理，先处理完X再处理Z
    
    参数:
        coords: 顶点列表 [(x, y, z), ...]，按角度顺序
        clamp_x_neg, clamp_x_pos, clamp_z_neg, clamp_z_pos: 各方向挤压值
    
    返回值:
        挤压后的顶点列表
    """
    if not coords:
        return coords
    
    n = len(coords)
    
    # 将 clamp 值转换为最大允许比例 (0=压到中心, 1=不挤压)
    # -1 -> 1.0 (不挤压), 0 -> 0.0 (压到中心), 1 -> 1.0 (不挤压)
    max_ratio_x_neg = abs(clamp_x_neg)
    max_ratio_x_pos = abs(clamp_x_pos)
    max_ratio_z_neg = abs(clamp_z_neg)
    max_ratio_z_pos = abs(clamp_z_pos)
    
    # 初始化：新坐标从原始坐标开始
    new_coords = [(x, y, z) for x, y, z in coords]
    
    # 辅助函数：对单个方向应用层级挤压
    def squeeze_direction(coords_3d, max_ratio_neg, max_ratio_pos, axis_idx):
        """
        对指定轴应用层级挤压
        
        axis_idx: 0=X, 2=Z
        """
        if max_ratio_neg >= 0.999 and max_ratio_pos >= 0.999:
            return coords_3d  # 该方向不挤压
        
        n = len(coords_3d)
        new_vals = [c[axis_idx] for c in coords_3d]  # 复制当前轴的值
        
        # 按符号分组处理
        for is_negative in [True, False]:
            max_ratio = max_ratio_neg if is_negative else max_ratio_pos
            
            if max_ratio >= 0.999:
                continue  # 该方向不挤压
            
            # 获取该方向上的所有顶点索引和坐标绝对值
            dir_items = [(i, abs(c[axis_idx])) for i, c in enumerate(coords_3d)
                        if (c[axis_idx] < 0) == is_negative and abs(c[axis_idx]) > 1e-9]
            if not dir_items:
                continue
            
            # 按绝对值从大到小排序
            dir_items.sort(key=lambda x: x[1], reverse=True)
            
            # 获取唯一的层级（从大到小）
            unique_levels = []
            for _, val in dir_items:
                if not unique_levels or abs(val - unique_levels[-1]) > 1e-9:
                    unique_levels.append(val)
            
            # 从外向内处理每个层级
            prev_boundary = None  # 外层挤压后的边界位置
            
            for level_idx, level_val in enumerate(unique_levels):
                # 找到该层级的所有顶点
                indices = [i for i, v in dir_items if abs(v - level_val) < 1e-9]
                
                if level_idx == 0:
                    # 最外层：按比例挤压
                    new_pos = level_val * max_ratio
                    for idx in indices:
                        new_vals[idx] = new_pos * (-1 if is_negative else 1)
                    prev_boundary = new_pos
                else:
                    # 内层：检查是否被外层带动
                    if prev_boundary < level_val:
                        # 外层已经压到本层内侧，本层被带动
                        # 本层也被压到边界位置（保持与外层一起移动）
                        new_pos = prev_boundary
                        for idx in indices:
                            new_vals[idx] = new_pos * (-1 if is_negative else 1)
                        # prev_boundary 保持不变（所有内层都压到同一位置）
                    # 否则外层还没压到本层，保持原状
        
        # 构建新的3D坐标
        result = []
        for i, (x, y, z) in enumerate(coords_3d):
            if axis_idx == 0:
                result.append((new_vals[i], y, z))
            else:
                result.append((x, y, new_vals[i]))
        return result
    
    # 先处理 X 方向
    new_coords = squeeze_direction(new_coords, max_ratio_x_neg, max_ratio_x_pos, axis_idx=0)
    # 再处理 Z 方向
    new_coords = squeeze_direction(new_coords, max_ratio_z_neg, max_ratio_z_pos, axis_idx=2)
    
    return new_coords


def get_rounded_rect_point(angle: float, radius_x: float, radius_z: float, 
                           corner_radii: Tuple[float, ...],
                           deformation: float = 0.0) -> Tuple[float, float]:
    """
    Generate rounded rectangle point (support teardrop deformation)
    
    Algorithm:
    1. Calculate standard rounded rect/circle/square point first
    2. Apply teardrop deformation (shrink X based on Z)
    3. Apply radius_x, radius_z scaling
    
    Teardrop deformation:
    - deformation=0: normal shape
    - deformation=0.5: shrink X based on Z
    - deformation=1: bottom shrinks to point
    """
    normalized_angle = angle % (2 * math.pi)
    
    # Determine quadrant
    if math.pi / 2 <= normalized_angle < math.pi:
        corner_idx = 0  # top-left
        sign_x, sign_z = -1, 1
    elif 0 <= normalized_angle < math.pi / 2:
        corner_idx = 1  # top-right
        sign_x, sign_z = 1, 1
    elif 3 * math.pi / 2 <= normalized_angle < 2 * math.pi:
        corner_idx = 2  # bottom-right
        sign_x, sign_z = 1, -1
    else:
        corner_idx = 3  # bottom-left
        sign_x, sign_z = -1, -1
    
    corner_radius = corner_radii[corner_idx] if corner_idx < len(corner_radii) else 1.0
    
    # Step 1: Calculate standard rounded rect point (in unit square/circle)
    if corner_radius <= 0.001:
        # Pure square
        abs_cos = abs(math.cos(normalized_angle))
        abs_sin = abs(math.sin(normalized_angle))
        if abs_cos < 0.0001 or abs_sin < 0.0001:
            t = 1.0
        else:
            t = min(1.0 / abs_cos, 1.0 / abs_sin)
        unit_x = t * math.cos(normalized_angle)
        unit_z = t * math.sin(normalized_angle)
        
    elif corner_radius >= 0.999:
        # Pure circle
        unit_x = math.cos(normalized_angle)
        unit_z = math.sin(normalized_angle)
        
    else:
        # Rounded rectangle
        r = corner_radius
        cx = sign_x * (1 - r)
        cz = sign_z * (1 - r)
        
        # Tangent points
        tangent1_x, tangent1_z = sign_x, sign_z * (1 - r)
        
        # Angles
        angle_tangent1 = math.atan2(tangent1_z, tangent1_x) % (2 * math.pi)
        angle_corner = math.atan2(sign_z, sign_x) % (2 * math.pi)
        
        # Check if on arc
        angle_diff = abs(normalized_angle - angle_corner)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        diff_tangent = abs(angle_tangent1 - angle_corner)
        if diff_tangent > math.pi:
            diff_tangent = 2 * math.pi - diff_tangent
        
        if angle_diff <= diff_tangent:
            # On arc
            unit_x = cx + r * math.cos(normalized_angle)
            unit_z = cz + r * math.sin(normalized_angle)
        else:
            # On straight edge (square boundary)
            abs_cos = abs(math.cos(normalized_angle))
            abs_sin = abs(math.sin(normalized_angle))
            if abs_cos < 0.0001 or abs_sin < 0.0001:
                t = 1.0
            else:
                t = min(1.0 / abs_cos, 1.0 / abs_sin)
            unit_x = t * math.cos(normalized_angle)
            unit_z = t * math.sin(normalized_angle)
    
    # Step 2: Apply teardrop deformation
    # Shrink X based on Z: lower Z means more shrink
    if deformation > 0.001:
        z_norm = max(-1.0, min(1.0, unit_z))
        # Linear interpolation: Z=1 -> width_ratio=1, Z=-1 -> width_ratio=0
        width_ratio = (z_norm + 1.0) / 2.0
        scale = 1.0 - deformation * (1.0 - width_ratio)
        unit_x = unit_x * scale
    
    # Step 3: Apply scaling
    x = radius_x * unit_x
    z = radius_z * unit_z
    
    return x, z


def generate_nose_cone(mesh: Mesh, params: FuselageParams,
                        position: Tuple[float, float, float],
                        rotation: Tuple[float, float, float],
                        segments: int = 24,
                        subdivisions: int = 5) -> int:
    """
    生成NoseCone（冯·卡门曲线锥形）
    
    特点:
    - 类似参考文件的平滑锥形
    - 底部保持圆柱形状，顶部逐渐变细到尖点
    - 使用平滑曲线（类似冯·卡门曲线）而非线性插值
    - 支持所有Fuselage的形变参数
    
    坐标系:
    - 圆锥沿Y轴延伸（从 -length/2 到 +length/2）
    - 底部在下方，顶部（尖端）在上方
    """
    R = rotation_matrix(rotation)
    pos = np.array(position)
    
    # 圆柱长度的一半
    half_len = params.length / 2
    
    # partScale 缩放因子
    scale_x, scale_y, scale_z = params.part_scale
    has_part_scale = abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6 or abs(scale_z - 1.0) > 1e-6
    
    # 生成圆周上的点
    angle_step = 2 * math.pi / segments
    
    # 分离顶部和底部的圆角半径
    if len(params.corner_radiuses) >= 8:
        top_corners = params.corner_radiuses[:4]
        bottom_corners = params.corner_radiuses[4:]
    else:
        top_corners = (1.0, 1.0, 1.0, 1.0)
        bottom_corners = (1.0, 1.0, 1.0, 1.0)
    
    # 分离deformations参数
    top_deformation = params.deformations[0] if len(params.deformations) >= 1 else 0.0
    vertical_shear = params.deformations[1] if len(params.deformations) >= 2 else 0.0
    bottom_deformation = params.deformations[2] if len(params.deformations) >= 3 else 0.0
    
    # 分离 clampDistances 参数
    if len(params.clamp_distances) >= 8:
        top_clamp = params.clamp_distances[:4]
        bottom_clamp = params.clamp_distances[4:]
    else:
        top_clamp = (1.0, 1.0, 1.0, 1.0)
        bottom_clamp = (1.0, 1.0, 1.0, 1.0)
    
    # 共 subdivisions+1 个截面
    num_rings = subdivisions + 1
    
    # 存储每一层的顶点索引
    ring_indices = []  # 每个元素是 [(v_idx, vt_idx, vn_idx), ...]
    
    # 生成每一层的截面
    for ring in range(num_rings):
        # 归一化高度 (0 = 底部, 1 = 顶部)
        t = ring / subdivisions
        
        # 当前层的高度（Y坐标）
        y = -half_len + t * params.length
        
        # 计算当前层的半径比例 - 精确匹配参考文件形状
        # 参考文件的半径变化（t从0到1）：1.0 -> 0.958 -> 0.863 -> 0.649 -> 0.351 -> 0.0
        # 这是一个典型的冯·卡门曲线 (von Karman ogive)
        if t < 0.99:  # 非尖点部分
            # 使用自定义曲线来匹配参考文件的半径变化
            # 调整系数以匹配: t=0.2->0.958, t=0.4->0.863, t=0.6->0.649, t=0.8->0.351
            # 使用 1 - t^2.1 曲线，平衡各层误差
            ease_t = math.pow(t, 2.1)
            radius_ratio_x = params.bottom_scale_x * (1.0 - ease_t) + params.top_scale_x * ease_t
            radius_ratio_z = params.bottom_scale_z * (1.0 - ease_t) + params.top_scale_z * ease_t
        else:
            # 尖点
            radius_ratio_x = 0.0
            radius_ratio_z = 0.0
        
        # 当前层的偏移（从底部到顶部渐变）
        current_offset_x = params.offset_x * t
        current_offset_z = params.offset_z * t
        
        # 当前层的deformation（从底部到顶部渐变）
        current_deformation = bottom_deformation * (1.0 - t) + top_deformation * t
        
        # 当前层的圆角半径（从底部到顶部渐变）
        current_corners = []
        for i in range(4):
            r = bottom_corners[i] * (1.0 - t) + top_corners[i] * t
            current_corners.append(r)
        current_corners = tuple(current_corners)
        
        # 当前层的clamp（从底部到顶部渐变）
        current_clamp = []
        for i in range(4):
            c = bottom_clamp[i] * (1.0 - t) + top_clamp[i] * t
            current_clamp.append(c)
        current_clamp = tuple(current_clamp)
        
        # 计算当前层的有效半径
        current_rx = params.radius_x * radius_ratio_x
        current_rz = params.radius_z * radius_ratio_z
        
        # 生成原始坐标
        raw_coords = []
        for i in range(segments):
            angle = i * angle_step
            
            # 获取截面形状点
            x, z = get_rounded_rect_point(angle, params.radius_x, params.radius_z,
                                          current_corners, current_deformation)
            x *= radius_ratio_x
            z *= radius_ratio_z
            x += current_offset_x
            z += current_offset_z
            raw_coords.append((x, y, z))
        
        # 应用层级挤压
        squeezed_coords = apply_clamp_to_ring(
            raw_coords,
            current_clamp[0], current_clamp[1],
            current_clamp[2], current_clamp[3]
        )
        
        # 应用 partScale
        if has_part_scale:
            squeezed_coords = [
                (x * scale_x, y * scale_y, z * scale_z) for x, y, z in squeezed_coords
            ]
        
        # 变换到世界坐标并添加顶点
        ring_verts = []
        
        # 检查是否是尖点（半径为0）
        is_tip = radius_ratio_x < 1e-6 and radius_ratio_z < 1e-6
        
        if is_tip:
            # 尖点：只生成一个顶点，所有角度共享
            x, y_local, z = squeezed_coords[0]
            
            # 应用竖切斜率
            z_normalized = z / (current_rz * scale_z) if current_rz * scale_z > 0.001 else 0
            cut_depth = 2 * params.offset_y * vertical_shear * t
            y_final = y_local + cut_depth * (z_normalized - 1) / 2
            
            # 变换到世界坐标
            local_pos = np.array([x, y_final, z])
            world_pos = R @ local_pos + pos
            
            # 尖点的法线朝上
            normal = np.array([0, 1, 0])
            world_normal = R @ normal
            
            # UV坐标（中心点）
            u, v = 0.5, t
            
            # 添加顶点到网格
            v_idx = mesh.add_vertex(world_pos[0], world_pos[1], world_pos[2])
            vn_idx = mesh.add_normal(world_normal[0], world_normal[1], world_normal[2])
            vt_idx = mesh.add_uv(u, v)
            
            # 为所有角度重复相同的顶点索引
            for i in range(segments):
                ring_verts.append((v_idx, vt_idx, vn_idx))
        else:
            # 正常截面：为每个角度生成顶点
            for i in range(segments):
                x, y_local, z = squeezed_coords[i]
                
                # 应用竖切斜率
                z_normalized = z / (current_rz * scale_z) if current_rz * scale_z > 0.001 else 0
                cut_depth = 2 * params.offset_y * vertical_shear * t
                y_final = y_local + cut_depth * (z_normalized - 1) / 2
                
                # 变换到世界坐标
                local_pos = np.array([x, y_final, z])
                world_pos = R @ local_pos + pos
                
                # 计算法线
                cos_a = math.cos(i * angle_step)
                sin_a = math.sin(i * angle_step)
                
                # 根据当前层的形状计算法线
                if current_rx > 1e-6 and current_rz > 1e-6:
                    nx = cos_a / current_rx
                    nz = sin_a / current_rz
                    norm = math.sqrt(nx**2 + nz**2)
                    if norm > 0:
                        nx /= norm
                        nz /= norm
                else:
                    nx, nz = cos_a, sin_a
                
                # 圆锥侧面的法线 - 根据曲线斜率计算Y分量
                # 计算曲线在该点的导数
                if t < 0.99:
                    dt = 0.01
                    t_next = min(t + dt, 0.99)
                    ease_t_next = math.pow(t_next, 2.1)
                    radius_next_x = params.bottom_scale_x * (1.0 - ease_t_next) + params.top_scale_x * ease_t_next
                    dr = radius_next_x - radius_ratio_x
                    # 斜率 = dr/dt，法线Y分量与斜率成正比
                    slope = dr / dt
                    ny = -slope * math.sqrt(nx**2 + nz**2)
                else:
                    ny = 0.0
                
                normal = np.array([nx, ny, nz])
                normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else np.array([nx, 0, nz])
                
                world_normal = R @ normal
                
                # UV坐标
                u = i / segments
                v = t
                
                # 添加顶点到网格
                v_idx = mesh.add_vertex(world_pos[0], world_pos[1], world_pos[2])
                vn_idx = mesh.add_normal(world_normal[0], world_normal[1], world_normal[2])
                vt_idx = mesh.add_uv(u, v)
                ring_verts.append((v_idx, vt_idx, vn_idx))
        
        ring_indices.append(ring_verts)
    
    # 生成侧面（四边形网格）
    for ring in range(subdivisions):
        current_ring = ring_indices[ring]
        next_ring = ring_indices[ring + 1]
        
        for i in range(segments):
            next_i = (i + 1) % segments
            
            # 当前四边形的四个顶点
            curr_i = current_ring[i]
            curr_next = current_ring[next_i]
            next_next = next_ring[next_i]
            next_i_vert = next_ring[i]
            
            # 分解为两个三角面
            mesh.add_face(curr_i[0], curr_next[0], next_next[0],
                         curr_i[1], curr_next[1], next_next[1],
                         curr_i[2], curr_next[2], next_next[2])
            mesh.add_face(curr_i[0], next_next[0], next_i_vert[0],
                         curr_i[1], next_next[1], next_i_vert[1],
                         curr_i[2], next_next[2], next_i_vert[2])
    
    # 生成底部端盖（如果是闭合的）
    if params.bottom_scale_x > 1e-6 or params.bottom_scale_z > 1e-6:
        bottom_ring = ring_indices[0]
        
        # 计算底部中心点
        center_y = -half_len * scale_y
        center_x = -params.offset_x * scale_x
        center_z = -params.offset_z * scale_z
        center_local = np.array([center_x, center_y, center_z])
        center_world = R @ center_local + pos
        
        normal_down = R @ np.array([0, -1, 0])
        v_center = mesh.add_vertex(center_world[0], center_world[1], center_world[2])
        vn_center = mesh.add_normal(normal_down[0], normal_down[1], normal_down[2])
        vt_center = mesh.add_uv(0.5, 0.0)
        
        for i in range(segments):
            next_i = (i + 1) % segments
            b_i = bottom_ring[i]
            b_next = bottom_ring[next_i]
            mesh.add_face(v_center, b_next[0], b_i[0],
                         vt_center, b_next[1], b_i[1],
                         vn_center, b_next[2], b_i[2])
    
    # 返回生成的顶点数量
    total_verts = num_rings * segments
    if params.bottom_scale_x > 1e-6 or params.bottom_scale_z > 1e-6:
        total_verts += 1  # 底部中心点
    return total_verts


def generate_ellipse_cylinder(mesh: Mesh, params: FuselageParams, 
                               position: Tuple[float, float, float],
                               rotation: Tuple[float, float, float],
                               segments: int = 24,
                               inlet_wall_thickness: float = 0.0) -> int:
    """
    生成椭圆/圆角矩形截面圆柱体
    
    坐标系说明:
    - 圆柱沿Y轴延伸 (从 -length/2 到 +length/2)
    - XZ平面是椭圆/圆角矩形截面
    - Z轴是椭圆长边方向
    
    参数:
        inlet_wall_thickness: 如果 > 0，生成空心圆柱(Inlet)，值为壁厚
    
    返回生成的顶点数量
    """
    is_inlet = inlet_wall_thickness > 1e-6
    # 旋转矩阵
    R = rotation_matrix(rotation)
    pos = np.array(position)
    
    # 圆柱长度的一半
    half_len = params.length / 2
    
    # partScale 缩放因子（后续会用到）
    scale_x, scale_y, scale_z = params.part_scale
    has_part_scale = abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6 or abs(scale_z - 1.0) > 1e-6
    
    # 生成圆周上的点
    angle_step = 2 * math.pi / segments
    
    # 顶点索引缓存
    bottom_indices = []
    top_indices = []
    bottom_inner_indices = [] if is_inlet else None
    top_inner_indices = [] if is_inlet else None
    
    # 分离顶部和底部的圆角半径 (各4个角)
    if len(params.corner_radiuses) >= 8:
        # SR2格式: 前4个是顶部4个角，后4个是底部4个角
        top_corners = params.corner_radiuses[:4]
        bottom_corners = params.corner_radiuses[4:]
    else:
        top_corners = (1.0, 1.0, 1.0, 1.0)
        bottom_corners = (1.0, 1.0, 1.0, 1.0)
    
    # 分离deformations参数: (顶部水滴, 竖切斜率, 底部水滴)
    top_deformation = params.deformations[0] if len(params.deformations) >= 1 else 0.0
    vertical_shear = params.deformations[1] if len(params.deformations) >= 2 else 0.0
    bottom_deformation = params.deformations[2] if len(params.deformations) >= 3 else 0.0
    
    # 分离 clampDistances 参数
    # 格式: tx_neg, tx_pos, tz_neg, tz_pos, bx_neg, bx_pos, bz_neg, bz_pos
    if len(params.clamp_distances) >= 8:
        top_clamp = params.clamp_distances[:4]   # 顶部: x_neg, x_pos, z_neg, z_pos
        bottom_clamp = params.clamp_distances[4:]  # 底部: x_neg, x_pos, z_neg, z_pos
    else:
        top_clamp = (1.0, 1.0, 1.0, 1.0)
        bottom_clamp = (1.0, 1.0, 1.0, 1.0)
    
    # 竖切斜率: 顶部Y坐标根据Z位置线性变化
    # vertical_shear=0: 顶部是平面 (Y = half_len)
    # vertical_shear=0.5: 顶部是斜面，Y范围 = half_len ± offset_y*0.5
    # vertical_shear=1: 顶部是斜面，Y范围 = half_len ± offset_y
    # 公式: Y = half_len + vertical_shear * offset_y * (Z / radius_z)
    
    # ========== 第一步：生成所有原始顶点坐标 ==========
    # 先生成所有原始坐标，然后应用层级挤压
    
    raw_bottom_coords = []  # [(x, y, z), ...]
    raw_top_coords = []     # [(x, y, z), ...]
    raw_bottom_inner = [] if is_inlet else None
    raw_top_inner = [] if is_inlet else None
    
    for i in range(segments):
        angle = i * angle_step
        
        # 底部原始坐标（外圈）
        bx, bz = get_rounded_rect_point(angle, params.radius_x, 
                                        params.radius_z, bottom_corners,
                                        bottom_deformation)
        bx *= params.bottom_scale_x
        bz *= params.bottom_scale_z
        bx -= params.offset_x
        bz -= params.offset_z
        raw_bottom_coords.append((bx, -half_len, bz))
        
        # 底部内圈坐标（Inlet模式）
        if is_inlet:
            dist = math.sqrt(bx**2 + bz**2)
            if dist > inlet_wall_thickness:
                ratio = (dist - inlet_wall_thickness) / dist
                raw_bottom_inner.append((bx * ratio, -half_len, bz * ratio))
            else:
                raw_bottom_inner.append((0, -half_len, 0))
        
        # 顶部原始坐标（外圈）
        tx, tz = get_rounded_rect_point(angle, params.radius_x, 
                                        params.radius_z, top_corners,
                                        top_deformation)
        tx *= params.top_scale_x
        tz *= params.top_scale_z
        tx += params.offset_x
        tz += params.offset_z
        raw_top_coords.append((tx, half_len, tz))
        
        # 顶部内圈坐标（Inlet模式）
        if is_inlet:
            dist = math.sqrt(tx**2 + tz**2)
            if dist > inlet_wall_thickness:
                ratio = (dist - inlet_wall_thickness) / dist
                raw_top_inner.append((tx * ratio, half_len, tz * ratio))
            else:
                raw_top_inner.append((0, half_len, 0))
    
    # ========== 第二步：应用层级挤压效果 ==========
    # 对底部和顶部分别应用 clamp 挤压
    squeezed_bottom = apply_clamp_to_ring(
        raw_bottom_coords,
        bottom_clamp[0], bottom_clamp[1],  # x_neg, x_pos
        bottom_clamp[2], bottom_clamp[3]   # z_neg, z_pos
    )
    squeezed_top = apply_clamp_to_ring(
        raw_top_coords,
        top_clamp[0], top_clamp[1],  # x_neg, x_pos
        top_clamp[2], top_clamp[3]   # z_neg, z_pos
    )
    
    if is_inlet:
        squeezed_bottom_inner = apply_clamp_to_ring(
            raw_bottom_inner,
            bottom_clamp[0], bottom_clamp[1],
            bottom_clamp[2], bottom_clamp[3]
        )
        squeezed_top_inner = apply_clamp_to_ring(
            raw_top_inner,
            top_clamp[0], top_clamp[1],
            top_clamp[2], top_clamp[3]
        )
    
    # ========== 第三步：应用 partScale 整体缩放 ==========
    # partScale 在 clamp 后、旋转前应用
    if has_part_scale:
        squeezed_bottom = [
            (x * scale_x, y * scale_y, z * scale_z) for x, y, z in squeezed_bottom
        ]
        squeezed_top = [
            (x * scale_x, y * scale_y, z * scale_z) for x, y, z in squeezed_top
        ]
        if is_inlet:
            squeezed_bottom_inner = [
                (x * scale_x, y * scale_y, z * scale_z) for x, y, z in squeezed_bottom_inner
            ]
            squeezed_top_inner = [
                (x * scale_x, y * scale_y, z * scale_z) for x, y, z in squeezed_top_inner
            ]
    
    # ========== 第四步：生成网格顶点 ==========
    for i in range(segments):
        bx, by, bz = squeezed_bottom[i]
        tx, ty, tz = squeezed_top[i]
        
        # 应用竖切到顶部Y坐标
        z_normalized = tz / (params.radius_z * params.top_scale_z) if params.radius_z * params.top_scale_z > 0.001 else 0
        cut_depth = 2 * params.offset_y * vertical_shear
        ty += cut_depth * (z_normalized - 1) / 2
        
        # 变换到世界坐标
        local_b = np.array([bx, by, bz])
        world_b = R @ local_b + pos
        
        local_t = np.array([tx, ty, tz])
        world_t = R @ local_t + pos
        
        # 计算法线 (基于缩放后的半径)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # 计算有效半径（半径 * scale）
        bottom_rx = params.radius_x * params.bottom_scale_x
        bottom_rz = params.radius_z * params.bottom_scale_z
        top_rx = params.radius_x * params.top_scale_x
        top_rz = params.radius_z * params.top_scale_z
        
        # 计算斜面斜率 (Y对Z的变化率)
        # dY/dZ = offset_y * vertical_shear / (2 * radius_z * top_scale_z)
        slope_y = params.offset_y * vertical_shear / (2 * params.radius_z * params.top_scale_z) if params.radius_z * params.top_scale_z > 0.001 else 0
        
        # 底部法线 (使用缩放后的半径，避免除零)
        if bottom_rx > 1e-6 and bottom_rz > 1e-6:
            nbx = cos_a / bottom_rx
            nbz = sin_a / bottom_rz
            norm = math.sqrt(nbx**2 + nbz**2)
            if norm > 0:
                nbx /= norm
                nbz /= norm
        elif bottom_rx > 1e-6:
            nbx, nbz = 0, (1.0 if sin_a > 0 else -1.0)
        elif bottom_rz > 1e-6:
            nbx, nbz = (1.0 if cos_a > 0 else -1.0), 0
        else:
            nbx, nbz = cos_a, sin_a
        normal_b = np.array([nbx, 0, nbz])
        
        # 顶部法线 (考虑斜面，法线应该有Y分量)
        if top_rx > 1e-6 and top_rz > 1e-6:
            ntx = cos_a / top_rx
            ntz = sin_a / top_rz
            # 考虑斜面后的法线: 原始法线 + 斜率影响
            # 对于斜面，法线需要旋转以垂直于斜面
            # 简化: nty 与 ntz * slope_y 相关
            nty = -slope_y * ntz  # 负号因为法线要朝外
            norm = math.sqrt(ntx**2 + nty**2 + ntz**2)
            if norm > 0:
                ntx /= norm
                nty /= norm
                ntz /= norm
        elif top_rx > 1e-6:
            ntx, ntz = 0, (1.0 if sin_a > 0 else -1.0)
            nty = -slope_y * ntz
            norm = math.sqrt(nty**2 + ntz**2)
            if norm > 0:
                nty /= norm
                ntz /= norm
        elif top_rz > 1e-6:
            ntx, ntz = (1.0 if cos_a > 0 else -1.0), 0
            nty = 0
        else:
            ntx, ntz = cos_a, sin_a
            nty = -slope_y * ntz
            norm = math.sqrt(ntx**2 + nty**2 + ntz**2)
            if norm > 0:
                ntx /= norm
                nty /= norm
                ntz /= norm
        normal_t = np.array([ntx, nty, ntz])
        
        # 变换法线
        world_normal_b = R @ normal_b
        world_normal_t = R @ normal_t
        
        # UV: u沿圆周
        u = i / segments
        
        # 添加底部外圈顶点
        v_idx_b = mesh.add_vertex(world_b[0], world_b[1], world_b[2])
        vn_idx_b = mesh.add_normal(world_normal_b[0], world_normal_b[1], world_normal_b[2])
        vt_idx_b = mesh.add_uv(u, 0.0)
        bottom_indices.append((v_idx_b, vt_idx_b, vn_idx_b))
        
        # 添加顶部外圈顶点
        v_idx_t = mesh.add_vertex(world_t[0], world_t[1], world_t[2])
        vn_idx_t = mesh.add_normal(world_normal_t[0], world_normal_t[1], world_normal_t[2])
        vt_idx_t = mesh.add_uv(u, 1.0)
        top_indices.append((v_idx_t, vt_idx_t, vn_idx_t))
        
        # Inlet模式：添加内圈顶点
        if is_inlet:
            bxi, byi, bzi = squeezed_bottom_inner[i]
            txi, tyi, tzi = squeezed_top_inner[i]
            
            # 应用竖切到顶部内圈Y坐标
            z_norm_i = tzi / (params.radius_z * params.top_scale_z) if params.radius_z * params.top_scale_z > 0.001 else 0
            tyi += cut_depth * (z_norm_i - 1) / 2
            
            # 变换到世界坐标
            local_bi = np.array([bxi, byi, bzi])
            world_bi = R @ local_bi + pos
            local_ti = np.array([txi, tyi, tzi])
            world_ti = R @ local_ti + pos
            
            # 内圈法线（朝内，与外圈相反）
            normal_bi = np.array([-nbx, 0, -nbz])
            normal_ti = np.array([-ntx, -nty, -ntz])
            world_normal_bi = R @ normal_bi
            world_normal_ti = R @ normal_ti
            
            # 添加底部内圈顶点
            v_idx_bi = mesh.add_vertex(world_bi[0], world_bi[1], world_bi[2])
            vn_idx_bi = mesh.add_normal(world_normal_bi[0], world_normal_bi[1], world_normal_bi[2])
            vt_idx_bi = mesh.add_uv(u, 0.0)
            bottom_inner_indices.append((v_idx_bi, vt_idx_bi, vn_idx_bi))
            
            # 添加顶部内圈顶点
            v_idx_ti = mesh.add_vertex(world_ti[0], world_ti[1], world_ti[2])
            vn_idx_ti = mesh.add_normal(world_normal_ti[0], world_normal_ti[1], world_normal_ti[2])
            vt_idx_ti = mesh.add_uv(u, 1.0)
            top_inner_indices.append((v_idx_ti, vt_idx_ti, vn_idx_ti))
    
    # 生成侧面四边形
    for i in range(segments):
        next_i = (i + 1) % segments
        
        # 当前四边形的四个顶点
        # bottom[i], bottom[next], top[next], top[i]
        b_i = bottom_indices[i]
        b_next = bottom_indices[next_i]
        t_next = top_indices[next_i]
        t_i = top_indices[i]
        
        # 分解为两个三角面
        # (b_i, b_next, t_next) 和 (b_i, t_next, t_i)
        mesh.add_face(b_i[0], b_next[0], t_next[0],
                     b_i[1], b_next[1], t_next[1],
                     b_i[2], b_next[2], t_next[2])
        mesh.add_face(b_i[0], t_next[0], t_i[0],
                     b_i[1], t_next[1], t_i[1],
                     b_i[2], t_next[2], t_i[2])
    
    # Inlet模式：生成内侧面
    if is_inlet:
        for i in range(segments):
            next_i = (i + 1) % segments
            bi_i = bottom_inner_indices[i]
            bi_next = bottom_inner_indices[next_i]
            ti_next = top_inner_indices[next_i]
            ti_i = top_inner_indices[i]
            
            # 内侧面（法线朝内，面的顺序与外侧面相反）
            mesh.add_face(bi_i[0], ti_i[0], ti_next[0],
                         bi_i[1], ti_i[1], ti_next[1],
                         bi_i[2], ti_i[2], ti_next[2])
            mesh.add_face(bi_i[0], ti_next[0], bi_next[0],
                         bi_i[1], ti_next[1], bi_next[1],
                         bi_i[2], ti_next[2], bi_next[2])
    
    # 生成端盖
    if is_inlet:
        # Inlet模式：生成环形端盖
        normal_down = R @ np.array([0, -1, 0])
        vn_down_idx = mesh.add_normal(normal_down[0], normal_down[1], normal_down[2])
        
        for i in range(segments):
            next_i = (i + 1) % segments
            bo_i = bottom_indices[i]
            bo_next = bottom_indices[next_i]
            bi_next = bottom_inner_indices[next_i]
            bi_i = bottom_inner_indices[i]
            
            # 底部环形端盖（两个三角面）
            mesh.add_face(bo_i[0], bo_next[0], bi_next[0],
                         bo_i[1], bo_next[1], bi_next[1],
                         vn_down_idx, vn_down_idx, vn_down_idx)
            mesh.add_face(bo_i[0], bi_next[0], bi_i[0],
                         bo_i[1], bi_next[1], bi_i[1],
                         vn_down_idx, vn_down_idx, vn_down_idx)
        
        normal_up = R @ np.array([0, 1, 0])
        vn_up_idx = mesh.add_normal(normal_up[0], normal_up[1], normal_up[2])
        
        for i in range(segments):
            next_i = (i + 1) % segments
            to_i = top_indices[i]
            to_next = top_indices[next_i]
            ti_next = top_inner_indices[next_i]
            ti_i = top_inner_indices[i]
            
            # 顶部环形端盖（两个三角面）
            mesh.add_face(to_i[0], ti_i[0], ti_next[0],
                         to_i[1], ti_i[1], ti_next[1],
                         vn_up_idx, vn_up_idx, vn_up_idx)
            mesh.add_face(to_i[0], ti_next[0], to_next[0],
                         to_i[1], ti_next[1], to_next[1],
                         vn_up_idx, vn_up_idx, vn_up_idx)
        
        return len(bottom_indices) * 4
    else:
        # 普通模式：生成实心端盖
        # 底部端盖
        center_bottom_local = np.array([
            -params.offset_x * scale_x, 
            -half_len * scale_y, 
            -params.offset_z * scale_z
        ])
        world_center_bottom = R @ center_bottom_local + pos
        normal_down = R @ np.array([0, -1, 0])
        v_center_bottom = mesh.add_vertex(world_center_bottom[0], world_center_bottom[1], world_center_bottom[2])
        vn_center_bottom = mesh.add_normal(normal_down[0], normal_down[1], normal_down[2])
        vt_center_bottom = mesh.add_uv(0.5, 0.5)
        
        for i in range(segments):
            next_i = (i + 1) % segments
            b_i = bottom_indices[i]
            b_next = bottom_indices[next_i]
            mesh.add_face(v_center_bottom, b_next[0], b_i[0],
                         vt_center_bottom, b_next[1], b_i[1],
                         vn_center_bottom, b_next[2], b_i[2])
        
        # 顶部端盖
        cut_depth = 2 * params.offset_y * vertical_shear * scale_y
        center_top_local = np.array([
            params.offset_x * scale_x, 
            half_len * scale_y - cut_depth/2, 
            params.offset_z * scale_z
        ])
        world_center_top = R @ center_top_local + pos
        normal_up = R @ np.array([0, 1, 0])
        v_center_top = mesh.add_vertex(world_center_top[0], world_center_top[1], world_center_top[2])
        vn_center_top = mesh.add_normal(normal_up[0], normal_up[1], normal_up[2])
        vt_center_top = mesh.add_uv(0.5, 0.5)
        
        for i in range(segments):
            next_i = (i + 1) % segments
            t_i = top_indices[i]
            t_next = top_indices[next_i]
            mesh.add_face(v_center_top, t_i[0], t_next[0],
                         vt_center_top, t_i[1], t_next[1],
                         vn_center_top, t_i[2], t_next[2])
        
        return len(bottom_indices) + len(top_indices) + 2


def parse_fuselage_params(fuselage_elem) -> FuselageParams:
    """从XML元素解析Fuselage参数"""
    params = FuselageParams()
    
    # 解析offset (格式: "x,y,z")
    if 'offset' in fuselage_elem.attrib:
        offset = parse_vector(fuselage_elem.attrib['offset'])
        params.offset_x = offset[0]
        params.offset_y = offset[1]
        params.offset_z = offset[2]
    
    # 解析topScale (格式: "x,z")
    if 'topScale' in fuselage_elem.attrib:
        scale = parse_scale(fuselage_elem.attrib['topScale'])
        params.top_scale_x = scale[0]
        params.top_scale_z = scale[1]
    
    # 解析bottomScale (格式: "x,z")
    if 'bottomScale' in fuselage_elem.attrib:
        scale = parse_scale(fuselage_elem.attrib['bottomScale'])
        params.bottom_scale_x = scale[0]
        params.bottom_scale_z = scale[1]
    
    # 解析deformations
    if 'deformations' in fuselage_elem.attrib:
        params.deformations = parse_vector(fuselage_elem.attrib['deformations'])
    
    # 解析cornerRadiuses (格式: "r0,r1,r2,r3,r4,r5,r6,r7")
    if 'cornerRadiuses' in fuselage_elem.attrib:
        radii = [float(x) for x in fuselage_elem.attrib['cornerRadiuses'].split(',')]
        params.corner_radiuses = tuple(radii)
    
    # 解析clampDistances (格式: "tx_neg,tx_pos,tz_neg,tz_pos,bx_neg,bx_pos,bz_neg,bz_pos")
    if 'clampDistances' in fuselage_elem.attrib:
        clamp_vals = [float(x) for x in fuselage_elem.attrib['clampDistances'].split(',')]
        params.clamp_distances = tuple(clamp_vals)
    
    # 从volume估算长度（如果需要）
    if 'volume' in fuselage_elem.attrib:
        volume = float(fuselage_elem.attrib['volume'])
        # 简化估算: 体积 / (pi * rx * rz) 
        # 但这里我们使用用户提供的长度
    
    return params


def parse_materials(root: ET.Element) -> List[Material]:
    """从XML根元素解析材质列表"""
    materials = []
    
    # 尝试从 DesignerSettings/Theme 或 Themes/Theme 中查找
    theme = root.find('.//DesignerSettings/Theme')
    if theme is None:
        theme = root.find('.//Themes/Theme')
    
    if theme is not None:
        for mat_elem in theme.findall('Material'):
            name = mat_elem.get('name', 'Unknown')
            color_hex = mat_elem.get('color', 'FFFFFF')
            metallic = float(mat_elem.get('m', '0'))
            smoothness = float(mat_elem.get('s', '0'))
            
            # 解析颜色 (hex to RGB 0-1)
            try:
                r = int(color_hex[0:2], 16) / 255.0
                g = int(color_hex[2:4], 16) / 255.0
                b = int(color_hex[4:6], 16) / 255.0
            except:
                r, g, b = 1.0, 1.0, 1.0
            
            # 粗糙度: 游戏内 s=0 是最粗糙(roughness=1), s=1 是最光滑(roughness=0)
            roughness = 1.0 - smoothness
            
            materials.append(Material(name, (r, g, b), metallic, roughness))
    
    return materials


def get_material_index(part_elem: ET.Element) -> int:
    """获取零件使用的材质索引（materials属性的第一位）"""
    materials_attr = part_elem.get('materials', '0,0,0,0,0')
    try:
        indices = [int(x) for x in materials_attr.split(',')]
        return indices[0] if indices else 0
    except:
        return 0


def write_mtl(materials: List[Material], filename: str, used_indices: set):
    """写入MTL材质文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# SimpleRockets 2 Material Export\n")
        f.write("# Format: Standard MTL with PBR extensions\n\n")
        
        for idx in sorted(used_indices):
            if idx < len(materials):
                mat = materials[idx]
                # 材质名使用索引，方便对应
                f.write(f"newmtl mat_{idx}\n")
                f.write(f"# {mat.name}\n")
                
                # 基础色 (Kd)
                f.write(f"Kd {mat.color[0]:.6f} {mat.color[1]:.6f} {mat.color[2]:.6f}\n")
                
                # 环境光 (Ka) - 通常与Kd相同
                f.write(f"Ka {mat.color[0]:.6f} {mat.color[1]:.6f} {mat.color[2]:.6f}\n")
                
                # 高光 (Ks) - 金属度越高，高光色越接近基础色
                ks_r = mat.color[0] * mat.metallic
                ks_g = mat.color[1] * mat.metallic
                ks_b = mat.color[2] * mat.metallic
                f.write(f"Ks {ks_r:.6f} {ks_g:.6f} {ks_b:.6f}\n")
                
                # 高光指数 (Ns) - 根据粗糙度转换
                # roughness 0 = 光滑 = 高Ns, roughness 1 = 粗糙 = 低Ns
                ns = (1.0 - mat.roughness) * 1000.0
                f.write(f"Ns {ns:.6f}\n")
                
                # 溶解度/透明度 (d) - 默认为1.0（不透明）
                f.write(f"d 1.0\n")
                
                # 照明模型
                f.write(f"illum 2\n")
                
                # PBR扩展（Blender等现代渲染器支持）
                f.write(f"# PBR Extensions\n")
                f.write(f"# Metallic: {mat.metallic:.6f}\n")
                f.write(f"# Roughness: {mat.roughness:.6f}\n")
                
                # 使用 Blender 的 Principled BSDF 扩展格式
                f.write(f"Pr {mat.roughness:.6f}\n")
                f.write(f"Pm {mat.metallic:.6f}\n")
                
                f.write("\n")


def convert_sr2_to_obj(xml_file: str, obj_file: str, 
                        default_radius_x: float = 1.0,
                        default_radius_z: float = 1.0):
    """
    将SimpleRockets 2的XML转换为OBJ文件
    
    参数:
        xml_file: 输入XML文件路径
        obj_file: 输出OBJ文件路径
        default_radius_x: 默认椭圆短边半径 (X轴)
        default_radius_z: 默认椭圆长边半径 (Z轴)
        default_length: 默认圆柱长度
    """
    # 解析XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 解析材质列表
    materials = parse_materials(root)
    print(f"解析到 {len(materials)} 种材质")
    
    # 创建网格
    mesh = Mesh()
    
    # 找到所有Part元素
    parts = root.findall('.//Part')
    
    print(f"找到 {len(parts)} 个部件")
    
    # 记录使用的材质索引
    used_material_indices = set()
    
    for part in parts:
        part_id = part.get('id', 'unknown')
        part_type = part.get('partType', 'unknown')
        
        # 解析位置和旋转
        pos_text = part.get('position', '0,0,0')
        position = parse_vector(pos_text)
        
        rot_text = part.get('rotation', '0,0,0')
        # 解析旋转（XML中是度数，可能需要转换，但这里直接用）
        try:
            rotation = parse_vector(rot_text)
        except:
            rotation = (0, 0, 0)
        
        # 获取材质索引
        mat_idx = get_material_index(part)
        used_material_indices.add(mat_idx)
        mat_name = f"mat_{mat_idx}" if mat_idx < len(materials) else "mat_0"
        
        print(f"处理部件 {part_id} ({part_type}) at {position}, rot {rotation}, material={mat_idx}")
        
        if part_type in ('Fuselage1', 'Strut1'):
            # 找到Fuselage子元素（Strut1和Fuselage1使用相同的Fuselage参数结构）
            fuselage = part.find('Fuselage')
            if fuselage is not None:
                params = parse_fuselage_params(fuselage)
                
                # 使用默认参数（从用户提供的信息）
                params.radius_x = default_radius_x
                params.radius_z = default_radius_z
                # 长度从 offset_y 计算: 长度 = offset_y * 2
                params.length = params.offset_y * 2
                
                # 从Config元素解析partScale
                config = part.find('Config')
                if config is not None and 'partScale' in config.attrib:
                    params.part_scale = parse_vector(config.attrib['partScale'])
                
                # Strut1 默认圆角为 0.5（半圆滑），游戏不会记录默认值
                if part_type == 'Strut1' and 'cornerRadiuses' not in fuselage.attrib:
                    params.corner_radiuses = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
                
                type_name = "Strut" if part_type == 'Strut1' else "Fuselage"
                print(f"  {type_name}参数: length={params.length}, "
                      f"radius=({params.radius_x}, {params.radius_z}), "
                      f"offset=({params.offset_x}, {params.offset_y}, {params.offset_z}), "
                      f"topScale=({params.top_scale_x}, {params.top_scale_z}), "
                      f"bottomScale=({params.bottom_scale_x}, {params.bottom_scale_z}), "
                      f"partScale=({params.part_scale[0]}, {params.part_scale[1]}, {params.part_scale[2]})")
                
                # 设置当前材质
                mesh.set_material(mat_name)
                
                # 生成圆柱体 (默认24边)
                vertex_count = generate_ellipse_cylinder(
                    mesh, params, position, rotation, segments=24
                )
                print(f"  生成了 {vertex_count} 个顶点")
            else:
                print(f"  警告: {part_type}部件没有Fuselage子元素")
        
        elif part_type == 'Inlet1':
            # 找到Fuselage子元素（Inlet1也有Fuselage参数）
            fuselage = part.find('Fuselage')
            if fuselage is not None:
                params = parse_fuselage_params(fuselage)
                
                # 使用默认参数
                params.radius_x = default_radius_x
                params.radius_z = default_radius_z
                params.length = params.offset_y * 2
                
                # 从Config元素解析partScale
                config = part.find('Config')
                if config is not None and 'partScale' in config.attrib:
                    params.part_scale = parse_vector(config.attrib['partScale'])
                
                print(f"  Inlet参数: length={params.length}, "
                      f"radius=({params.radius_x}, {params.radius_z}), "
                      f"offset=({params.offset_x}, {params.offset_y}, {params.offset_z}), "
                      f"topScale=({params.top_scale_x}, {params.top_scale_z}), "
                      f"bottomScale=({params.bottom_scale_x}, {params.bottom_scale_z}), "
                      f"partScale=({params.part_scale[0]}, {params.part_scale[1]}, {params.part_scale[2]}), "
                      f"wall=0.045")
                
                # 设置当前材质
                mesh.set_material(mat_name)
                
                # 生成空心圆柱（Inlet）- 使用 inlet_wall_thickness 参数
                vertex_count = generate_ellipse_cylinder(
                    mesh, params, position, rotation, segments=24, inlet_wall_thickness=0.045
                )
                print(f"  生成了 {vertex_count} 个顶点")
            else:
                print(f"  警告: Inlet1部件没有Fuselage子元素")
        
        elif part_type == 'NoseCone1':
            # NoseCone1: 球形尖圆锥体，中间有5段细分
            fuselage = part.find('Fuselage')
            if fuselage is not None:
                params = parse_fuselage_params(fuselage)
                
                # 使用默认参数
                params.radius_x = default_radius_x
                params.radius_z = default_radius_z
                params.length = params.offset_y * 2
                
                # 从Config元素解析partScale
                config = part.find('Config')
                if config is not None and 'partScale' in config.attrib:
                    params.part_scale = parse_vector(config.attrib['partScale'])
                
                print(f"  NoseCone参数: length={params.length}, "
                      f"radius=({params.radius_x}, {params.radius_z}), "
                      f"offset=({params.offset_x}, {params.offset_y}, {params.offset_z}), "
                      f"topScale=({params.top_scale_x}, {params.top_scale_z}), "
                      f"bottomScale=({params.bottom_scale_x}, {params.bottom_scale_z}), "
                      f"partScale=({params.part_scale[0]}, {params.part_scale[1]}, {params.part_scale[2]})")
                
                # 设置当前材质
                mesh.set_material(mat_name)
                
                # 生成NoseCone（5段细分）
                vertex_count = generate_nose_cone(
                    mesh, params, position, rotation, segments=24, subdivisions=5
                )
                print(f"  生成了 {vertex_count} 个顶点")
            else:
                print(f"  警告: NoseCone1部件没有Fuselage子元素")
        
        # 其他部件类型可以在这里扩展
        # elif part_type == 'CommandChip1':
        #     pass
        # elif part_type == 'Block1':
        #     pass
    
    # 生成MTL文件名
    import os
    base_name, ext = os.path.splitext(obj_file)
    mtl_file = base_name + ".mtl"
    mtl_filename = os.path.basename(mtl_file)
    
    # 写入MTL文件
    write_mtl(materials, mtl_file, used_material_indices)
    print(f"\n材质已导出到: {mtl_file}")
    
    # 写入OBJ文件
    mesh.write_obj(obj_file, mtl_filename)
    print(f"模型已导出到: {obj_file}")
    print(f"总顶点数: {len(mesh.vertices)}")
    print(f"总面数: {len(mesh.faces)}")


def main():
    import os
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建 Input 和 Output 文件夹路径
    input_dir = os.path.join(script_dir, 'Input')
    output_dir = os.path.join(script_dir, 'Output')
    
    # 确保 Output 文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 命令行参数处理
    if len(sys.argv) >= 3:
        # 如果提供的是完整路径，直接使用；否则从 Input 文件夹查找
        xml_input = sys.argv[1]
        obj_input = sys.argv[2]
        
        if os.path.isabs(xml_input) or os.path.dirname(xml_input):
            xml_file = xml_input
        else:
            xml_file = os.path.join(input_dir, xml_input)
        
        if os.path.isabs(obj_input) or os.path.dirname(obj_input):
            obj_file = obj_input
        else:
            obj_file = os.path.join(output_dir, obj_input)
    else:
        # 默认文件：从 Input 读取，输出到 Output
        xml_file = os.path.join(input_dir, 'Test-Juno2OBJ.xml')
        obj_file = os.path.join(output_dir, 'Test-Juno2OBJ.obj')
    
    # 检查输入文件是否存在
    if not os.path.exists(xml_file):
        print(f"错误: 输入文件不存在: {xml_file}")
        print(f"请将 XML 文件放入 Input 文件夹: {input_dir}")
        sys.exit(1)
    
    # 转换XML到OBJ
    # 长度从 offset_y 自动计算: 长度 = offset_y * 2
    convert_sr2_to_obj(
        xml_file=xml_file,
        obj_file=obj_file,
        default_radius_x=1.0,   # 基础椭圆短边半径
        default_radius_z=1.0    # 基础椭圆长边半径
    )


if __name__ == '__main__':
    main()
