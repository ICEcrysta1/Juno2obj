#!/usr/bin/env python3
"""
共享数据模型和工具函数
包含所有模块共享的数据结构和基础工具函数
"""

import xml.etree.ElementTree as ET
import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class FuselageParams:
    """Fuselage参数类，从SimpleRockets 2部件解析"""
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


@dataclass
class PartData:
    """零件数据类，用于在模块间传递"""
    part_id: str = "unknown"
    part_type: str = "unknown"
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    material_name: str = "default"
    material_index: int = 0
    # Fuselage参数（如果是Fuselage类型零件）
    fuselage_params: Optional[FuselageParams] = None
    # 额外参数
    wall_thickness: float = 0.0  # 用于Inlet/Fairing等空心零件
    subdivisions: int = 5  # 用于NoseCone
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于序列化"""
        result = {
            'part_id': self.part_id,
            'part_type': self.part_type,
            'position': self.position,
            'rotation': self.rotation,
            'material_name': self.material_name,
            'material_index': self.material_index,
            'wall_thickness': self.wall_thickness,
            'subdivisions': self.subdivisions,
        }
        if self.fuselage_params:
            params = self.fuselage_params
            result['fuselage_params'] = {
                'length': params.length,
                'radius_x': params.radius_x,
                'radius_z': params.radius_z,
                'offset_x': params.offset_x,
                'offset_y': params.offset_y,
                'offset_z': params.offset_z,
                'top_scale_x': params.top_scale_x,
                'top_scale_z': params.top_scale_z,
                'bottom_scale_x': params.bottom_scale_x,
                'bottom_scale_z': params.bottom_scale_z,
                'deformations': params.deformations,
                'corner_radiuses': params.corner_radiuses,
                'clamp_distances': params.clamp_distances,
                'part_scale': params.part_scale,
            }
        else:
            result['fuselage_params'] = None
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PartData':
        """从字典创建"""
        part = cls(
            part_id=data['part_id'],
            part_type=data['part_type'],
            position=tuple(data['position']),
            rotation=tuple(data['rotation']),
            material_name=data['material_name'],
            material_index=data['material_index'],
            wall_thickness=data.get('wall_thickness', 0.0),
            subdivisions=data.get('subdivisions', 5),
        )
        if data.get('fuselage_params'):
            fp = data['fuselage_params']
            part.fuselage_params = FuselageParams(
                length=fp['length'],
                radius_x=fp['radius_x'],
                radius_z=fp['radius_z'],
                offset_x=fp['offset_x'],
                offset_y=fp['offset_y'],
                offset_z=fp['offset_z'],
                top_scale_x=fp['top_scale_x'],
                top_scale_z=fp['top_scale_z'],
                bottom_scale_x=fp['bottom_scale_x'],
                bottom_scale_z=fp['bottom_scale_z'],
                deformations=tuple(fp['deformations']),
                corner_radiuses=tuple(fp['corner_radiuses']),
                clamp_distances=tuple(fp['clamp_distances']),
                part_scale=tuple(fp['part_scale']),
            )
        return part


@dataclass
class MeshData:
    """网格数据类，用于缓存"""
    part_id: str
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    normals: List[Tuple[float, float, float]] = field(default_factory=list)
    uvs: List[Tuple[float, float]] = field(default_factory=list)
    faces: List[Tuple[int, int, int, int, int, int, int, int, int, str, str]] = field(default_factory=list)
    # face: (v1,v2,v3, vt1,vt2,vt3, vn1,vn2,vn3, material_name, part_id_str)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'part_id': self.part_id,
            'vertices': self.vertices,
            'normals': self.normals,
            'uvs': self.uvs,
            'faces': self.faces,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MeshData':
        """从字典创建"""
        mesh = cls(part_id=data['part_id'])
        mesh.vertices = [tuple(v) for v in data['vertices']]
        mesh.normals = [tuple(n) for n in data['normals']]
        mesh.uvs = [tuple(uv) for uv in data['uvs']]
        mesh.faces = [tuple(f) for f in data['faces']]
        return mesh


# ============== 工具函数 ==============

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


def clean_material_name(name: str) -> str:
    """清理材质名称以符合USD命名规范"""
    return name.replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_")


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
