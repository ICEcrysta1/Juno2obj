#!/usr/bin/env python3
"""
网格生成模块
接受解析模块传递的数据，进行几何变换，生成顶点和面数据
输出缓存文件供法线计算模块使用
"""

import math
import numpy as np
import json
import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from models import (
    PartData, FuselageParams, MeshData,
    rotation_matrix, get_rounded_rect_point
)


@dataclass
class RawMeshData:
    """原始网格数据，用于法线计算前存储"""
    part_id: str
    material_name: str
    # 顶点数据
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    uvs: List[Tuple[float, float]] = field(default_factory=list)
    # 面数据 (v1,v2,v3, vt1,vt2,vt3, material_name, part_id)
    faces: List[Tuple[int, int, int, int, int, int, str, str]] = field(default_factory=list)
    # 额外信息用于法线计算
    ring_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'part_id': self.part_id,
            'material_name': self.material_name,
            'vertices': self.vertices,
            'uvs': self.uvs,
            'faces': self.faces,
            'ring_info': self.ring_info,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RawMeshData':
        mesh = cls(
            part_id=data['part_id'],
            material_name=data['material_name']
        )
        mesh.vertices = [tuple(v) for v in data['vertices']]
        mesh.uvs = [tuple(uv) for uv in data['uvs']]
        mesh.faces = [tuple(f) for f in data['faces']]
        mesh.ring_info = data.get('ring_info', {})
        return mesh


class MeshGenerator:
    """网格生成器"""
    
    def __init__(self, segments: int = 24):
        self.segments = segments
        self.raw_meshes: List[RawMeshData] = []
    
    def generate_from_parts(self, parts: List[PartData]) -> List[RawMeshData]:
        """
        从零件数据列表生成网格
        
        参数:
            parts: PartData列表
        
        返回:
            RawMeshData列表
        """
        print(f"[Generator] 开始生成网格，共 {len(parts)} 个零件")
        self.raw_meshes = []
        
        for part in parts:
            raw_mesh = self._generate_part_mesh(part)
            if raw_mesh:
                self.raw_meshes.append(raw_mesh)
                print(f"[Generator]   零件 {part.part_id} ({part.part_type}): "
                      f"{len(raw_mesh.vertices)} 顶点, {len(raw_mesh.faces)} 面")
        
        print(f"[Generator] 网格生成完成，共 {len(self.raw_meshes)} 个网格")
        return self.raw_meshes
    
    def _generate_part_mesh(self, part: PartData) -> Optional[RawMeshData]:
        """为单个零件生成网格"""
        if not part.fuselage_params:
            return None
        
        raw_mesh = RawMeshData(
            part_id=part.part_id,
            material_name=part.material_name
        )
        
        if part.part_type in ('Fuselage1', 'Strut1'):
            self._generate_ellipse_cylinder(part, raw_mesh, is_inlet=False)
        elif part.part_type in ('Inlet1', 'FairingBase1', 'Fairing1'):
            self._generate_ellipse_cylinder(part, raw_mesh, is_inlet=True)
        elif part.part_type == 'NoseCone1':
            self._generate_nose_cone(part, raw_mesh, is_hollow=False)
        elif part.part_type == 'FairingNoseCone1':
            self._generate_nose_cone(part, raw_mesh, is_hollow=True)
        else:
            return None
        
        return raw_mesh
    
    def _generate_ellipse_cylinder(self, part: PartData, raw_mesh: RawMeshData, is_inlet: bool):
        """生成椭圆/圆角矩形截面圆柱体"""
        params = part.fuselage_params
        R = rotation_matrix(part.rotation)
        pos = np.array(part.position)
        
        half_len = params.length / 2
        scale_x, scale_y, scale_z = params.part_scale
        has_part_scale = abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6 or abs(scale_z - 1.0) > 1e-6
        
        angle_step = 2 * math.pi / self.segments
        
        # 分离顶部和底部参数
        if len(params.corner_radiuses) >= 8:
            top_corners = params.corner_radiuses[:4]
            bottom_corners = params.corner_radiuses[4:]
        else:
            top_corners = (1.0, 1.0, 1.0, 1.0)
            bottom_corners = (1.0, 1.0, 1.0, 1.0)
        
        top_deformation = params.deformations[0] if len(params.deformations) >= 1 else 0.0
        vertical_shear = params.deformations[1] if len(params.deformations) >= 2 else 0.0
        bottom_deformation = params.deformations[2] if len(params.deformations) >= 3 else 0.0
        
        if len(params.clamp_distances) >= 8:
            top_clamp = params.clamp_distances[:4]
            bottom_clamp = params.clamp_distances[4:]
        else:
            top_clamp = (1.0, 1.0, 1.0, 1.0)
            bottom_clamp = (1.0, 1.0, 1.0, 1.0)
        
        wall_thickness = part.wall_thickness if is_inlet else 0.0
        
        # 检测顶面和底面是否收缩到点（scale_x 和 scale_z 都接近0）
        bottom_is_point = params.bottom_scale_x < 1e-6 and params.bottom_scale_z < 1e-6
        top_is_point = params.top_scale_x < 1e-6 and params.top_scale_z < 1e-6
        
        # 生成原始坐标
        raw_bottom_coords = []
        raw_top_coords = []
        raw_bottom_inner = [] if is_inlet else None
        raw_top_inner = [] if is_inlet else None
        
        for i in range(self.segments):
            angle = i * angle_step
            
            # 底部
            if bottom_is_point:
                # 如果底面是点，所有顶点都在中心
                bx = -params.offset_x
                bz = -params.offset_z
            else:
                bx, bz = get_rounded_rect_point(angle, params.radius_x, params.radius_z, 
                                                bottom_corners, bottom_deformation)
                bx *= params.bottom_scale_x
                bz *= params.bottom_scale_z
                bx -= params.offset_x
                bz -= params.offset_z
            raw_bottom_coords.append((bx, -half_len, bz))
            
            if is_inlet:
                if bottom_is_point:
                    raw_bottom_inner.append((0, -half_len, 0))
                else:
                    dist = math.sqrt(bx**2 + bz**2)
                    if dist > wall_thickness:
                        ratio = (dist - wall_thickness) / dist
                        raw_bottom_inner.append((bx * ratio, -half_len, bz * ratio))
                    else:
                        raw_bottom_inner.append((0, -half_len, 0))
            
            # 顶部
            if top_is_point:
                # 如果顶面是点，所有顶点都在中心
                tx = params.offset_x
                tz = params.offset_z
            else:
                tx, tz = get_rounded_rect_point(angle, params.radius_x, params.radius_z,
                                                top_corners, top_deformation)
                tx *= params.top_scale_x
                tz *= params.top_scale_z
                tx += params.offset_x
                tz += params.offset_z
            raw_top_coords.append((tx, half_len, tz))
            
            if is_inlet:
                if top_is_point:
                    raw_top_inner.append((0, half_len, 0))
                else:
                    dist = math.sqrt(tx**2 + tz**2)
                    if dist > wall_thickness:
                        ratio = (dist - wall_thickness) / dist
                        raw_top_inner.append((tx * ratio, half_len, tz * ratio))
                    else:
                        raw_top_inner.append((0, half_len, 0))
        
        # 应用clamp和partScale
        squeezed_bottom = self._apply_clamp(raw_bottom_coords, bottom_clamp)
        squeezed_top = self._apply_clamp(raw_top_coords, top_clamp)
        
        if is_inlet:
            squeezed_bottom_inner = self._apply_clamp(raw_bottom_inner, bottom_clamp)
            squeezed_top_inner = self._apply_clamp(raw_top_inner, top_clamp)
        
        if has_part_scale:
            squeezed_bottom = [(x * scale_x, y * scale_y, z * scale_z) for x, y, z in squeezed_bottom]
            squeezed_top = [(x * scale_x, y * scale_y, z * scale_z) for x, y, z in squeezed_top]
            if is_inlet:
                squeezed_bottom_inner = [(x * scale_x, y * scale_y, z * scale_z) for x, y, z in squeezed_bottom_inner]
                squeezed_top_inner = [(x * scale_x, y * scale_y, z * scale_z) for x, y, z in squeezed_top_inner]
        
        # 生成顶点
        bottom_indices = []
        top_indices = []
        bottom_inner_indices = [] if is_inlet else None
        top_inner_indices = [] if is_inlet else None
        
        # 预计算底部中心点（如果底面是点）
        if bottom_is_point:
            bx, by, bz = squeezed_bottom[0]  # 所有点都一样，取第一个
            local_b = np.array([bx, by, bz])
            world_b = R @ local_b + pos
            v_center_bottom = len(raw_mesh.vertices)
            raw_mesh.vertices.append(tuple(world_b))
            vt_center_bottom = len(raw_mesh.uvs)
            raw_mesh.uvs.append((0.5, 0.5))
        
        # 预计算顶部中心点（如果顶面是点）
        if top_is_point:
            tx, ty, tz = squeezed_top[0]  # 所有点都一样，取第一个
            # 应用竖切到顶部Y
            z_normalized = tz / (params.radius_z * params.top_scale_z) if params.radius_z * params.top_scale_z > 0.001 else 0
            cut_depth = 2 * params.offset_y * vertical_shear
            ty += cut_depth * (z_normalized - 1) / 2
            
            local_t = np.array([tx, ty, tz])
            world_t = R @ local_t + pos
            v_center_top = len(raw_mesh.vertices)
            raw_mesh.vertices.append(tuple(world_t))
            vt_center_top = len(raw_mesh.uvs)
            raw_mesh.uvs.append((0.5, 0.5))
        
        # 预计算底部内层中心点（如果是inlet且底面是点）
        if is_inlet and bottom_is_point:
            v_center_bottom_inner = v_center_bottom  # 同一点
            vt_center_bottom_inner = vt_center_bottom
        
        # 预计算顶部内层中心点（如果是inlet且顶面是点）
        if is_inlet and top_is_point:
            v_center_top_inner = v_center_top  # 同一点
            vt_center_top_inner = vt_center_top
        
        for i in range(self.segments):
            u = i / self.segments
            
            # 底部顶点
            if bottom_is_point:
                # 底面是点，所有索引指向同一个中心顶点
                bottom_indices.append((v_center_bottom, vt_center_bottom))
            else:
                bx, by, bz = squeezed_bottom[i]
                local_b = np.array([bx, by, bz])
                world_b = R @ local_b + pos
                
                v_idx_b = len(raw_mesh.vertices)
                raw_mesh.vertices.append(tuple(world_b))
                vt_idx_b = len(raw_mesh.uvs)
                raw_mesh.uvs.append((u, 0.0))
                bottom_indices.append((v_idx_b, vt_idx_b))
            
            # 顶部顶点
            if top_is_point:
                # 顶面是点，所有索引指向同一个中心顶点
                top_indices.append((v_center_top, vt_center_top))
            else:
                tx, ty, tz = squeezed_top[i]
                # 应用竖切到顶部Y
                z_normalized = tz / (params.radius_z * params.top_scale_z) if params.radius_z * params.top_scale_z > 0.001 else 0
                cut_depth = 2 * params.offset_y * vertical_shear
                ty += cut_depth * (z_normalized - 1) / 2
                
                local_t = np.array([tx, ty, tz])
                world_t = R @ local_t + pos
                
                v_idx_t = len(raw_mesh.vertices)
                raw_mesh.vertices.append(tuple(world_t))
                vt_idx_t = len(raw_mesh.uvs)
                raw_mesh.uvs.append((u, 1.0))
                top_indices.append((v_idx_t, vt_idx_t))
            
            if is_inlet:
                # 底部内层顶点
                if bottom_is_point:
                    bottom_inner_indices.append((v_center_bottom_inner, vt_center_bottom_inner))
                else:
                    bxi, byi, bzi = squeezed_bottom_inner[i]
                    local_bi = np.array([bxi, byi, bzi])
                    world_bi = R @ local_bi + pos
                    
                    v_idx_bi = len(raw_mesh.vertices)
                    raw_mesh.vertices.append(tuple(world_bi))
                    vt_idx_bi = len(raw_mesh.uvs)
                    raw_mesh.uvs.append((u, 0.0))
                    bottom_inner_indices.append((v_idx_bi, vt_idx_bi))
                
                # 顶部内层顶点
                if top_is_point:
                    top_inner_indices.append((v_center_top_inner, vt_center_top_inner))
                else:
                    txi, tyi, tzi = squeezed_top_inner[i]
                    z_norm_i = tzi / (params.radius_z * params.top_scale_z) if params.radius_z * params.top_scale_z > 0.001 else 0
                    cut_depth = 2 * params.offset_y * vertical_shear
                    tyi += cut_depth * (z_norm_i - 1) / 2
                    
                    local_ti = np.array([txi, tyi, tzi])
                    world_ti = R @ local_ti + pos
                    
                    v_idx_ti = len(raw_mesh.vertices)
                    raw_mesh.vertices.append(tuple(world_ti))
                    vt_idx_ti = len(raw_mesh.uvs)
                    raw_mesh.uvs.append((u, 1.0))
                    top_inner_indices.append((v_idx_ti, vt_idx_ti))
        
        # 记录ring信息供法线计算使用
        raw_mesh.ring_info = {
            'bottom_indices': [idx for idx, _ in bottom_indices],
            'top_indices': [idx for idx, _ in top_indices],
            'bottom_inner_indices': [idx for idx, _ in bottom_inner_indices] if is_inlet else None,
            'top_inner_indices': [idx for idx, _ in top_inner_indices] if is_inlet else None,
            'bottom_corners': bottom_corners,
            'top_corners': top_corners,
            'is_inlet': is_inlet,
            'segments': self.segments,
            'bottom_is_point': bottom_is_point,
            'top_is_point': top_is_point,
        }
        
        # 生成面 (暂时不计算法线，索引为-1)
        if bottom_is_point and top_is_point:
            # 两面都是点，不生成侧面（这是一个退化的圆柱）
            pass
        elif bottom_is_point:
            # 底面是点，顶面是圆：生成三角形扇（从顶点到底边）
            for i in range(self.segments):
                next_i = (i + 1) % self.segments
                b_i = bottom_indices[i]  # 这是中心点
                t_i = top_indices[i]
                t_next = top_indices[next_i]
                # 一个三角形：中心点 -> 顶边当前点 -> 顶边下一个点
                raw_mesh.faces.append((b_i[0], t_i[0], t_next[0], b_i[1], t_i[1], t_next[1], raw_mesh.material_name, part.part_id))
        elif top_is_point:
            # 顶面是点，底面是圆：生成三角形扇（从底边到顶点）
            for i in range(self.segments):
                next_i = (i + 1) % self.segments
                b_i = bottom_indices[i]
                b_next = bottom_indices[next_i]
                t_i = top_indices[i]  # 这是中心点
                # 一个三角形：底边当前点 -> 顶点 -> 底边下一个点
                raw_mesh.faces.append((b_i[0], t_i[0], b_next[0], b_i[1], t_i[1], b_next[1], raw_mesh.material_name, part.part_id))
        else:
            # 正常的四边形侧面
            for i in range(self.segments):
                next_i = (i + 1) % self.segments
                
                b_i = bottom_indices[i]
                b_next = bottom_indices[next_i]
                t_next = top_indices[next_i]
                t_i = top_indices[i]
                
                # 两个三角面，法线索引暂为0
                raw_mesh.faces.append((b_i[0], t_i[0], t_next[0], b_i[1], t_i[1], t_next[1], raw_mesh.material_name, part.part_id))
                raw_mesh.faces.append((b_i[0], t_next[0], b_next[0], b_i[1], t_next[1], b_next[1], raw_mesh.material_name, part.part_id))
        
        if is_inlet:
            # 内层侧面生成（注意法线方向与外层面相反）
            if bottom_is_point and top_is_point:
                # 两面都是点，不生成侧面
                pass
            elif bottom_is_point:
                # 底面是点：三角形扇
                for i in range(self.segments):
                    next_i = (i + 1) % self.segments
                    bi_i = bottom_inner_indices[i]  # 中心点
                    ti_i = top_inner_indices[i]
                    ti_next = top_inner_indices[next_i]
                    # 注意内层面法线方向相反
                    raw_mesh.faces.append((bi_i[0], ti_next[0], ti_i[0], bi_i[1], ti_next[1], ti_i[1], raw_mesh.material_name, part.part_id))
            elif top_is_point:
                # 顶面是点：三角形扇
                for i in range(self.segments):
                    next_i = (i + 1) % self.segments
                    bi_i = bottom_inner_indices[i]
                    bi_next = bottom_inner_indices[next_i]
                    ti_i = top_inner_indices[i]  # 中心点
                    raw_mesh.faces.append((bi_i[0], bi_next[0], ti_i[0], bi_i[1], bi_next[1], ti_i[1], raw_mesh.material_name, part.part_id))
            else:
                # 正常的四边形侧面
                for i in range(self.segments):
                    next_i = (i + 1) % self.segments
                    bi_i = bottom_inner_indices[i]
                    bi_next = bottom_inner_indices[next_i]
                    ti_next = top_inner_indices[next_i]
                    ti_i = top_inner_indices[i]
                    
                    raw_mesh.faces.append((bi_i[0], ti_next[0], ti_i[0], bi_i[1], ti_next[1], ti_i[1], raw_mesh.material_name, part.part_id))
                    raw_mesh.faces.append((bi_i[0], bi_next[0], ti_next[0], bi_i[1], bi_next[1], ti_next[1], raw_mesh.material_name, part.part_id))
            
            # 底端端盖 - 仅在底面不是点且不是空心到点的情况下生成
            if not bottom_is_point:
                for i in range(self.segments):
                    next_i = (i + 1) % self.segments
                    bo_i = bottom_indices[i]
                    bo_next = bottom_indices[next_i]
                    bi_next = bottom_inner_indices[next_i]
                    bi_i = bottom_inner_indices[i]
                    
                    raw_mesh.faces.append((bo_i[0], bo_next[0], bi_next[0], bo_i[1], bo_next[1], bi_next[1], raw_mesh.material_name, part.part_id))
                    raw_mesh.faces.append((bo_i[0], bi_next[0], bi_i[0], bo_i[1], bi_next[1], bi_i[1], raw_mesh.material_name, part.part_id))
            
            # 顶端端盖 - 仅在顶面不是点且不是空心到点的情况下生成
            if not top_is_point:
                for i in range(self.segments):
                    next_i = (i + 1) % self.segments
                    to_i = top_indices[i]
                    to_next = top_indices[next_i]
                    ti_next = top_inner_indices[next_i]
                    ti_i = top_inner_indices[i]
                    
                    raw_mesh.faces.append((to_i[0], ti_i[0], ti_next[0], to_i[1], ti_i[1], ti_next[1], raw_mesh.material_name, part.part_id))
                    raw_mesh.faces.append((to_i[0], ti_next[0], to_next[0], to_i[1], ti_next[1], to_next[1], raw_mesh.material_name, part.part_id))
        else:
            # 实心端盖
            if not bottom_is_point:
                center_bottom_local = np.array([
                    -params.offset_x * scale_x, 
                    -half_len * scale_y, 
                    -params.offset_z * scale_z
                ])
                world_center_bottom = R @ center_bottom_local + pos
                v_center_bottom = len(raw_mesh.vertices)
                raw_mesh.vertices.append(tuple(world_center_bottom))
                vt_center_bottom = len(raw_mesh.uvs)
                raw_mesh.uvs.append((0.5, 0.5))
                
                for i in range(self.segments):
                    next_i = (i + 1) % self.segments
                    b_i = bottom_indices[i]
                    b_next = bottom_indices[next_i]
                    raw_mesh.faces.append((v_center_bottom, b_i[0], b_next[0], vt_center_bottom, b_i[1], b_next[1], raw_mesh.material_name, part.part_id))
            
            if not top_is_point:
                cut_depth = 2 * params.offset_y * vertical_shear * scale_y
                center_top_local = np.array([
                    params.offset_x * scale_x, 
                    half_len * scale_y - cut_depth/2, 
                    params.offset_z * scale_z
                ])
                world_center_top = R @ center_top_local + pos
                v_center_top = len(raw_mesh.vertices)
                raw_mesh.vertices.append(tuple(world_center_top))
                vt_center_top = len(raw_mesh.uvs)
                raw_mesh.uvs.append((0.5, 0.5))
                
                for i in range(self.segments):
                    next_i = (i + 1) % self.segments
                    t_i = top_indices[i]
                    t_next = top_indices[next_i]
                    raw_mesh.faces.append((v_center_top, t_next[0], t_i[0], vt_center_top, t_next[1], t_i[1], raw_mesh.material_name, part.part_id))
    
    def _generate_nose_cone(self, part: PartData, raw_mesh: RawMeshData, is_hollow: bool):
        """生成NoseCone（鼻锥）"""
        params = part.fuselage_params
        R = rotation_matrix(part.rotation)
        pos = np.array(part.position)
        
        half_len = params.length / 2
        scale_x, scale_y, scale_z = params.part_scale
        has_part_scale = abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6 or abs(scale_z - 1.0) > 1e-6
        
        angle_step = 2 * math.pi / self.segments
        subdivisions = part.subdivisions
        num_rings = subdivisions + 1
        
        if len(params.corner_radiuses) >= 8:
            top_corners = params.corner_radiuses[:4]
            bottom_corners = params.corner_radiuses[4:]
        else:
            top_corners = (1.0, 1.0, 1.0, 1.0)
            bottom_corners = (1.0, 1.0, 1.0, 1.0)
        
        top_deformation = params.deformations[0] if len(params.deformations) >= 1 else 0.0
        vertical_shear = params.deformations[1] if len(params.deformations) >= 2 else 0.0
        bottom_deformation = params.deformations[2] if len(params.deformations) >= 3 else 0.0
        
        if len(params.clamp_distances) >= 8:
            top_clamp = params.clamp_distances[:4]
            bottom_clamp = params.clamp_distances[4:]
        else:
            top_clamp = (1.0, 1.0, 1.0, 1.0)
            bottom_clamp = (1.0, 1.0, 1.0, 1.0)
        
        wall_thickness = part.wall_thickness if is_hollow else 0.0
        
        ring_indices = []
        inner_ring_indices = [] if is_hollow else None
        
        for ring in range(num_rings):
            t = ring / subdivisions
            y = -half_len + t * params.length
            
            if t < 0.99:
                ease_t = math.pow(t, 2.1)
                radius_ratio_x = params.bottom_scale_x * (1.0 - ease_t) + params.top_scale_x * ease_t
                radius_ratio_z = params.bottom_scale_z * (1.0 - ease_t) + params.top_scale_z * ease_t
            else:
                radius_ratio_x = 0.0
                radius_ratio_z = 0.0
            
            current_offset_x = params.offset_x * t
            current_offset_z = params.offset_z * t
            current_deformation = bottom_deformation * (1.0 - t) + top_deformation * t
            
            current_corners = []
            for i in range(4):
                r = bottom_corners[i] * (1.0 - t) + top_corners[i] * t
                current_corners.append(r)
            current_corners = tuple(current_corners)
            
            current_clamp = []
            for i in range(4):
                c = bottom_clamp[i] * (1.0 - t) + top_clamp[i] * t
                current_clamp.append(c)
            current_clamp = tuple(current_clamp)
            
            current_rx = params.radius_x * radius_ratio_x
            current_rz = params.radius_z * radius_ratio_z
            
            raw_coords = []
            for i in range(self.segments):
                angle = i * angle_step
                x, z = get_rounded_rect_point(angle, params.radius_x, params.radius_z,
                                              current_corners, current_deformation)
                x *= radius_ratio_x
                z *= radius_ratio_z
                x += current_offset_x
                z += current_offset_z
                raw_coords.append((x, y, z))
            
            squeezed_coords = self._apply_clamp(raw_coords, current_clamp)
            
            if has_part_scale:
                squeezed_coords = [(x * scale_x, y * scale_y, z * scale_z) for x, y, z in squeezed_coords]
            
            ring_verts = []
            inner_ring_verts = [] if is_hollow else None
            
            is_tip = radius_ratio_x < 1e-6 and radius_ratio_z < 1e-6
            
            if is_tip:
                x, y_local, z = squeezed_coords[0]
                z_normalized = z / (current_rz * scale_z) if current_rz * scale_z > 0.001 else 0
                cut_depth = 2 * params.offset_y * vertical_shear * t
                y_final = y_local + cut_depth * (z_normalized - 1) / 2
                
                local_pos = np.array([x, y_final, z])
                world_pos = R @ local_pos + pos
                
                v_idx = len(raw_mesh.vertices)
                raw_mesh.vertices.append(tuple(world_pos))
                vt_idx = len(raw_mesh.uvs)
                raw_mesh.uvs.append((0.5, t))
                
                for i in range(self.segments):
                    ring_verts.append((v_idx, vt_idx))
                    if is_hollow:
                        inner_ring_verts.append((v_idx, vt_idx))
            else:
                if is_hollow:
                    inner_coords = []
                    for i in range(self.segments):
                        x, y_local, z = squeezed_coords[i]
                        dist = math.sqrt(x**2 + z**2)
                        if dist > wall_thickness:
                            ratio = (dist - wall_thickness) / dist
                            inner_coords.append((x * ratio, y_local, z * ratio))
                        else:
                            inner_coords.append((0, y_local, 0))
                
                for i in range(self.segments):
                    x, y_local, z = squeezed_coords[i]
                    z_normalized = z / (current_rz * scale_z) if current_rz * scale_z > 0.001 else 0
                    cut_depth = 2 * params.offset_y * vertical_shear * t
                    y_final = y_local + cut_depth * (z_normalized - 1) / 2
                    
                    local_pos = np.array([x, y_final, z])
                    world_pos = R @ local_pos + pos
                    
                    u = i / self.segments
                    
                    v_idx = len(raw_mesh.vertices)
                    raw_mesh.vertices.append(tuple(world_pos))
                    vt_idx = len(raw_mesh.uvs)
                    raw_mesh.uvs.append((u, t))
                    ring_verts.append((v_idx, vt_idx))
                    
                    if is_hollow:
                        xi, yi_local, zi = inner_coords[i]
                        zi_normalized = zi / (current_rz * scale_z) if current_rz * scale_z > 0.001 else 0
                        cut_depth_i = 2 * params.offset_y * vertical_shear * t
                        y_final_i = yi_local + cut_depth_i * (zi_normalized - 1) / 2
                        
                        local_pos_i = np.array([xi, y_final_i, zi])
                        world_pos_i = R @ local_pos_i + pos
                        
                        v_idx_i = len(raw_mesh.vertices)
                        raw_mesh.vertices.append(tuple(world_pos_i))
                        vt_idx_i = len(raw_mesh.uvs)
                        raw_mesh.uvs.append((u, t))
                        inner_ring_verts.append((v_idx_i, vt_idx_i))
            
            ring_indices.append(ring_verts)
            if is_hollow:
                inner_ring_indices.append(inner_ring_verts)
        
        # 生成侧面
        for ring in range(subdivisions):
            current_ring = ring_indices[ring]
            next_ring = ring_indices[ring + 1]
            
            for i in range(self.segments):
                next_i = (i + 1) % self.segments
                
                curr_i = current_ring[i]
                curr_next = current_ring[next_i]
                next_next = next_ring[next_i]
                next_i_vert = next_ring[i]
                
                raw_mesh.faces.append((curr_i[0], next_i_vert[0], next_next[0], 
                                      curr_i[1], next_i_vert[1], next_next[1], raw_mesh.material_name, part.part_id))
                raw_mesh.faces.append((curr_i[0], next_next[0], curr_next[0],
                                      curr_i[1], next_next[1], curr_next[1], raw_mesh.material_name, part.part_id))
        
        if is_hollow:
            for ring in range(subdivisions):
                current_ring = inner_ring_indices[ring]
                next_ring = inner_ring_indices[ring + 1]
                
                for i in range(self.segments):
                    next_i = (i + 1) % self.segments
                    
                    curr_i = current_ring[i]
                    curr_next = current_ring[next_i]
                    next_next = next_ring[next_i]
                    next_i_vert = next_ring[i]
                    
                    raw_mesh.faces.append((curr_i[0], curr_next[0], next_next[0],
                                          curr_i[1], curr_next[1], next_next[1], raw_mesh.material_name, part.part_id))
                    raw_mesh.faces.append((curr_i[0], next_next[0], next_i_vert[0],
                                          curr_i[1], next_next[1], next_i_vert[1], raw_mesh.material_name, part.part_id))
        
        # 底部端盖
        if params.bottom_scale_x > 1e-6 or params.bottom_scale_z > 1e-6:
            bottom_ring = ring_indices[0]
            
            if is_hollow:
                bottom_inner_ring = inner_ring_indices[0]
                
                for i in range(self.segments):
                    next_i = (i + 1) % self.segments
                    bo_i = bottom_ring[i]
                    bo_next = bottom_ring[next_i]
                    bi_next = bottom_inner_ring[next_i]
                    bi_i = bottom_inner_ring[i]
                    
                    raw_mesh.faces.append((bo_i[0], bo_next[0], bi_next[0],
                                          bo_i[1], bo_next[1], bi_next[1], raw_mesh.material_name, part.part_id))
                    raw_mesh.faces.append((bo_i[0], bi_next[0], bi_i[0],
                                          bo_i[1], bi_next[1], bi_i[1], raw_mesh.material_name, part.part_id))
            else:
                center_y = -half_len * scale_y
                center_x = -params.offset_x * scale_x
                center_z = -params.offset_z * scale_z
                center_local = np.array([center_x, center_y, center_z])
                center_world = R @ center_local + pos
                
                v_center = len(raw_mesh.vertices)
                raw_mesh.vertices.append(tuple(center_world))
                vt_center = len(raw_mesh.uvs)
                raw_mesh.uvs.append((0.5, 0.5))
                
                for i in range(self.segments):
                    next_i = (i + 1) % self.segments
                    b_i = bottom_ring[i]
                    b_next = bottom_ring[next_i]
                    
                    raw_mesh.faces.append((v_center, b_i[0], b_next[0],
                                          vt_center, b_i[1], b_next[1], raw_mesh.material_name, part.part_id))
    
    def _apply_clamp(self, coords: List[Tuple[float, float, float]], 
                     clamp_vals: Tuple[float, ...]) -> List[Tuple[float, float, float]]:
        """应用层级挤压效果"""
        # 简化的clamp实现，完整实现参考原代码
        max_ratio_x_neg = abs(clamp_vals[0]) if len(clamp_vals) > 0 else 1.0
        max_ratio_x_pos = abs(clamp_vals[1]) if len(clamp_vals) > 1 else 1.0
        max_ratio_z_neg = abs(clamp_vals[2]) if len(clamp_vals) > 2 else 1.0
        max_ratio_z_pos = abs(clamp_vals[3]) if len(clamp_vals) > 3 else 1.0
        
        # 如果都不挤压，直接返回
        if max_ratio_x_neg >= 0.999 and max_ratio_x_pos >= 0.999 and max_ratio_z_neg >= 0.999 and max_ratio_z_pos >= 0.999:
            return coords
        
        new_coords = []
        for x, y, z in coords:
            new_x = x
            new_z = z
            
            # X方向挤压
            if x < 0 and max_ratio_x_neg < 0.999:
                new_x = x * max_ratio_x_neg
            elif x > 0 and max_ratio_x_pos < 0.999:
                new_x = x * max_ratio_x_pos
            
            # Z方向挤压
            if z < 0 and max_ratio_z_neg < 0.999:
                new_z = z * max_ratio_z_neg
            elif z > 0 and max_ratio_z_pos < 0.999:
                new_z = z * max_ratio_z_pos
            
            new_coords.append((new_x, y, new_z))
        
        return new_coords
    
    def save_cache(self, cache_dir: str) -> List[str]:
        """保存所有网格到缓存文件"""
        os.makedirs(cache_dir, exist_ok=True)
        cache_files = []
        
        for i, mesh in enumerate(self.raw_meshes):
            cache_file = os.path.join(cache_dir, f"part_{mesh.part_id}_gen.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(mesh.to_dict(), f, indent=2)
            cache_files.append(cache_file)
        
        print(f"[Generator] 已保存 {len(cache_files)} 个缓存文件到: {cache_dir}")
        return cache_files


def generate_meshes(parts: List[PartData], segments: int = 24) -> List[RawMeshData]:
    """便捷函数：从零件数据生成网格"""
    generator = MeshGenerator(segments)
    return generator.generate_from_parts(parts)


if __name__ == '__main__':
    import sys
    from parser import parse_sr2_xml
    
    if len(sys.argv) < 2:
        print("用法: python generator.py <parsed_json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    parts = [PartData.from_dict(p) for p in data['parts']]
    
    generator = MeshGenerator()
    meshes = generator.generate_from_parts(parts)
    
    cache_dir = os.path.join(os.path.dirname(json_file), 'cache_gen')
    generator.save_cache(cache_dir)
    
    print(f"\n生成完成!")
    print(f"  网格数量: {len(meshes)}")
