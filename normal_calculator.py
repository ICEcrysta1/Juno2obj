#!/usr/bin/env python3
"""
法线计算模块
读取生成模块的缓存文件，计算每个顶点的法线
保存带法线的缓存文件供合并模块使用
"""

import math
import json
import os
import glob
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from models import MeshData


@dataclass
class NormalMeshData:
    """带法线的网格数据"""
    part_id: str
    material_name: str
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    normals: List[Tuple[float, float, float]] = field(default_factory=list)
    uvs: List[Tuple[float, float]] = field(default_factory=list)
    # face: (v1,v2,v3, vt1,vt2,vt3, vn1,vn2,vn3, material_name, part_id)
    faces: List[Tuple[int, int, int, int, int, int, int, int, int, str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'part_id': self.part_id,
            'material_name': self.material_name,
            'vertices': self.vertices,
            'normals': self.normals,
            'uvs': self.uvs,
            'faces': self.faces,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NormalMeshData':
        mesh = cls(
            part_id=data['part_id'],
            material_name=data['material_name']
        )
        mesh.vertices = [tuple(v) for v in data['vertices']]
        mesh.normals = [tuple(n) for n in data['normals']]
        mesh.uvs = [tuple(uv) for uv in data['uvs']]
        mesh.faces = [tuple(f) for f in data['faces']]
        return mesh


class NormalCalculator:
    """法线计算器"""
    
    def __init__(self, use_smooth_normals: bool = True):
        self.use_smooth_normals = use_smooth_normals
        self.meshes: List[NormalMeshData] = []
    
    def process_cache_files(self, cache_dir: str) -> List[NormalMeshData]:
        """
        处理缓存目录中的所有生成文件，计算法线
        
        参数:
            cache_dir: 生成模块的缓存目录
        
        返回:
            NormalMeshData列表
        """
        cache_files = sorted(glob.glob(os.path.join(cache_dir, 'part_*_gen.json')))
        print(f"[NormalCalc] 找到 {len(cache_files)} 个缓存文件")
        
        self.meshes = []
        for cache_file in cache_files:
            print(f"[NormalCalc] 处理: {os.path.basename(cache_file)}")
            mesh = self._process_single_file(cache_file)
            if mesh:
                self.meshes.append(mesh)
                print(f"[NormalCalc]   顶点: {len(mesh.vertices)}, 法线: {len(mesh.normals)}, 面: {len(mesh.faces)}")
        
        print(f"[NormalCalc] 法线计算完成，共 {len(self.meshes)} 个网格")
        return self.meshes
    
    def _process_single_file(self, cache_file: str) -> Optional[NormalMeshData]:
        """处理单个缓存文件"""
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vertices = [tuple(v) for v in data['vertices']]
        uvs = [tuple(uv) for uv in data['uvs']]
        raw_faces = data['faces']
        ring_info = data.get('ring_info', {})
        
        normals = []
        faces = []
        
        # 如果存在ring_info，使用基于ring的法线计算
        if ring_info and vertices:
            faces = self._compute_normals_with_ring_info(
                vertices, raw_faces, ring_info, normals
            )
        else:
            # 没有ring_info，使用简单面法线
            faces = self._compute_simple_face_normals(vertices, raw_faces, normals)
        
        mesh = NormalMeshData(
            part_id=data['part_id'],
            material_name=data['material_name'],
            vertices=vertices,
            normals=normals,
            uvs=uvs,
            faces=faces
        )
        
        return mesh
    
    def _compute_normals_with_ring_info(self, vertices: List[Tuple[float, float, float]],
                                        raw_faces: List[tuple],
                                        ring_info: Dict[str, Any],
                                        normals: List[Tuple[float, float, float]]) -> List[tuple]:
        """基于ring_info计算法线"""
        segments = ring_info.get('segments', 24)
        bottom_indices = ring_info.get('bottom_indices', [])
        top_indices = ring_info.get('top_indices', [])
        bottom_inner_indices = ring_info.get('bottom_inner_indices')
        top_inner_indices = ring_info.get('top_inner_indices')
        bottom_corners = ring_info.get('bottom_corners', (1.0, 1.0, 1.0, 1.0))
        top_corners = ring_info.get('top_corners', (1.0, 1.0, 1.0, 1.0))
        is_inlet = ring_info.get('is_inlet', False)
        
        # 预计算所有侧面法线
        side_normal_map = {}  # 顶点索引 -> 法线索引
        
        if bottom_indices and top_indices and len(bottom_indices) == len(top_indices) == segments:
            bottom_coords = [vertices[idx] for idx in bottom_indices]
            top_coords = [vertices[idx] for idx in top_indices]
            
            for i in range(segments):
                vb = bottom_coords[i]
                vt = top_coords[i]
                
                # 计算法线
                normal_b, normal_t = self._compute_side_normals(vb, vt, bottom_coords, top_coords, i, segments)
                
                # 存储法线
                vn_b = len(normals)
                normals.append(normal_b)
                vn_t = len(normals)
                normals.append(normal_t)
                
                side_normal_map[bottom_indices[i]] = vn_b
                side_normal_map[top_indices[i]] = vn_t
        
        # 计算端盖法线
        cap_normal_map = {}
        if bottom_indices:
            bottom_coords = [vertices[idx] for idx in bottom_indices]
            cap_normals = self._compute_cap_normals(bottom_coords, bottom_corners, is_top=False)
            for i, idx in enumerate(bottom_indices):
                vn = len(normals)
                normals.append(cap_normals[i])
                cap_normal_map[idx] = vn
        
        if top_indices:
            top_coords = [vertices[idx] for idx in top_indices]
            cap_normals = self._compute_cap_normals(top_coords, top_corners, is_top=True)
            for i, idx in enumerate(top_indices):
                vn = len(normals)
                normals.append(cap_normals[i])
                cap_normal_map[idx] = vn
        
        # 内层法线
        inner_side_normal_map = {}
        inner_cap_normal_map = {}
        if is_inlet and bottom_inner_indices and top_inner_indices:
            inner_bottom_coords = [vertices[idx] for idx in bottom_inner_indices]
            inner_top_coords = [vertices[idx] for idx in top_inner_indices]
            
            for i in range(segments):
                vb = inner_bottom_coords[i]
                vt = inner_top_coords[i]
                
                normal_b, normal_t = self._compute_side_normals(vb, vt, inner_bottom_coords, inner_top_coords, i, segments)
                
                # 内层法线朝内（取反）
                vn_b = len(normals)
                normals.append((-normal_b[0], -normal_b[1], -normal_b[2]))
                vn_t = len(normals)
                normals.append((-normal_t[0], -normal_t[1], -normal_t[2]))
                
                inner_side_normal_map[bottom_inner_indices[i]] = vn_b
                inner_side_normal_map[top_inner_indices[i]] = vn_t
            
            # 内层端盖法线（朝内）
            inner_cap_normals = self._compute_cap_normals(inner_bottom_coords, bottom_corners, is_top=False, invert=True)
            for i, idx in enumerate(bottom_inner_indices):
                vn = len(normals)
                normals.append(inner_cap_normals[i])
                inner_cap_normal_map[idx] = vn
            
            inner_cap_normals = self._compute_cap_normals(inner_top_coords, top_corners, is_top=True, invert=True)
            for i, idx in enumerate(top_inner_indices):
                vn = len(normals)
                normals.append(inner_cap_normals[i])
                inner_cap_normal_map[idx] = vn
        
        # 为每个面分配法线
        faces = []
        for raw_face in raw_faces:
            v1, v2, v3, vt1, vt2, vt3, mat_name, part_id = raw_face
            
            # 根据顶点位置决定使用哪种法线
            vn1 = self._get_normal_for_vertex(v1, side_normal_map, cap_normal_map, inner_side_normal_map, inner_cap_normal_map)
            vn2 = self._get_normal_for_vertex(v2, side_normal_map, cap_normal_map, inner_side_normal_map, inner_cap_normal_map)
            vn3 = self._get_normal_for_vertex(v3, side_normal_map, cap_normal_map, inner_side_normal_map, inner_cap_normal_map)
            
            # 如果找不到预计算的法线，计算面法线
            if vn1 < 0:
                vn1 = vn2 = vn3 = self._compute_face_normal(vertices, v1, v2, v3, normals)
            elif vn2 < 0 or vn3 < 0:
                vn_avg = self._compute_face_normal(vertices, v1, v2, v3, normals)
                if vn2 < 0:
                    vn2 = vn_avg
                if vn3 < 0:
                    vn3 = vn_avg
            
            faces.append((v1, v2, v3, vt1, vt2, vt3, vn1, vn2, vn3, mat_name, part_id))
        
        return faces
    
    def _get_normal_for_vertex(self, v_idx: int,
                                side_normal_map: Dict[int, int],
                                cap_normal_map: Dict[int, int],
                                inner_side_normal_map: Dict[int, int],
                                inner_cap_normal_map: Dict[int, int]) -> int:
        """获取顶点对应的法线索引"""
        if v_idx in side_normal_map:
            return side_normal_map[v_idx]
        if v_idx in cap_normal_map:
            return cap_normal_map[v_idx]
        if v_idx in inner_side_normal_map:
            return inner_side_normal_map[v_idx]
        if v_idx in inner_cap_normal_map:
            return inner_cap_normal_map[v_idx]
        return -1
    
    def _compute_side_normals(self, vb: Tuple[float, float, float], 
                              vt: Tuple[float, float, float],
                              bottom_coords: List[Tuple[float, float, float]],
                              top_coords: List[Tuple[float, float, float]],
                              i: int, segments: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """计算侧面法线"""
        # 切线
        tangent = (vt[0] - vb[0], vt[1] - vb[1], vt[2] - vb[2])
        
        # 底部径向
        center_x_b = sum(c[0] for c in bottom_coords) / len(bottom_coords)
        center_z_b = sum(c[2] for c in bottom_coords) / len(bottom_coords)
        dx_b = vb[0] - center_x_b
        dz_b = vb[2] - center_z_b
        h_b = math.sqrt(dx_b**2 + dz_b**2)
        
        if h_b > 1e-6:
            radial_x_b = dx_b / h_b
            radial_z_b = dz_b / h_b
        else:
            angle = i * 2 * math.pi / segments
            radial_x_b = math.cos(angle)
            radial_z_b = math.sin(angle)
        
        # 顶部径向
        center_x_t = sum(c[0] for c in top_coords) / len(top_coords)
        center_z_t = sum(c[2] for c in top_coords) / len(top_coords)
        dx_t = vt[0] - center_x_t
        dz_t = vt[2] - center_z_t
        h_t = math.sqrt(dx_t**2 + dz_t**2)
        
        if h_t > 1e-6:
            radial_x_t = dx_t / h_t
            radial_z_t = dz_t / h_t
        else:
            angle = i * 2 * math.pi / segments
            radial_x_t = math.cos(angle)
            radial_z_t = math.sin(angle)
        
        # 计算Y分量
        tangent_len_sq = tangent[0]**2 + tangent[1]**2 + tangent[2]**2
        
        if tangent_len_sq > 1e-6 and h_b > 1e-6:
            dot_b = radial_x_b * tangent[0] + radial_z_b * tangent[2]
            if abs(tangent[1]) > 1e-6:
                ny_bottom = -dot_b / tangent[1]
            else:
                ny_bottom = 0
        else:
            ny_bottom = 0
        
        if tangent_len_sq > 1e-6 and h_t > 1e-6:
            dot_t = radial_x_t * tangent[0] + radial_z_t * tangent[2]
            if abs(tangent[1]) > 1e-6:
                ny_top = -dot_t / tangent[1]
            else:
                ny_top = 0
        else:
            ny_top = 0
        
        # 归一化
        norm_b = math.sqrt(radial_x_b**2 + ny_bottom**2 + radial_z_b**2)
        norm_t = math.sqrt(radial_x_t**2 + ny_top**2 + radial_z_t**2)
        
        if norm_b > 1e-6:
            radial_x_b /= norm_b
            ny_bottom /= norm_b
            radial_z_b /= norm_b
        
        if norm_t > 1e-6:
            radial_x_t /= norm_t
            ny_top /= norm_t
            radial_z_t /= norm_t
        
        return (radial_x_b, ny_bottom, radial_z_b), (radial_x_t, ny_top, radial_z_t)
    
    def _compute_cap_normals(self, coords: List[Tuple[float, float, float]], 
                            corner_radii: Tuple[float, ...], 
                            is_top: bool = True, 
                            invert: bool = False) -> List[Tuple[float, float, float]]:
        """计算端盖法线"""
        n = len(coords)
        cap_normals = []
        
        center_x = sum(c[0] for c in coords) / n
        center_z = sum(c[2] for c in coords) / n
        
        avg_corner_radius = 0.5
        if corner_radii:
            avg_corner_radius = sum(corner_radii) / len(corner_radii)
        
        smooth_threshold = math.radians(150)
        if avg_corner_radius < 0.1:
            smooth_threshold = math.radians(170)
        elif avg_corner_radius > 0.9:
            smooth_threshold = math.radians(135)
        
        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n
            
            v_curr = coords[i]
            v_prev = coords[prev_i]
            v_next = coords[next_i]
            
            edge_prev = (v_prev[0] - v_curr[0], v_prev[2] - v_curr[2])
            edge_next = (v_next[0] - v_curr[0], v_next[2] - v_curr[2])
            
            len_prev = math.sqrt(edge_prev[0]**2 + edge_prev[1]**2)
            len_next = math.sqrt(edge_next[0]**2 + edge_next[1]**2)
            
            if len_prev > 1e-6 and len_next > 1e-6:
                edge_prev_n = (edge_prev[0] / len_prev, edge_prev[1] / len_prev)
                edge_next_n = (edge_next[0] / len_next, edge_next[1] / len_next)
                
                dot = edge_prev_n[0] * edge_next_n[0] + edge_prev_n[1] * edge_next_n[1]
                dot = max(-1, min(1, dot))
                angle = math.acos(dot)
            else:
                angle = math.pi
            
            if angle >= smooth_threshold:
                dx = center_x - v_curr[0]
                dz = center_z - v_curr[2]
                len_h = math.sqrt(dx**2 + dz**2)
                if len_h > 1e-6:
                    nx = dx / len_h
                    nz = dz / len_h
                else:
                    nx, nz = 0, 0
            else:
                bisector_x = -(edge_prev_n[0] + edge_next_n[0])
                bisector_z = -(edge_prev_n[1] + edge_next_n[1])
                bisector_len = math.sqrt(bisector_x**2 + bisector_z**2)
                if bisector_len > 1e-6:
                    nx = bisector_x / bisector_len
                    nz = bisector_z / bisector_len
                else:
                    dx = center_x - v_curr[0]
                    dz = center_z - v_curr[2]
                    len_h = math.sqrt(dx**2 + dz**2)
                    if len_h > 1e-6:
                        nx = dx / len_h
                        nz = dz / len_h
                    else:
                        nx, nz = 0, 0
            
            if avg_corner_radius > 0.01 and avg_corner_radius < 0.99:
                mix_factor = avg_corner_radius
                dx = center_x - v_curr[0]
                dz = center_z - v_curr[2]
                len_h = math.sqrt(dx**2 + dz**2)
                if len_h > 1e-6:
                    radial_nx = dx / len_h
                    radial_nz = dz / len_h
                else:
                    radial_nx, radial_nz = nx, nz
                nx = nx * (1 - mix_factor) + radial_nx * mix_factor
                nz = nz * (1 - mix_factor) + radial_nz * mix_factor
                norm = math.sqrt(nx**2 + nz**2)
                if norm > 1e-6:
                    nx /= norm
                    nz /= norm
            
            ny = 1.0 if is_top else -1.0
            if invert:
                nx, ny, nz = -nx, -ny, -nz
            
            cap_normals.append((nx, ny, nz))
        
        return cap_normals
    
    def _compute_simple_face_normals(self, vertices: List[Tuple[float, float, float]],
                                     raw_faces: List[tuple],
                                     normals: List[Tuple[float, float, float]]) -> List[tuple]:
        """为每个面计算简单的面法线"""
        faces = []
        
        for raw_face in raw_faces:
            v1, v2, v3, vt1, vt2, vt3, mat_name, part_id = raw_face
            
            vn = self._compute_face_normal(vertices, v1, v2, v3, normals)
            faces.append((v1, v2, v3, vt1, vt2, vt3, vn, vn, vn, mat_name, part_id))
        
        return faces
    
    def _compute_face_normal(self, vertices: List[Tuple[float, float, float]], 
                            v1: int, v2: int, v3: int,
                            normals: List[Tuple[float, float, float]]) -> int:
        """计算单个面的法线"""
        if v1 >= len(vertices) or v2 >= len(vertices) or v3 >= len(vertices):
            normals.append((0.0, 1.0, 0.0))
            return len(normals) - 1
        
        p1 = vertices[v1]
        p2 = vertices[v2]
        p3 = vertices[v3]
        
        edge1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        edge2 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
        
        nx = edge1[1] * edge2[2] - edge1[2] * edge2[1]
        ny = edge1[2] * edge2[0] - edge1[0] * edge2[2]
        nz = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        
        length = math.sqrt(nx**2 + ny**2 + nz**2)
        if length > 1e-6:
            nx /= length
            ny /= length
            nz /= length
        else:
            nx, ny, nz = 0.0, 1.0, 0.0
        
        normals.append((nx, ny, nz))
        return len(normals) - 1
    
    def save_cache(self, cache_dir: str) -> List[str]:
        """保存带法线的网格到缓存文件"""
        os.makedirs(cache_dir, exist_ok=True)
        cache_files = []
        
        for mesh in self.meshes:
            cache_file = os.path.join(cache_dir, f"part_{mesh.part_id}_normals.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(mesh.to_dict(), f, indent=2)
            cache_files.append(cache_file)
        
        print(f"[NormalCalc] 已保存 {len(cache_files)} 个法线缓存文件到: {cache_dir}")
        return cache_files


def calculate_normals(cache_dir: str) -> List[NormalMeshData]:
    """便捷函数：计算法线"""
    calculator = NormalCalculator()
    return calculator.process_cache_files(cache_dir)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python normal_calculator.py <gen_cache_dir>")
        sys.exit(1)
    
    gen_cache_dir = sys.argv[1]
    
    calculator = NormalCalculator()
    meshes = calculator.process_cache_files(gen_cache_dir)
    
    normal_cache_dir = os.path.join(os.path.dirname(gen_cache_dir), 'cache_normals')
    calculator.save_cache(normal_cache_dir)
    
    print(f"\n法线计算完成!")
    print(f"  网格数量: {len(meshes)}")
