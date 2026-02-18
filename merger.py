#!/usr/bin/env python3
"""
合并模块
读取带法线的缓存文件，合并所有网格，导出为USD格式
"""

import os
import sys
import json
import glob
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from dataclasses import dataclass, field

from models import Material, MeshData, clean_material_name


@dataclass
class MaterialData:
    """材质数据"""
    name: str
    color: Tuple[float, float, float]
    metallic: float
    roughness: float


class MeshMerger:
    """网格合并器"""
    
    def __init__(self):
        self.meshes: List[MeshData] = []
        self.materials: Dict[str, MaterialData] = {}
    
    def load_normal_cache(self, cache_dir: str) -> List[MeshData]:
        """
        加载法线计算模块的缓存文件
        
        参数:
            cache_dir: 法线缓存目录
        
        返回:
            MeshData列表
        """
        cache_files = sorted(glob.glob(os.path.join(cache_dir, 'part_*_normals.json')))
        print(f"[Merger] 找到 {len(cache_files)} 个法线缓存文件")
        
        self.meshes = []
        for cache_file in cache_files:
            print(f"[Merger] 加载: {os.path.basename(cache_file)}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            mesh = MeshData.from_dict(data)
            self.meshes.append(mesh)
        
        print(f"[Merger] 已加载 {len(self.meshes)} 个网格")
        return self.meshes
    
    def load_materials(self, materials_file: str):
        """加载材质信息"""
        if not os.path.exists(materials_file):
            print(f"[Merger] 警告: 材质文件不存在: {materials_file}")
            return
        
        with open(materials_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.materials = {}
        for mat in data.get('materials', []):
            clean_name = mat.get('clean_name', clean_material_name(mat['name']))
            self.materials[clean_name] = MaterialData(
                name=mat['name'],
                color=tuple(mat['color']),
                metallic=mat['metallic'],
                roughness=mat['roughness']
            )
        
        print(f"[Merger] 已加载 {len(self.materials)} 种材质")
    
    def merge_and_export(self, output_file: str, mesh_prefix: str = "Mesh", 
                         use_custom_normals: bool = True):
        """
        合并所有网格并导出为USD
        
        参数:
            output_file: 输出USD文件路径
            mesh_prefix: 网格名称前缀
            use_custom_normals: 是否使用自定义法线
        """
        print(f"[Merger] 开始合并并导出到: {output_file}")
        
        # 收集所有顶点、法线、UV和面
        all_vertices = []
        all_normals = []
        all_uvs = []
        all_faces = []  # (v1,v2,v3, vt1,vt2,vt3, vn1,vn2,vn3, material_name, part_id)
        
        vertex_offset = 0
        normal_offset = 0
        uv_offset = 0
        
        for mesh in self.meshes:
            # 偏移顶点索引
            for face in mesh.faces:
                v1, v2, v3, vt1, vt2, vt3, vn1, vn2, vn3, mat_name, part_id = face
                
                # 应用偏移（转为1-based索引）
                all_faces.append((
                    v1 + vertex_offset + 1,
                    v2 + vertex_offset + 1,
                    v3 + vertex_offset + 1,
                    vt1 + uv_offset + 1 if vt1 >= 0 else 0,
                    vt2 + uv_offset + 1 if vt2 >= 0 else 0,
                    vt3 + uv_offset + 1 if vt3 >= 0 else 0,
                    vn1 + normal_offset + 1 if vn1 >= 0 else 0,
                    vn2 + normal_offset + 1 if vn2 >= 0 else 0,
                    vn3 + normal_offset + 1 if vn3 >= 0 else 0,
                    mat_name,
                    part_id
                ))
            
            # 添加顶点、法线、UV
            all_vertices.extend(mesh.vertices)
            all_normals.extend(mesh.normals)
            all_uvs.extend(mesh.uvs)
            
            vertex_offset += len(mesh.vertices)
            normal_offset += len(mesh.normals)
            uv_offset += len(mesh.uvs)
        
        print(f"[Merger] 总计: {len(all_vertices)} 顶点, {len(all_normals)} 法线, {len(all_uvs)} UV, {len(all_faces)} 面")
        
        # 写入USD
        self._write_usd(output_file, all_vertices, all_normals, all_uvs, all_faces, 
                       mesh_prefix, use_custom_normals)
    
    def _write_usd(self, filename: str, vertices: List[Tuple[float, float, float]],
                   normals: List[Tuple[float, float, float]],
                   uvs: List[Tuple[float, float]],
                   faces: List[tuple],
                   mesh_prefix: str, use_custom_normals: bool):
        """写入USD文件"""
        try:
            # 添加本地依赖路径
            deps_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deps')
            if deps_path not in sys.path:
                sys.path.insert(0, deps_path)
            
            from pxr import Usd, UsdGeom, Sdf, Gf, UsdShade
        except ImportError as e:
            raise ImportError(f"USD库加载失败: {e}\n请确保 deps 目录包含 usd-core") from e
        
        # 确保输出目录存在
        abs_filename = os.path.abspath(filename)
        output_dir = os.path.dirname(abs_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"[Merger] 创建输出目录: {output_dir}")
        
        usd_filename = abs_filename.replace("\\", "/")
        
        # 创建舞台
        try:
            stage = Usd.Stage.CreateNew(usd_filename)
        except Exception as e:
            print(f"[Merger] 错误: 创建USD舞台失败: {e}")
            raise RuntimeError(f"USD创建失败: {filename}") from e
        
        # 设置单位
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        
        # 创建根prim
        root_name = mesh_prefix.replace(".", "_").replace("-", "_").replace(" ", "_") if mesh_prefix else "Model"
        if root_name and root_name[0].isdigit():
            root_name = "_" + root_name
        root_path = Sdf.Path(f"/{root_name}")
        root_prim = stage.DefinePrim(root_path, "Xform")
        stage.SetDefaultPrim(root_prim)
        
        # 按材质分组
        faces_by_material = defaultdict(list)
        for face in faces:
            v1, v2, v3, vt1, vt2, vt3, vn1, vn2, vn3, mat_name, part_id = face
            faces_by_material[mat_name].append(((v1-1, v2-1, v3-1), (vn1-1, vn2-1, vn3-1), part_id))
        
        print(f"[Merger] 按材质分组: {len(faces_by_material)} 组")
        
        mesh_index = 0
        for mat_name, face_data in faces_by_material.items():
            mesh_name = mat_name.replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_") if mat_name else f"Mesh_{mesh_index}"
            mesh_path = root_path.AppendChild(mesh_name)
            mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)
            
            # 过滤退化面
            valid_face_data = []
            degenerate_count = 0
            for f in face_data:
                face_verts = f[0]
                if len(set(face_verts)) == 3:
                    valid_face_data.append(f)
                else:
                    degenerate_count += 1
            
            if degenerate_count > 0:
                print(f"[Merger]   Mesh {mesh_name}: 过滤 {degenerate_count} 个退化面")
            
            if not valid_face_data:
                print(f"[Merger]   Mesh {mesh_name}: 无有效面，跳过")
                continue
            
            faces_list = [f[0] for f in valid_face_data]
            normal_indices = [f[1] for f in valid_face_data]
            part_ids = [f[2] for f in valid_face_data]
            
            # 构建局部顶点数组
            used_vertex_keys = set()
            for face, pid in zip(faces_list, part_ids):
                for v_idx in face:
                    used_vertex_keys.add((v_idx, pid))
            
            sorted_keys = sorted(used_vertex_keys, key=lambda x: (x[0], x[1]))
            global_to_local = {key: local_idx for local_idx, key in enumerate(sorted_keys)}
            
            local_vertices = [vertices[v_idx] for v_idx, pid in sorted_keys]
            mesh_prim.CreatePointsAttr(local_vertices)
            
            # 转换面索引
            local_faces = []
            for face, pid in zip(faces_list, part_ids):
                local_face = tuple(global_to_local[(v_idx, pid)] for v_idx in face)
                local_faces.append(local_face)
            
            # 设置法线
            if use_custom_normals and normals:
                face_varying_normals = []
                for face, face_ni in zip(faces_list, normal_indices):
                    for v_idx, ni in zip(face, face_ni):
                        if 0 <= ni < len(normals):
                            face_varying_normals.append(normals[ni])
                        else:
                            face_varying_normals.append((0.0, 1.0, 0.0))
                
                mesh_prim.CreateNormalsAttr(face_varying_normals)
                mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            
            # 设置UV
            if uvs:
                local_uvs = []
                for (v_idx, pid), local_idx in global_to_local.items():
                    if v_idx < len(uvs):
                        local_uvs.append((local_idx, uvs[v_idx]))
                    else:
                        local_uvs.append((local_idx, (0, 0)))
                local_uvs.sort(key=lambda x: x[0])
                local_uvs = [uv for idx, uv in local_uvs]
                
                tex_coords = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
                )
                tex_coords.Set(local_uvs)
            
            # 设置面数和顶点索引
            face_vertex_counts = [3] * len(local_faces)
            mesh_prim.CreateFaceVertexCountsAttr(face_vertex_counts)
            
            face_vertex_indices = [idx for face in local_faces for idx in face]
            mesh_prim.CreateFaceVertexIndicesAttr(face_vertex_indices)
            
            # 创建材质
            if mat_name:
                mat_path = root_path.AppendChild(f"Mat_{mesh_name}")
                material = UsdShade.Material.Define(stage, mat_path)
                
                if mat_name in self.materials:
                    mat_data = self.materials[mat_name]
                    shader_path = mat_path.AppendChild("PreviewSurface")
                    shader = UsdShade.Shader.Define(stage, shader_path)
                    shader.CreateIdAttr("UsdPreviewSurface")
                    
                    diffuse_color = Gf.Vec3f(mat_data.color[0], mat_data.color[1], mat_data.color[2])
                    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse_color)
                    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(mat_data.metallic)
                    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(mat_data.roughness)
                    
                    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                    material.CreateSurfaceOutput().ConnectToSource(shader_output)
                
                UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)
            
            mesh_index += 1
        
        # 保存文件
        stage.GetRootLayer().Save()
        print(f"[Merger] USD文件已保存: {filename}")


def merge_and_export(cache_dir: str, materials_file: str, output_file: str,
                    mesh_prefix: str = "Mesh", use_custom_normals: bool = True):
    """便捷函数：合并并导出"""
    merger = MeshMerger()
    merger.load_normal_cache(cache_dir)
    merger.load_materials(materials_file)
    merger.merge_and_export(output_file, mesh_prefix, use_custom_normals)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 4:
        print("用法: python merger.py <normals_cache_dir> <materials_file> <output_usd>")
        sys.exit(1)
    
    normals_cache_dir = sys.argv[1]
    materials_file = sys.argv[2]
    output_usd = sys.argv[3]
    
    merger = MeshMerger()
    merger.load_normal_cache(normals_cache_dir)
    merger.load_materials(materials_file)
    merger.merge_and_export(output_usd)
    
    print("\n导出完成!")
