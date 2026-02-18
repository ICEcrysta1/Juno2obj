#!/usr/bin/env python3
"""
骨骼USD导出模块
继承 merger.py 的功能，增加骨骼绑定支持

流程：
1. 加载普通网格缓存
2. 加载骨骼数据，为每个 mesh 附加 joint_id
3. 按 joint_id 分组合并网格
4. 导出带骨骼绑定的 USD
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 复用 merger.py 的基础功能
from merger import MeshMerger, MaterialData
from models import MeshData, clean_material_name


class SkeletonMerger(MeshMerger):
    """带骨骼绑定的USD导出器
    
    工作流程：
    1. 加载法线缓存（复用父类）
    2. 加载骨骼数据，标记每个 mesh 的 joint_id
    3. 按 joint_id 分组合并（而非按材质）
    4. 导出 UsdSkel 结构
    """
    
    def __init__(self):
        super().__init__()
        self.skeleton_data: Optional[Dict] = None
        self.joints_order: List[str] = []  # 关节拓扑排序
        self.mesh_to_joint: Dict[str, Optional[str]] = {}  # mesh part_id -> joint_id
    
    def load_skeleton_data(self, skeleton_file: str):
        """加载骨骼数据，并构建 joint 映射"""
        with open(skeleton_file, 'r', encoding='utf-8') as f:
            self.skeleton_data = json.load(f)
        
        # 构建关节拓扑顺序
        self._build_joints_order()
        
        # 构建 mesh -> joint 映射
        bindings = self.skeleton_data.get('bindings', {})
        for part_id, binding in bindings.items():
            self.mesh_to_joint[part_id] = binding.get('joint_id')
        
        joint_count = len(self.skeleton_data['joints'])
        print(f"[SkeletonMerger] 加载 {joint_count} 个关节，{len(bindings)} 个绑定")
    
    def _build_joints_order(self):
        """构建关节的拓扑排序（父在前，子在后）

        注意：必须与 _setup_skeleton 中的顺序完全一致！
        使用完整路径格式：joint_33/joint_35/joint_38
        """
        joints = self.skeleton_data['joints']

        # 创建 joint_id -> part_id 的反向映射
        joint_id_to_part = {j['joint_id']: pid for pid, j in joints.items()}

        def process_joint(joint_id: str, parent_path: str = ""):
            """递归处理关节，构建完整路径"""
            part_id = joint_id_to_part.get(joint_id)
            if not part_id:
                return

            # 构建完整路径
            path = f"{parent_path}/{joint_id}" if parent_path else joint_id
            self.joints_order.append(path)

            # 处理子关节
            joint = joints[part_id]
            for child_joint_id in joint['children']:
                process_joint(child_joint_id, path)

        # 从根关节开始
        for root_id in self.skeleton_data['root_joints']:
            process_joint(root_id)

        print(f"[SkeletonMerger] 关节顺序: {self.joints_order}")

    def merge_and_export(self, output_file: str, mesh_prefix: str = "Mesh",
                         use_custom_normals: bool = True):
        """导出带骨骼的 USD"""
        try:
            deps_path = os.path.join(os.path.dirname(__file__), 'deps')
            if deps_path not in sys.path:
                sys.path.insert(0, deps_path)
            
            from pxr import Usd, UsdGeom, Sdf, Gf, UsdShade, UsdSkel
        except ImportError as e:
            raise ImportError(f"USD库加载失败: {e}") from e
        
        # 确保输出目录
        abs_filename = os.path.abspath(output_file)
        output_dir = os.path.dirname(abs_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 创建舞台（使用 CreateOver 覆盖已存在的文件）
        usd_filename = abs_filename.replace("\\", "/")
        if os.path.exists(usd_filename):
            # 如果文件已存在，使用 Open 然后 Save，或者先删除
            try:
                os.remove(usd_filename)
            except PermissionError:
                # 如果无法删除（被占用），使用不同名称
                import time
                timestamp = int(time.time())
                base, ext = os.path.splitext(usd_filename)
                usd_filename = f"{base}_{timestamp}{ext}"
                print(f"[SkeletonMerger] Warning: Output file in use, saving to: {usd_filename}")
        
        stage = Usd.Stage.CreateNew(usd_filename)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        
        # 创建根
        root_name = clean_material_name(mesh_prefix) if mesh_prefix else "Model"
        if root_name[0].isdigit():
            root_name = "_" + root_name
        root_path = Sdf.Path(f"/{root_name}")
        root_prim = stage.DefinePrim(root_path, "Xform")
        stage.SetDefaultPrim(root_prim)
        
        # 创建 SkelRoot
        skel_root_path = root_path.AppendChild("SkelRoot")
        skel_root = UsdSkel.Root.Define(stage, skel_root_path)
        
        # 创建 Skeleton
        skeleton_path = skel_root_path.AppendChild("Skeleton")
        skeleton = UsdSkel.Skeleton.Define(stage, skeleton_path)
        
        # 设置关节
        self._setup_skeleton(skeleton)
        
        # 按关节分组合并并创建网格
        self._create_skinned_meshes(stage, skel_root_path, skeleton_path, 
                                     use_custom_normals)
        
        # 保存
        stage.GetRootLayer().Save()
        print(f"[SkeletonMerger] USD已保存: {output_file}")
    
    def _setup_skeleton(self, skeleton):
        """设置骨骼关节和绑定姿态"""
        from pxr import Gf
        
        joints = self.skeleton_data['joints']
        
        # 构建 joint paths (包含层级)
        joint_paths = []
        bind_transforms = []
        rest_transforms = []
        
        # 创建 part_id -> joint_id 的反向映射
        joint_id_to_part = {j['joint_id']: pid for pid, j in joints.items()}
        
        def process_joint(joint_id: str, parent_path: str = ""):
            part_id = joint_id_to_part.get(joint_id)
            if not part_id:
                return
            
            joint = joints[part_id]
            path = f"{parent_path}/{joint_id}" if parent_path else joint_id
            joint_paths.append(path)
            
            # 绑定矩阵 (bindTransforms) - 世界空间
            bind_mat = joint['bind_matrix']
            if len(bind_mat) != 16:
                bind_mat = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
            bind_transforms.append(Gf.Matrix4d(*bind_mat))
            
            # 休息姿态 (restTransforms) - 相对于父关节的本地空间
            local_mat = joint.get('local_matrix', bind_mat)
            if len(local_mat) != 16:
                local_mat = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
            rest_transforms.append(Gf.Matrix4d(*local_mat))
            
            # 处理子关节
            for child_joint_id in joint['children']:
                process_joint(child_joint_id, path)
        
        # 从根关节开始处理
        for root_id in self.skeleton_data['root_joints']:
            process_joint(root_id)
        
        # 设置属性
        skeleton.CreateJointsAttr(joint_paths)
        skeleton.CreateBindTransformsAttr(bind_transforms)
        skeleton.CreateRestTransformsAttr(rest_transforms)
        
        print(f"[SkeletonMerger] 设置 {len(joint_paths)} 个关节")
    
    def _create_skinned_meshes(self, stage, skel_root_path, skeleton_path, 
                                use_custom_normals: bool):
        """创建绑定到骨骼的网格
        
        按 (关节, 材质) 双重分组，保留材质分离的效果
        """
        from pxr import UsdGeom, Sdf, UsdSkel, UsdShade, Gf
        
        # 按 (joint_id, material) 分组 mesh
        meshes_by_joint_mat: Dict[Tuple[Optional[str], str], List[MeshData]] = defaultdict(list)
        
        print(f"[SkeletonMerger] 处理 {len(self.meshes)} 个 mesh...")
        sample_meshes = []
        for mesh in self.meshes:
            # 确保 part_id 是字符串类型
            part_id_str = str(mesh.part_id)
            joint_id = self.mesh_to_joint.get(part_id_str)
            if part_id_str in ['272', '337', '208']:
                sample_meshes.append((part_id_str, joint_id))
            
            # 从面中提取材质，按材质分组
            mat_faces: Dict[str, List] = defaultdict(list)
            for face in mesh.faces:
                mat_name = face[9] if len(face) > 9 else "default"
                mat_faces[mat_name].append(face)
            
            # 为每个材质创建子 mesh
            for mat_name, faces in mat_faces.items():
                sub_mesh = MeshData(part_id=f"{part_id_str}_{mat_name}")
                sub_mesh.vertices = mesh.vertices
                sub_mesh.normals = mesh.normals
                sub_mesh.uvs = mesh.uvs
                sub_mesh.faces = faces
                meshes_by_joint_mat[(joint_id, mat_name)].append(sub_mesh)
        
        print(f"[SkeletonMerger] 关键零件: {sample_meshes}")
        print(f"[SkeletonMerger] 按(关节,材质)分组: {len(meshes_by_joint_mat)} 组")
        
        # 为每个 (关节, 材质) 组合创建 Mesh
        for (joint_id, mat_name), mesh_list in meshes_by_joint_mat.items():
            if not mesh_list:
                continue
            
            # 合并同关节同材质的 mesh
            merged = self._merge_meshes(mesh_list)
            if not merged.vertices or not merged.faces:
                continue
            
            # 网格名称: Mesh_joint35_Default_0
            safe_joint = joint_id if joint_id else "Root"
            safe_mat = mat_name.replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_")
            mesh_name = f"Mesh_{safe_joint}_{safe_mat}"
            mesh_path = skel_root_path.AppendChild(mesh_name)
            
            usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
            
            # 设置几何
            valid_faces = self._setup_mesh_geometry(usd_mesh, merged, use_custom_normals)
            if not valid_faces:
                continue
            
            # 应用骨骼绑定
            if joint_id:
                self._apply_skinning(usd_mesh, joint_id, skeleton_path, 
                                     len(merged.vertices))
            
            # 应用材质
            if mat_name and mat_name in self.materials:
                self._create_and_bind_material(stage, usd_mesh, mat_name, skel_root_path)
    
    def _merge_meshes(self, meshes: List[MeshData]) -> MeshData:
        """合并多个网格"""
        merged = MeshData(part_id="merged")
        
        for mesh in meshes:
            v_offset = len(merged.vertices)
            uv_offset = len(merged.uvs)
            n_offset = len(merged.normals)
            
            merged.vertices.extend(mesh.vertices)
            merged.uvs.extend(mesh.uvs)
            merged.normals.extend(mesh.normals)
            
            # 调整面索引
            for face in mesh.faces:
                v1, v2, v3, vt1, vt2, vt3, vn1, vn2, vn3, mat, pid = face
                merged.faces.append((
                    v1 + v_offset,
                    v2 + v_offset,
                    v3 + v_offset,
                    vt1 + uv_offset if vt1 >= 0 else -1,
                    vt2 + uv_offset if vt2 >= 0 else -1,
                    vt3 + uv_offset if vt3 >= 0 else -1,
                    vn1 + n_offset if vn1 >= 0 else -1,
                    vn2 + n_offset if vn2 >= 0 else -1,
                    vn3 + n_offset if vn3 >= 0 else -1,
                    mat, pid
                ))
        
        return merged
    
    def _setup_mesh_geometry(self, usd_mesh, merged: MeshData, 
                              use_custom_normals: bool) -> List:
        """设置网格几何数据，返回有效面列表"""
        from pxr import UsdGeom, Sdf
        
        # 顶点
        if not merged.vertices:
            print(f"[Warning] Mesh {usd_mesh.GetPath()} has no vertices")
            usd_mesh.CreatePointsAttr([])
            usd_mesh.CreateFaceVertexCountsAttr([])
            usd_mesh.CreateFaceVertexIndicesAttr([])
            return []
        
        usd_mesh.CreatePointsAttr(merged.vertices)
        
        # 过滤有效面（非退化、索引有效）
        valid_faces = []
        vertex_count = len(merged.vertices)
        
        for face in merged.faces:
            if len(face) < 3:
                continue
            v1, v2, v3 = face[0], face[1], face[2]
            # 检查索引是否有效且不相同（非退化面）
            if (0 <= v1 < vertex_count and 
                0 <= v2 < vertex_count and 
                0 <= v3 < vertex_count and
                v1 != v2 and v2 != v3 and v1 != v3):
                valid_faces.append(face)
        
        if not valid_faces:
            print(f"[Warning] Mesh {usd_mesh.GetPath()} has no valid faces")
            usd_mesh.CreateFaceVertexCountsAttr([])
            usd_mesh.CreateFaceVertexIndicesAttr([])
            return []
        
        # 面
        face_counts = []
        face_indices = []
        for face in valid_faces:
            face_counts.append(3)
            face_indices.extend([face[0], face[1], face[2]])
        
        usd_mesh.CreateFaceVertexCountsAttr(face_counts)
        usd_mesh.CreateFaceVertexIndicesAttr(face_indices)
        
        # 法线
        if use_custom_normals and merged.normals and valid_faces:
            # Face-varying 法线
            face_varying_normals = []
            for face in valid_faces:
                for i in range(3):
                    ni = face[6 + i] if len(face) > 6 + i else -1  # vn1, vn2, vn3
                    if 0 <= ni < len(merged.normals):
                        face_varying_normals.append(merged.normals[ni])
                    else:
                        face_varying_normals.append((0.0, 1.0, 0.0))
            
            usd_mesh.CreateNormalsAttr(face_varying_normals)
            usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        
        # UV
        if merged.uvs:
            primvars = UsdGeom.PrimvarsAPI(usd_mesh)
            uv_primvar = primvars.CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            uv_primvar.Set(merged.uvs)
        
        # 返回有效面供材质处理使用
        return valid_faces
    
    def _apply_skinning(self, usd_mesh, joint_id: str, skeleton_path, 
                        vertex_count: int):
        """应用骨骼绑定"""
        from pxr import UsdSkel, Sdf, UsdGeom

        # 找到 joint 的完整路径（因为 joints_order 存储的是完整路径）
        # joint_id 可能是 "joint_38"，但 joints_order 中是 "joint_33/joint_35/joint_38"
        joint_path = None
        for path in self.joints_order:
            if path.endswith(joint_id) or path == joint_id:
                joint_path = path
                break

        if joint_path is None:
            print(f"[Warning] Joint {joint_id} not found in order")
            return

        joint_index = self.joints_order.index(joint_path)

        # 创建 SkelBindingAPI
        binding = UsdSkel.BindingAPI.Apply(usd_mesh.GetPrim())
        binding.CreateSkeletonRel().SetTargets([skeleton_path])

        # 每个顶点100%绑定到这个关节
        joint_indices = [joint_index] * vertex_count
        joint_weights = [1.0] * vertex_count

        # 创建 primvar
        primvars = UsdGeom.PrimvarsAPI(usd_mesh)

        indices_primvar = primvars.CreatePrimvar(
            "skel:jointIndices",
            Sdf.ValueTypeNames.IntArray,
            UsdGeom.Tokens.vertex
        )
        indices_primvar.Set(joint_indices)

        weights_primvar = primvars.CreatePrimvar(
            "skel:jointWeights",
            Sdf.ValueTypeNames.FloatArray,
            UsdGeom.Tokens.vertex
        )
        weights_primvar.Set(joint_weights)

    def _create_and_bind_material(self, stage, prim, mat_name: str, root_path):
        """创建并绑定材质"""
        from pxr import Sdf, UsdShade, Gf, UsdGeom
        
        mat_clean = mat_name.replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_")
        mat_path = root_path.AppendChild(f"Mat_{mat_clean}")
        
        # 避免重复创建
        if not stage.GetPrimAtPath(mat_path):
            material = UsdShade.Material.Define(stage, mat_path)
            
            mat_data = self.materials.get(mat_name)
            if mat_data:
                shader_path = mat_path.AppendChild("PreviewSurface")
                shader = UsdShade.Shader.Define(stage, shader_path)
                shader.CreateIdAttr("UsdPreviewSurface")
                
                diffuse_color = Gf.Vec3f(mat_data.color[0], mat_data.color[1], mat_data.color[2])
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse_color)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(mat_data.metallic)
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(mat_data.roughness)
                
                shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                material.CreateSurfaceOutput().ConnectToSource(shader_output)
        
        # 绑定材质
        mat_prim = stage.GetPrimAtPath(mat_path)
        if mat_prim:
            material = UsdShade.Material(mat_prim)
            UsdShade.MaterialBindingAPI(prim).Bind(material)


def merge_and_export_with_skeleton(cache_dir: str, materials_file: str,
                                   skeleton_file: str, output_file: str,
                                   mesh_prefix: str = "Mesh",
                                   use_custom_normals: bool = True):
    """便捷函数：骨骼导出"""
    exporter = SkeletonMerger()
    exporter.load_normal_cache(cache_dir)
    exporter.load_materials(materials_file)
    exporter.load_skeleton_data(skeleton_file)
    exporter.merge_and_export(output_file, mesh_prefix, use_custom_normals)
