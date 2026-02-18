#!/usr/bin/env python3
"""
骨骼构建模块
从零件连接关系和关节零件构建骨骼层级
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict


JOINT_TYPES = {'Rotator1', 'HingeRotator1'}


@dataclass
class Joint:
    """骨骼关节"""
    joint_id: str                    # 关节唯一ID (如 "joint_35")
    part_id: str                     # 对应的零件ID
    joint_type: str                  # "Rotator1" 或 "HingeRotator1"
    parent_joint_id: Optional[str]   # 父关节ID
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    bind_matrix: List[float] = field(default_factory=list)   # 世界空间矩阵
    local_matrix: List[float] = field(default_factory=list)  # 相对于父关节的本地矩阵
    children: List[str] = field(default_factory=list)        # 子关节ID列表


@dataclass
class Binding:
    """零件绑定信息"""
    part_id: str
    joint_id: Optional[str]          # None 表示绑定到根
    is_joint: bool                   # 是否本身是关节零件


class SkeletonBuilder:
    """骨骼构建器"""
    
    def __init__(self, parts_data: List[Dict], connections_data: Dict):
        """
        Args:
            parts_data: parser 输出的 parts 列表 (每个是dict)
            connections_data: connection_parser 输出的连接数据
        """
        # 确保 part_id 是字符串类型（防御性编程）
        self.parts = {str(p['part_id']): p for p in parts_data}
        self.child_to_parent = connections_data['child_to_parent']
        self.parent_to_children = connections_data.get('parent_to_children', {})
        
        self.joints: Dict[str, Joint] = {}          # part_id -> Joint
        self.bindings: Dict[str, Binding] = {}      # part_id -> Binding
        self.joint_parts: Set[str] = set()          # 关节零件ID集合
        
        # 缓存优化
        self._joint_id_to_part_cache: Dict[str, str] = {}  # joint_id -> part_id
        self._nearest_joint_cache: Dict[str, Optional[str]] = {}  # part_id -> nearest_joint_id
    
    def build(self) -> Dict:
        """
        构建骨骼和绑定数据
        
        Returns:
            {
                'joints': {...},           # 所有关节
                'bindings': {...},         # 所有零件绑定
                'root_joints': [...],      # 根关节ID列表
                'joint_hierarchy': {...}   # 关节层级树
            }
        """
        # 1. 识别关节零件
        self._identify_joints()
        
        # 2. 构建关节层级
        self._build_joint_hierarchy()
        
        # 3. 计算绑定矩阵
        self._calculate_bind_matrices()
        
        # 4. 为所有零件分配绑定
        self._assign_bindings()
        
        return {
            'joints': {k: asdict(v) for k, v in self.joints.items()},
            'bindings': {k: asdict(v) for k, v in self.bindings.items()},
            'root_joints': self._get_root_joints(),
            'joint_hierarchy': self._build_hierarchy_tree()
        }
    
    def _identify_joints(self):
        """识别所有关节零件"""
        for part_id, part in self.parts.items():
            if part.get('part_type') in JOINT_TYPES:
                joint = Joint(
                    joint_id=f"joint_{part_id}",
                    part_id=part_id,
                    joint_type=part['part_type'],
                    parent_joint_id=None,
                    position=tuple(part['position']),
                    rotation=tuple(part['rotation']),
                    children=[]
                )
                self.joints[part_id] = joint
                self.joint_parts.add(part_id)
    
    def _build_joint_hierarchy(self):
        """构建关节之间的父子层级"""
        for part_id, joint in self.joints.items():
            # 沿着零件树向上找最近的关节父
            parent_joint_id = self._find_parent_joint(part_id)
            if parent_joint_id:
                joint.parent_joint_id = parent_joint_id
                # 找到父关节的part_id
                for pid, j in self.joints.items():
                    if j.joint_id == parent_joint_id:
                        self.joints[pid].children.append(joint.joint_id)
                        break
    
    def _find_parent_joint(self, part_id: str) -> Optional[str]:
        """找到零件的关节父"""
        current = self.child_to_parent.get(part_id)
        while current:
            if current in self.joint_parts:
                return self.joints[current].joint_id
            current = self.child_to_parent.get(current)
        return None
    
    def _calculate_bind_matrices(self):
        """计算每个关节的绑定矩阵
        
        bindTransforms: 关节的世界变换（绑定姿态）
        restTransforms: 关节相对于父关节的本地变换
        """
        # 第一步：计算所有关节的世界矩阵
        world_matrices = {}
        for part_id, joint in self.joints.items():
            world_matrices[part_id] = self._compute_matrix(
                joint.position, 
                joint.rotation
            )
        
        # 第二步：计算本地矩阵（相对于父关节）
        for part_id, joint in self.joints.items():
            if joint.parent_joint_id is None:
                # 根关节：本地矩阵 = 世界矩阵
                joint.local_matrix = world_matrices[part_id]
            else:
                # 子关节：本地矩阵 = 父世界矩阵的逆 × 子世界矩阵
                parent_part_id = self._joint_id_to_part_id(joint.parent_joint_id)
                if parent_part_id:
                    parent_world = np.array(world_matrices[parent_part_id]).reshape(4, 4, order='F')
                    child_world = np.array(world_matrices[part_id]).reshape(4, 4, order='F')
                    
                    parent_inv = np.linalg.inv(parent_world)
                    local_matrix = parent_inv @ child_world
                    
                    joint.local_matrix = local_matrix.flatten('F').tolist()
                else:
                    joint.local_matrix = world_matrices[part_id]
            
            # bind_matrix 使用世界矩阵（用于绑定姿态）
            joint.bind_matrix = world_matrices[part_id]
    
    def _joint_id_to_part_id(self, joint_id: str) -> Optional[str]:
        """从 joint_id 找到对应的 part_id（使用缓存）"""
        if joint_id not in self._joint_id_to_part_cache:
            for part_id, joint in self.joints.items():
                if joint.joint_id == joint_id:
                    self._joint_id_to_part_cache[joint_id] = part_id
                    return part_id
            self._joint_id_to_part_cache[joint_id] = None
            return None
        return self._joint_id_to_part_cache[joint_id]
    
    def _compute_matrix(self, pos: Tuple, rot: Tuple) -> List[float]:
        """从位置和旋转计算4x4变换矩阵（列优先）"""
        from models import rotation_matrix
        
        R = rotation_matrix(rot)  # 使用现有的 rotation_matrix
        
        # 构建 4x4 矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        
        # USD 使用列优先
        return T.flatten('F').tolist()
    
    def _assign_bindings(self):
        """为每个零件分配绑定关节
        
        策略：从每个关节出发，向下遍历其子树，将所有非关节零件绑定到该关节。
        如果遇到另一个关节，则停止该分支的遍历（由那个关节处理自己的子树）。
        """
        # 首先，所有关节绑定到自己
        for part_id in self.joint_parts:
            self.bindings[part_id] = Binding(
                part_id=part_id,
                joint_id=self.joints[part_id].joint_id,
                is_joint=True
            )
        
        # 从每个关节（不只是根关节）开始，向下遍历其子树
        for part_id, joint in self.joints.items():
            self._bind_subtree(part_id, joint.joint_id)
    
    def _bind_subtree(self, root_part_id: str, joint_id: str):
        """递归绑定子树中的所有零件到指定关节
        
        Args:
            root_part_id: 子树的根零件ID（这是一个关节）
            joint_id: 要绑定到的关节ID
        """
        # 获取该零件的所有直接子零件
        children = self.parent_to_children.get(root_part_id, [])
        
        for child_id in children:
            if child_id in self.joint_parts:
                # 遇到另一个关节，跳过（它会由自己的_bind_subtree处理）
                continue
            
            if child_id not in self.bindings:
                # 绑定该零件到当前关节
                self.bindings[child_id] = Binding(
                    part_id=child_id,
                    joint_id=joint_id,
                    is_joint=False
                )
                
                # 递归处理该零件的子零件
                self._bind_subtree_recursive(child_id, joint_id)
    
    def _bind_subtree_recursive(self, part_id: str, joint_id: str):
        """递归绑定子树（非关节零件）
        
        Args:
            part_id: 当前零件ID
            joint_id: 要绑定到的关节ID
        """
        # 获取该零件的所有直接子零件
        children = self.parent_to_children.get(part_id, [])
        
        for child_id in children:
            if child_id in self.joint_parts:
                # 遇到另一个关节，停止该分支
                continue
            
            if child_id not in self.bindings:
                # 绑定该零件到当前关节
                self.bindings[child_id] = Binding(
                    part_id=child_id,
                    joint_id=joint_id,
                    is_joint=False
                )
                
                # 继续递归
                self._bind_subtree_recursive(child_id, joint_id)

    
    def _get_root_joints(self) -> List[str]:
        """获取根关节（没有关节父的）"""
        return [
            j.joint_id for j in self.joints.values() 
            if j.parent_joint_id is None
        ]
    
    def _build_hierarchy_tree(self) -> Dict:
        """构建层级树（用于调试）"""
        joints_by_id = {j.joint_id: j for j in self.joints.values()}
        
        def build_tree(joint_id: str) -> Dict:
            joint = joints_by_id.get(joint_id)
            if not joint:
                return {}
            return {
                'joint_id': joint_id,
                'children': [build_tree(c) for c in joint.children]
            }
        
        return {jid: build_tree(jid) for jid in self._get_root_joints()}


def build_skeleton(parts_file: str, connections_file: str, output_file: str):
    """便捷函数：从文件构建骨骼"""
    with open(parts_file, 'r') as f:
        parts_data = json.load(f)['parts']
    
    with open(connections_file, 'r') as f:
        connections_data = json.load(f)
    
    builder = SkeletonBuilder(parts_data, connections_data)
    result = builder.build()
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result
