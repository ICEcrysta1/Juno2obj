#!/usr/bin/env python3
"""
连接解析模块
解析XML中的Connection关系，构建零件父子树
"""

import xml.etree.ElementTree as ET
import json
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict


@dataclass
class Connection:
    """连接关系 - partA是子，partB是父"""
    part_a: str          # 子零件ID
    part_b: str          # 父零件ID
    attach_a: str        # 子连接点
    attach_b: str        # 父连接点


class ConnectionParser:
    """连接解析器"""
    
    # 关节零件类型
    JOINT_TYPES = {'Rotator1', 'HingeRotator1'}
    
    def __init__(self):
        self.connections: List[Connection] = []
        self.child_to_parent: Dict[str, str] = {}      # part_id -> parent_id
        self.parent_to_children: Dict[str, List[str]] = {}  # part_id -> [child_ids]
        self.all_connections: Dict[str, List[str]] = {}  # part_id -> [所有可能的父ID]
    
    def parse_file(self, xml_file: str) -> Dict:
        """
        解析XML文件中的Connections
        
        Returns:
            {
                'connections': [...],           # 原始连接列表
                'child_to_parent': {...},       # 子->父映射
                'parent_to_children': {...},    # 父->子映射
                'root_parts': [...]             # 根零件（没有父的）
            }
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 首先识别所有关节零件
        joint_parts = set()
        parts_elem = root.find('.//Parts')
        if parts_elem is not None:
            for part_elem in parts_elem.findall('Part'):
                part_type = part_elem.get('partType', '')
                if any(jt in part_type for jt in self.JOINT_TYPES):
                    joint_parts.add(part_elem.get('id'))
        
        # 解析所有 Connection，保存所有可能的父
        connections_elem = root.find('.//Connections')
        if connections_elem is not None:
            for conn_elem in connections_elem.findall('Connection'):
                conn = Connection(
                    part_a=conn_elem.get('partA'),
                    part_b=conn_elem.get('partB'),
                    attach_a=conn_elem.get('attachPointsA', ''),
                    attach_b=conn_elem.get('attachPointsB', '')
                )
                self.connections.append(conn)
                
                # 保存所有可能的父关系
                if conn.part_a not in self.all_connections:
                    self.all_connections[conn.part_a] = []
                self.all_connections[conn.part_a].append(conn.part_b)
                
                # 默认使用最后一个连接作为父（保持向后兼容）
                self.child_to_parent[conn.part_a] = conn.part_b
                
                # 构建父->子映射
                if conn.part_b not in self.parent_to_children:
                    self.parent_to_children[conn.part_b] = []
                self.parent_to_children[conn.part_b].append(conn.part_a)
        
        # 检测并修复循环连接
        self._fix_cycles(joint_parts)
        
        # 找出根零件（没有父零件的）
        all_parts = set(self.child_to_parent.keys()) | set(self.parent_to_children.keys())
        root_parts = [p for p in all_parts if p not in self.child_to_parent]
        
        return {
            'connections': [asdict(c) for c in self.connections],
            'child_to_parent': self.child_to_parent,
            'parent_to_children': self.parent_to_children,
            'root_parts': root_parts
        }
    
    def _fix_cycles(self, joint_parts: Set[str]):
        """检测并修复循环连接，选择通往关节的路径"""
        for part_id in list(self.child_to_parent.keys()):
            # 检查从该零件出发是否会形成循环
            visited = set()
            current = part_id
            cycle_path = []
            while current and current not in visited:
                visited.add(current)
                cycle_path.append(current)
                current = self.child_to_parent.get(current)
            
            if current in visited:
                # 发现循环，检查是否有其他路径通往关节
                if part_id in self.all_connections and len(self.all_connections[part_id]) > 1:
                    # 尝试其他父
                    for alternative_parent in self.all_connections[part_id]:
                        if alternative_parent == self.child_to_parent[part_id]:
                            continue
                        
                        # 检查这条路径是否会形成循环
                        visited_alt = set()
                        current_alt = alternative_parent
                        while current_alt and current_alt not in visited_alt:
                            visited_alt.add(current_alt)
                            current_alt = self.child_to_parent.get(current_alt)
                        
                        if current_alt not in visited_alt:
                            # 这条路径不会形成循环，使用它
                            old_parent = self.child_to_parent[part_id]
                            self.child_to_parent[part_id] = alternative_parent
                            
                            # 更新 parent_to_children
                            if old_parent in self.parent_to_children and part_id in self.parent_to_children[old_parent]:
                                self.parent_to_children[old_parent].remove(part_id)
                            if alternative_parent not in self.parent_to_children:
                                self.parent_to_children[alternative_parent] = []
                            if part_id not in self.parent_to_children[alternative_parent]:
                                self.parent_to_children[alternative_parent].append(part_id)
                            break
    
    def save(self, xml_file: str, output_file: str):
        """解析并保存到JSON"""
        data = self.parse_file(xml_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return data


# 便捷函数
def parse_connections(xml_file: str) -> Dict:
    parser = ConnectionParser()
    return parser.parse_file(xml_file)
