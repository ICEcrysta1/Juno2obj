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
    
    def __init__(self):
        self.connections: List[Connection] = []
        self.child_to_parent: Dict[str, str] = {}      # part_id -> parent_id
        self.parent_to_children: Dict[str, List[str]] = {}  # part_id -> [child_ids]
    
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
        
        # 解析所有 Connection
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
                self.child_to_parent[conn.part_a] = conn.part_b
                
                # 构建父->子映射
                if conn.part_b not in self.parent_to_children:
                    self.parent_to_children[conn.part_b] = []
                self.parent_to_children[conn.part_b].append(conn.part_a)
        
        # 找出根零件（没有父零件的）
        all_parts = set(self.child_to_parent.keys()) | set(self.parent_to_children.keys())
        root_parts = [p for p in all_parts if p not in self.child_to_parent]
        
        return {
            'connections': [asdict(c) for c in self.connections],
            'child_to_parent': self.child_to_parent,
            'parent_to_children': self.parent_to_children,
            'root_parts': root_parts
        }
    
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
