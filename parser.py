#!/usr/bin/env python3
"""
XML解析模块
解析SimpleRockets 2的XML文件，提取零件属性，传递给生成模块
"""

import xml.etree.ElementTree as ET
import json
import os
from typing import List, Tuple, Optional, Dict, Any
from models import (
    PartData, FuselageParams, Material,
    parse_fuselage_params, parse_materials, get_material_index, clean_material_name
)


class SR2XMLParser:
    """SR2 XML解析器"""
    
    def __init__(self, default_radius_x: float = 1.0, default_radius_z: float = 1.0):
        self.default_radius_x = default_radius_x
        self.default_radius_z = default_radius_z
        self.materials: List[Material] = []
        self.parts: List[PartData] = []
    
    def parse_file(self, xml_file: str) -> Dict[str, Any]:
        """
        解析XML文件，返回解析结果
        
        返回:
            {
                'materials': List[Material],
                'parts': List[PartData],
                'metadata': Dict[str, Any]
            }
        """
        print(f"[Parser] 解析XML文件: {xml_file}")
        
        # 解析XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 解析材质列表
        self.materials = parse_materials(root)
        print(f"[Parser] 解析到 {len(self.materials)} 种材质")
        
        # 找到所有Part元素
        xml_parts = root.findall('.//Part')
        print(f"[Parser] 找到 {len(xml_parts)} 个部件")
        
        self.parts = []
        used_material_indices = set()
        
        for part_elem in xml_parts:
            part_data = self._parse_part(part_elem)
            if part_data:
                self.parts.append(part_data)
                used_material_indices.add(part_data.material_index)
        
        print(f"[Parser] 成功解析 {len(self.parts)} 个有效部件")
        
        # 构建元数据
        metadata = {
            'source_file': xml_file,
            'default_radius_x': self.default_radius_x,
            'default_radius_z': self.default_radius_z,
            'total_parts': len(self.parts),
            'total_materials': len(self.materials),
        }
        
        return {
            'materials': self.materials,
            'parts': self.parts,
            'metadata': metadata
        }
    
    def _parse_part(self, part_elem: ET.Element) -> Optional[PartData]:
        """解析单个零件"""
        part_id = part_elem.get('id', 'unknown')
        part_type = part_elem.get('partType', 'unknown')
        
        # 解析位置和旋转
        pos_text = part_elem.get('position', '0,0,0')
        try:
            position = tuple(float(x) for x in pos_text.split(','))
        except:
            position = (0.0, 0.0, 0.0)
        
        rot_text = part_elem.get('rotation', '0,0,0')
        try:
            rotation = tuple(float(x) for x in rot_text.split(','))
        except:
            rotation = (0.0, 0.0, 0.0)
        
        # 获取材质索引和名称
        mat_idx = get_material_index(part_elem)
        if mat_idx < len(self.materials):
            raw_name = self.materials[mat_idx].name
            mat_name = clean_material_name(raw_name)
        else:
            mat_name = "default"
        
        # 创建零件数据对象
        part_data = PartData(
            part_id=part_id,
            part_type=part_type,
            position=position,
            rotation=rotation,
            material_name=mat_name,
            material_index=mat_idx
        )
        
        # 根据零件类型解析特定参数
        if part_type in ('Fuselage1', 'Strut1'):
            self._parse_fuselage_part(part_elem, part_data, part_type)
        elif part_type in ('Inlet1', 'FairingBase1', 'Fairing1'):
            self._parse_inlet_part(part_elem, part_data, part_type)
        elif part_type == 'NoseCone1':
            self._parse_nose_cone_part(part_elem, part_data, part_type)
        elif part_type == 'FairingNoseCone1':
            self._parse_fairing_nose_cone_part(part_elem, part_data, part_type)
        elif part_type in ('Rotator1', 'HingeRotator1'):
            # 关节零件 - 只解析基本位置信息，不需要几何
            self._parse_joint_part(part_elem, part_data, part_type)
        else:
            # 不支持的零件类型
            return None
        
        return part_data
    
    def _parse_fuselage_part(self, part_elem: ET.Element, part_data: PartData, part_type: str):
        """解析Fuselage/Strut类型零件"""
        fuselage = part_elem.find('Fuselage')
        if fuselage is None:
            print(f"[Parser] 警告: {part_type}部件 {part_data.part_id} 没有Fuselage子元素")
            return
        
        params = parse_fuselage_params(fuselage)
        
        # 使用默认参数
        params.radius_x = self.default_radius_x
        params.radius_z = self.default_radius_z
        # 长度从 offset_y 计算: 长度 = offset_y * 2
        params.length = params.offset_y * 2
        
        # 从Config元素解析partScale
        config = part_elem.find('Config')
        if config is not None and 'partScale' in config.attrib:
            scale_text = config.attrib['partScale']
            params.part_scale = tuple(float(x) for x in scale_text.split(','))
        
        # Strut1 默认圆角为 0.5
        if part_type == 'Strut1' and 'cornerRadiuses' not in fuselage.attrib:
            params.corner_radiuses = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        
        part_data.fuselage_params = params
        
        type_name = "Strut" if part_type == 'Strut1' else "Fuselage"
        print(f"[Parser]   {type_name} {part_data.part_id}: "
              f"length={params.length:.3f}, "
              f"radius=({params.radius_x:.2f}, {params.radius_z:.2f}), "
              f"offset=({params.offset_x:.3f}, {params.offset_y:.3f}, {params.offset_z:.3f}), "
              f"topScale=({params.top_scale_x:.2f}, {params.top_scale_z:.2f}), "
              f"bottomScale=({params.bottom_scale_x:.2f}, {params.bottom_scale_z:.2f})")
    
    def _parse_inlet_part(self, part_elem: ET.Element, part_data: PartData, part_type: str):
        """解析Inlet/Fairing类型零件（空心圆柱）"""
        fuselage = part_elem.find('Fuselage')
        if fuselage is None:
            print(f"[Parser] 警告: {part_type}部件 {part_data.part_id} 没有Fuselage子元素")
            return
        
        params = parse_fuselage_params(fuselage)
        
        # 使用默认参数
        params.radius_x = self.default_radius_x
        params.radius_z = self.default_radius_z
        params.length = params.offset_y * 2
        
        # 从Config元素解析partScale
        config = part_elem.find('Config')
        if config is not None and 'partScale' in config.attrib:
            scale_text = config.attrib['partScale']
            params.part_scale = tuple(float(x) for x in scale_text.split(','))
        
        part_data.fuselage_params = params
        part_data.wall_thickness = 0.045  # 默认壁厚
        
        type_name = part_type.replace('1', '')
        print(f"[Parser]   {type_name} {part_data.part_id}: "
              f"length={params.length:.3f}, wall={part_data.wall_thickness:.3f}")
    
    def _parse_nose_cone_part(self, part_elem: ET.Element, part_data: PartData, part_type: str):
        """解析NoseCone类型零件"""
        fuselage = part_elem.find('Fuselage')
        if fuselage is None:
            print(f"[Parser] 警告: {part_type}部件 {part_data.part_id} 没有Fuselage子元素")
            return
        
        params = parse_fuselage_params(fuselage)
        
        # 使用默认参数
        params.radius_x = self.default_radius_x
        params.radius_z = self.default_radius_z
        params.length = params.offset_y * 2
        
        # 从Config元素解析partScale
        config = part_elem.find('Config')
        if config is not None and 'partScale' in config.attrib:
            scale_text = config.attrib['partScale']
            params.part_scale = tuple(float(x) for x in scale_text.split(','))
        
        part_data.fuselage_params = params
        part_data.subdivisions = 5  # NoseCone使用5段细分
        
        print(f"[Parser]   NoseCone {part_data.part_id}: "
              f"length={params.length:.3f}, subdivisions={part_data.subdivisions}")
    
    def _parse_fairing_nose_cone_part(self, part_elem: ET.Element, part_data: PartData, part_type: str):
        """解析FairingNoseCone类型零件（空心鼻锥）"""
        fuselage = part_elem.find('Fuselage')
        if fuselage is None:
            print(f"[Parser] 警告: {part_type}部件 {part_data.part_id} 没有Fuselage子元素")
            return
        
        params = parse_fuselage_params(fuselage)
        
        # 使用默认参数
        params.radius_x = self.default_radius_x
        params.radius_z = self.default_radius_z
        params.length = params.offset_y * 2
        
        # 从Config元素解析partScale
        config = part_elem.find('Config')
        if config is not None and 'partScale' in config.attrib:
            scale_text = config.attrib['partScale']
            params.part_scale = tuple(float(x) for x in scale_text.split(','))
        
        part_data.fuselage_params = params
        part_data.wall_thickness = 0.045
        part_data.subdivisions = 5
        
        print(f"[Parser]   FairingNoseCone {part_data.part_id}: "
              f"length={params.length:.3f}, wall={part_data.wall_thickness:.3f}")
    
    def _parse_joint_part(self, part_elem: ET.Element, part_data: PartData, part_type: str):
        """解析关节零件（Rotator1/HingeRotator1）
        
        关节零件不需要生成几何体，只需要位置和旋转信息
        """
        # 关节零件不需要 Fuselage 参数
        # 只需要基本的位置和旋转（已在 _parse_part 中解析）
        
        print(f"[Parser]   Joint {part_data.part_id}: "
              f"type={part_type}, "
              f"pos=({part_data.position[0]:.2f}, {part_data.position[1]:.2f}, {part_data.position[2]:.2f})")
    
    def save_to_json(self, output_file: str) -> str:
        """将解析结果保存为JSON文件"""
        data = {
            'metadata': {
                'default_radius_x': self.default_radius_x,
                'default_radius_z': self.default_radius_z,
                'total_parts': len(self.parts),
                'total_materials': len(self.materials),
            },
            'materials': [
                {
                    'name': m.name,
                    'clean_name': clean_material_name(m.name),
                    'color': m.color,
                    'metallic': m.metallic,
                    'roughness': m.roughness
                }
                for m in self.materials
            ],
            'parts': [p.to_dict() for p in self.parts]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[Parser] 解析结果已保存到: {output_file}")
        return output_file


def parse_sr2_xml(xml_file: str, 
                  default_radius_x: float = 1.0,
                  default_radius_z: float = 1.0) -> Dict[str, Any]:
    """
    解析SR2 XML文件的便捷函数
    
    参数:
        xml_file: XML文件路径
        default_radius_x: 默认椭圆短边半径
        default_radius_z: 默认椭圆长边半径
    
    返回:
        包含materials、parts和metadata的字典
    """
    parser = SR2XMLParser(default_radius_x, default_radius_z)
    return parser.parse_file(xml_file)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python parser.py <xml_file> [output_json]")
        sys.exit(1)
    
    xml_file = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else xml_file.replace('.xml', '_parsed.json')
    
    parser = SR2XMLParser()
    result = parser.parse_file(xml_file)
    parser.save_to_json(output_json)
    
    print(f"\n解析完成!")
    print(f"  材质数量: {len(result['materials'])}")
    print(f"  零件数量: {len(result['parts'])}")
