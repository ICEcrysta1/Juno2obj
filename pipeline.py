#!/usr/bin/env python3
"""
SR2 to USD 转换流水线
整合所有模块：解析 -> 生成 -> 法线计算 -> 合并导出
"""

import os
import sys
import json
import shutil
import argparse
from typing import Optional

from models import clean_material_name
from parser import SR2XMLParser
from generator import MeshGenerator
from normal_calculator import NormalCalculator
from merger import MeshMerger


class SR2ToUSDPipeline:
    """SR2到USD转换流水线"""
    
    def __init__(self, 
                 default_radius_x: float = 1.0,
                 default_radius_z: float = 1.0,
                 segments: int = 24,
                 use_custom_normals: bool = True,
                 keep_cache: bool = False):
        """
        初始化流水线
        
        参数:
            default_radius_x: 默认椭圆短边半径
            default_radius_z: 默认椭圆长边半径
            segments: 圆柱分段数
            use_custom_normals: 是否使用自定义法线
            keep_cache: 是否保留缓存文件
        """
        self.default_radius_x = default_radius_x
        self.default_radius_z = default_radius_z
        self.segments = segments
        self.use_custom_normals = use_custom_normals
        self.keep_cache = keep_cache
        
        # 子模块
        self.parser = SR2XMLParser(default_radius_x, default_radius_z)
        self.generator = MeshGenerator(segments)
        self.normal_calculator = NormalCalculator(use_smooth_normals=True)
        self.merger = MeshMerger()
    
    def run(self, xml_file: str, output_file: str, mesh_prefix: Optional[str] = None) -> bool:
        """
        运行完整流水线
        
        参数:
            xml_file: 输入XML文件
            output_file: 输出USD文件
            mesh_prefix: 网格名称前缀（默认从XML文件名提取）
        
        返回:
            是否成功
        """
        if mesh_prefix is None:
            mesh_prefix = os.path.splitext(os.path.basename(xml_file))[0]
            mesh_prefix = clean_material_name(mesh_prefix)
        
        # 创建缓存目录
        base_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.sr2_cache')
        cache_dir = os.path.join(base_cache_dir, mesh_prefix)
        gen_cache_dir = os.path.join(cache_dir, 'gen')
        normal_cache_dir = os.path.join(cache_dir, 'normals')
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(gen_cache_dir, exist_ok=True)
        os.makedirs(normal_cache_dir, exist_ok=True)
        
        try:
            # ===== 步骤1: 解析XML =====
            print("=" * 60)
            print("步骤 1/4: 解析XML文件")
            print("=" * 60)
            
            result = self.parser.parse_file(xml_file)
            parts = result['parts']
            materials = result['materials']
            
            if not parts:
                print("[Pipeline] 错误: 没有找到有效的零件")
                return False
            
            # 保存解析结果供后续使用
            materials_file = os.path.join(cache_dir, 'materials.json')
            self.parser.save_to_json(materials_file)
            
            # ===== 步骤2: 生成网格 =====
            print("\n" + "=" * 60)
            print("步骤 2/4: 生成网格")
            print("=" * 60)
            
            raw_meshes = self.generator.generate_from_parts(parts)
            if not raw_meshes:
                print("[Pipeline] 错误: 没有生成任何网格")
                return False
            
            self.generator.save_cache(gen_cache_dir)
            
            # ===== 步骤3: 计算法线 =====
            print("\n" + "=" * 60)
            print("步骤 3/4: 计算法线")
            print("=" * 60)
            
            if self.use_custom_normals:
                normal_meshes = self.normal_calculator.process_cache_files(gen_cache_dir)
                if not normal_meshes:
                    print("[Pipeline] 警告: 法线计算失败，将使用自动法线")
                    self.use_custom_normals = False
                else:
                    self.normal_calculator.save_cache(normal_cache_dir)
                    processed_cache_dir = normal_cache_dir
            else:
                print("[Pipeline] 跳过自定义法线计算，使用自动法线")
                processed_cache_dir = gen_cache_dir
            
            # ===== 步骤4: 合并导出 =====
            print("\n" + "=" * 60)
            print("步骤 4/4: 合并并导出USD")
            print("=" * 60)
            
            self.merger.load_normal_cache(processed_cache_dir)
            self.merger.load_materials(materials_file)
            self.merger.merge_and_export(output_file, mesh_prefix, self.use_custom_normals)
            
            # 统计信息
            total_vertices = sum(len(m.vertices) for m in self.merger.meshes)
            total_faces = sum(len(m.faces) for m in self.merger.meshes)
            
            print("\n" + "=" * 60)
            print("转换完成!")
            print("=" * 60)
            print(f"输出文件: {output_file}")
            print(f"总顶点数: {total_vertices}")
            print(f"总面数: {total_faces}")
            print(f"材质数量: {len(materials)}")
            
            # 清理缓存
            if not self.keep_cache:
                print(f"[Pipeline] 清理缓存目录: {cache_dir}")
                shutil.rmtree(cache_dir, ignore_errors=True)
            else:
                print(f"[Pipeline] 缓存保留在: {cache_dir}")
            
            return True
            
        except Exception as e:
            print(f"[Pipeline] 错误: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='SimpleRockets 2 到 USD 转换器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python pipeline.py input.xml output.usda
  python pipeline.py input.xml --radius-x 2.0 --radius-z 1.5
  python pipeline.py input.xml --no-normals --keep-cache
        '''
    )
    
    parser.add_argument('input', help='输入XML文件路径')
    parser.add_argument('output', nargs='?', help='输出USD文件路径（默认为Output目录）')
    parser.add_argument('--radius-x', type=float, default=1.0, help='默认椭圆短边半径（默认: 1.0）')
    parser.add_argument('--radius-z', type=float, default=1.0, help='默认椭圆长边半径（默认: 1.0）')
    parser.add_argument('--segments', type=int, default=24, help='圆柱分段数（默认: 24）')
    parser.add_argument('--no-normals', action='store_true', help='不使用自定义法线')
    parser.add_argument('--keep-cache', action='store_true', help='保留缓存文件')
    parser.add_argument('--prefix', help='网格名称前缀（默认从文件名提取）')
    
    args = parser.parse_args()
    
    # 处理输入文件
    xml_file = args.input
    if not os.path.isabs(xml_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, 'Input')
        xml_file = os.path.join(input_dir, xml_file)
    
    if not os.path.exists(xml_file):
        print(f"错误: 输入文件不存在: {xml_file}")
        sys.exit(1)
    
    # 处理输出文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'Output')
    os.makedirs(output_dir, exist_ok=True)
    
    if args.output:
        # 如果输出路径不是绝对路径且没有目录分隔符，放到 Output 目录下
        if not os.path.isabs(args.output) and not os.path.dirname(args.output):
            output_file = os.path.join(output_dir, args.output)
        else:
            output_file = args.output
    else:
        basename = os.path.splitext(os.path.basename(xml_file))[0]
        output_file = os.path.join(output_dir, f"{basename}.usda")
    
    # 创建并运行流水线
    pipeline = SR2ToUSDPipeline(
        default_radius_x=args.radius_x,
        default_radius_z=args.radius_z,
        segments=args.segments,
        use_custom_normals=not args.no_normals,
        keep_cache=args.keep_cache
    )
    
    success = pipeline.run(xml_file, output_file, args.prefix)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
