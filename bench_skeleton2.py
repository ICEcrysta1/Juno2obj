#!/usr/bin/env python3
import json
import time
import sys
from skeleton_builder import SkeletonBuilder

with open('.sr2_cache/USS_Alaska2_4/materials.json', 'r') as f:
    parts_data = json.load(f)['parts']
with open('.sr2_cache/USS_Alaska2_4/connections.json', 'r') as f:
    conn_data = json.load(f)

print(f'零件数: {len(parts_data)}')
print(f'连接数: {len(conn_data["connections"])}')

builder = SkeletonBuilder(parts_data, conn_data)

t0 = time.time()
builder._identify_joints()
t1 = time.time()
print(f'识别关节: {t1-t0:.2f}s, 找到 {len(builder.joints)} 个')

t0 = time.time()
builder._build_joint_hierarchy()
t1 = time.time()
print(f'构建层级: {t1-t0:.2f}s')

t0 = time.time()
builder._calculate_bind_matrices()
t1 = time.time()
print(f'计算矩阵: {t1-t0:.2f}s')

t0 = time.time()
builder._assign_bindings()
t1 = time.time()
print(f'分配绑定: {t1-t0:.2f}s')
