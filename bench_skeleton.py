#!/usr/bin/env python3
import json
import time
import sys

with open('.sr2_cache/USS_Alaska2_4/materials.json', 'r') as f:
    parts_data = json.load(f)['parts']
with open('.sr2_cache/USS_Alaska2_4/connections.json', 'r') as f:
    conn_data = json.load(f)

print(f'零件数: {len(parts_data)}')
print(f'连接数: {len(conn_data["connections"])}')

from skeleton_builder import SkeletonBuilder

t0 = time.time()
builder = SkeletonBuilder(parts_data, conn_data)
data = builder.build()
t1 = time.time()

print(f'骨骼构建耗时: {t1-t0:.2f}s')
print(f'关节数: {len(data["joints"])}')
print(f'绑定数: {len(data["bindings"])}')
