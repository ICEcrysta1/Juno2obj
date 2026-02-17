import re

with open('Output/Test-Juno2OBJ.usda', 'r') as f:
    content = f.read()

# Find Base_Color mesh section
parts = content.split('def Mesh "Base_Color"')
section = parts[1].split('def ')[0]

# Extract points and face indices
points_match = re.search(r'point3f\[\] points = \[([^\]]+)\]', section, re.DOTALL)
normals_match = re.search(r'normal3f\[\] normals = \[([^\]]+)\]', section, re.DOTALL)
indices_match = re.search(r'int\[\] faceVertexIndices = \[([^\]]+)\]', section, re.DOTALL)

if points_match and normals_match and indices_match:
    # Parse all points
    points = []
    for m in re.finditer(r'\(([^)]+)\)', points_match.group(1)):
        nums = [float(x.strip()) for x in m.group(1).split(',')]
        points.append(nums)
    
    # Parse all normals
    normals = []
    for m in re.finditer(r'\(([^)]+)\)', normals_match.group(1)):
        nums = [float(x.strip()) for x in m.group(1).split(',')]
        normals.append(nums)
    
    print(f'Total points: {len(points)}')
    print(f'Total normals: {len(normals)}')
    
    # NoseCone 18 is at (6.855002, -2.077645, -5.264554), length=2.0
    # Bottom should be at Y around -3.077645
    print('\n=== NoseCone 18 vertices (X~6.85, Y~-3.08, Z~-5.26) ===')
    
    for i, p in enumerate(points):
        if 6.8 < p[0] < 6.9 and -3.5 < p[1] < -2.5 and -6.0 < p[2] < -4.0:
            n = normals[i] if i < len(normals) else [0, 0, 0]
            ny_marker = ''
            if abs(n[1] + 1.0) < 0.1:
                ny_marker = ' <-- DOWN!'
            elif abs(n[1] - 1.0) < 0.1:
                ny_marker = ' <-- UP!'
            elif abs(n[0]) < 0.1 and abs(n[2]) < 0.1:
                ny_marker = ' <-- VERTICAL!'
            print(f'  [{i:3d}] ({p[0]:.3f}, {p[1]:.4f}, {p[2]:.3f}) n=({n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}){ny_marker}')
