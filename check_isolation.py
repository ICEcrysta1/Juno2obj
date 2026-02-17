import re

with open('Output/Test2.usda', 'r') as f:
    content = f.read()

# Find Base_Color mesh section
parts = content.split('def Mesh "Base_Color"')
if len(parts) > 1:
    section = parts[1].split('def ')[0]
    
    # Extract points and normals
    points_match = re.search(r'point3f\[\] points = \[([^\]]+)\]', section, re.DOTALL)
    normals_match = re.search(r'normal3f\[\] normals = \[([^\]]+)\]', section, re.DOTALL)
    
    if points_match and normals_match:
        points = []
        for m in re.finditer(r'\(([^)]+)\)', points_match.group(1)):
            nums = [float(x.strip()) for x in m.group(1).split(',')]
            points.append(nums)
        
        normals = []
        for m in re.finditer(r'\(([^)]+)\)', normals_match.group(1)):
            nums = [float(x.strip()) for x in m.group(1).split(',')]
            normals.append(nums)
        
        # Check NoseCone (part 18) bottom vertices at (6.855, -3.078, -5.265)
        # These should have DOWNWARD normals for the cap
        print('=== Checking NoseCone bottom vertices ===')
        for i, p in enumerate(points):
            if 6.85 < p[0] < 6.86 and -3.10 < p[1] < -3.05 and -5.30 < p[2] < -5.20:
                n = normals[i] if i < len(normals) else [0, 0, 0]
                marker = ''
                if abs(n[1] + 1.0) < 0.1:
                    marker = ' <-- DOWN (cap)!'
                elif abs(n[1] - 1.0) < 0.1:
                    marker = ' <-- UP!'
                print(f'  [{i}] pos=({p[0]:.3f}, {p[1]:.4f}, {p[2]:.3f}) n=({n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}){marker}')
