# SimpleRockets 2 to OBJ Converter

将 SimpleRockets 2 (朱诺：新起源) 的飞船设计文件 (.xml) 转换为 OBJ 3D 模型格式的脚本工具。

## 功能特性

- 支持 **Fuselage1**（机身/燃料箱）- 椭圆截面圆柱体
- 支持 **Inlet1**（进气口）- 空心圆柱体
- 支持 **NoseCone1**（鼻锥）- 冯·卡门曲线锥形
- 完整支持 SR2 的形变参数：
  - offset（偏移）
  - topScale/bottomScale（顶部/底部缩放）
  - deformations（形变：水滴形、竖切斜率）
  - cornerRadiuses（圆角半径）
  - clampDistances（层级挤压）
  - partScale（整体缩放）
- 自动导出材质文件 (.mtl)
- 保留原始颜色和材质属性

## 环境要求

- Python 3.8 或更高版本
- NumPy 库

## 安装

1. 克隆或下载本仓库
2. 安装依赖：

```bash
pip install numpy
```

## 文件夹结构

```
Juno2obj/
├── sr2_to_obj.py      # 主脚本
├── Input/             # 输入文件夹（放置 .xml 文件）
├── Output/            # 输出文件夹（生成 .obj 和 .mtl 文件）
└── README.md          # 本文件
```

## 使用方法

### 方法一：使用批处理文件（推荐 Windows 用户）

#### 双击运行（推荐）

1. 将你的 `.xml` 文件放入 `Input` 文件夹
2. 双击运行 `run.bat`
3. 脚本会显示所有可用的 XML 文件列表
4. **直接回车**使用默认文件 `Test-Juno2OBJ.xml`，或**输入文件名**选择其他文件
5. 转换完成后询问是否打开 `Output` 文件夹

**示例交互：**
```
[Input 文件夹中的 XML 文件]
  - Test-Juno2OBJ.xml
  - My Rocket.xml
  - Fighter Craft.xml

[默认] Test-Juno2OBJ.xml （直接回车使用默认）

请输入要转换的文件名: My Rocket.xml
[完成] 转换成功！
```

### 方法二：直接使用 Python 脚本

#### 准备输入文件

1. 在 SimpleRockets 2 中导出飞船设计为 `.xml` 文件
2. 将 `.xml` 文件放入 `Input` 文件夹

#### 运行脚本

**方式一：使用默认文件（推荐新手）**

将输入文件命名为 `Test-Juno2OBJ.xml` 放入 `Input` 文件夹，然后运行：

```bash
python sr2_to_obj.py
```

输出文件将自动生成在 `Output/Test-Juno2OBJ.obj`

**方式二：指定文件名**

```bash
python sr2_to_obj.py "你的飞船.xml" "你的飞船.obj"
```

脚本会自动从 `Input/你的飞船.xml` 读取，输出到 `Output/你的飞船.obj`

> **提示**：如果文件名包含空格，请务必使用双引号 `"` 包裹文件名。

**方式三：指定完整路径**

```bash
python sr2_to_obj.py "C:/path/to/input.xml" "C:/path/to/output.obj"
```

## 示例

```batch
:: 使用批处理（交互式，推荐）
run.bat
```

```bash
# 使用 Python 脚本 - 默认文件
python sr2_to_obj.py

# 指定文件名（推荐加引号，防止空格问题）
python sr2_to_obj.py "My Rocket.xml" "My Rocket.obj"

# 完整路径
python sr2_to_obj.py "F:/My Crafts/ship.xml" "F:/My Models/ship.obj"
```

## 输出文件

脚本会生成两个文件：

- `.obj` - 3D 模型文件（顶点、面、法线、UV）
- `.mtl` - 材质文件（颜色、金属度、粗糙度）

## 参数说明

脚本内置以下默认参数（可在代码中修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `default_radius_x` | 1.0 | 椭圆截面 X 轴半径（短边）|
| `default_radius_z` | 1.0 | 椭圆截面 Z 轴半径（长边）|

## 注意事项

1. **XML 来源**：必须从 SimpleRockets 2 导出 `.xml` 格式的飞船设计文件
2. **材质**：目前支持基础颜色、金属度和粗糙度，不支持纹理贴图
3. **部件类型**：目前仅支持 Fuselage1、Inlet1、NoseCone1，其他部件会被跳过
4. **文件编码**：XML 文件使用 UTF-8 编码

## 故障排除

### 错误：输入文件不存在

```
错误: 输入文件不存在: F:\...\Input\Test-Juno2OBJ.xml
请将 XML 文件放入 Input 文件夹
```

**解决方法**：确保 XML 文件已放入 `Input` 文件夹，且文件名正确。如果文件名包含空格，请用双引号包裹：
```bash
python sr2_to_obj.py "My Ship.xml" "My Ship.obj"
```

### 错误：找不到模块 'numpy'

```
ModuleNotFoundError: No module named 'numpy'
```

**解决方法**：
- **如果使用 `run.bat`**：脚本会自动安装 numpy
- **如果使用 Python 脚本**：运行 `pip install numpy` 安装依赖

### 双击 run.bat 后一闪而过

**解决方法**：
1. 确保已安装 Python，并添加到系统环境变量
2. 在命令行中运行 `run.bat` 查看具体错误信息
3. 检查 `Input` 文件夹中是否有 XML 文件

## 许可证

MIT License