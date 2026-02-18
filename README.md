# 🚀 SimpleRockets 2 飞船导出器

> 一键将《朱诺：新起源》的飞船设计转换为 3D 模型！

## 📖 这是什么？

这是一个将 **SimpleRockets 2**（朱诺：新起源）游戏中的飞船设计文件 (`.xml`) 转换为 **USD 3D 模型格式**的工具。

转换后的文件可以在 Blender、Maya、Unity、Unreal Engine 等软件中打开使用！

## ✨ 支持的游戏部件

| 部件类型 | 说明 |
|---------|------|
|  Fuselage1 | 机身/燃料箱（实心椭圆圆柱） |
|  Strut1 | 支架/连接杆 |
|  Inlet1 | 进气口（空心） |
|  FairingBase1 | 整流罩底座（空心） |
|  Fairing1 | 整流罩筒（空心） |
|  NoseCone1 | 鼻锥 |
|  FairingNoseCone1 | 整流罩鼻锥（空心） |

## 🚀 快速开始（小白专用）

### 第一步：准备环境

1. **安装 Python**（如果没有的话）
   - 访问 https://www.python.org/downloads/
   - 下载 Python 3.8 或更高版本
   - **⚠️ 安装时务必勾选 "Add Python to PATH"**

2. **下载本工具**
   - 下载本仓库的所有文件，解压到一个文件夹

### 第二步：准备飞船文件

1. **导出作品文件**
   - 存档位于一般在C:\Users\"你的用户名"\AppData\LocalLow\Jundroo\SimpleRockets 2\UserData\CraftDesigns

2. **放入 Input 文件夹**
   - 将导出的 `.xml` 文件复制到本工具的 `Input` 文件夹中

### 第三步：一键转换！

**最简单的方法：双击运行 `run_pipeline.bat`**

```
1. 双击 run_pipeline.bat
2. 程序会自动扫描 Input 文件夹中的飞船文件
3. 输入数字选择要转换的文件
4. 等待转换完成
5. 在 Output 文件夹中找到你的 .usda 文件！
```

## 📂 文件夹说明

```
Juno2obj/
├── run_pipeline.bat     # ⭐ 双击这个开始使用！
├── Input/               # 把游戏的 .xml 文件放这里
│   └── (你的飞船.xml)
├── Output/              # 转换后的 .usda 文件会出现在这里
│   └── (你的飞船.usda)
├── deps/                # USD 库（首次运行自动下载）
└── pipeline.py          # 转换引擎（高级用户使用）
```

## 🔧 高级用法（命令行）

如果你熟悉命令行，也可以直接使用 Python 脚本：

```bash
# 基本用法
python pipeline.py "你的飞船.xml"

# 指定输出文件名
python pipeline.py "你的飞船.xml" "输出名称.usda"

# 调整参数
python pipeline.py "你的飞船.xml" --radius-x 2.0 --segments 36

# 完整参数
python pipeline.py "你的飞船.xml" "输出.usda" ^
    --radius-x 1.5 ^
    --radius-z 2.0 ^
    --segments 48 ^
    --no-normals
```

### 命令行参数说明

| 参数 | 说明 |
|------|------|
| `input` | 输入的 XML 文件名（必填） |
| `output` | 输出的 USD 文件名（可选，默认与输入同名） |
| `--radius-x` | 椭圆短轴半径（默认：1.0） |
| `--radius-z` | 椭圆长轴半径（默认：1.0） |
| `--segments` | 圆柱分段数（默认：24） |
| `--no-normals` | 禁用自定义法线计算 |
| `--keep-cache` | 保留临时缓存文件 |

## 🛠️ 故障排除

### ❌ 双击 run.bat 后一闪而过

**解决方法**：
1. 确保已安装 Python
2. 安装 Python 时**必须勾选 "Add Python to PATH"**
3. 右键 `run_pipeline.bat` → **以管理员身份运行**

### ❌ 提示 "未检测到 Python"

**解决方法**：
1. 访问 https://www.python.org/downloads/
2. 下载并安装 Python 3.8+
3. **⚠️ 安装时勾选 "Add Python to PATH"**
4. 重新打开 run.bat

### ❌ 提示 "USD 库安装失败"

**解决方法**：
1. 确保电脑已连接网络
2. 删除 `deps` 文件夹（如果有的话）
3. 重新运行 run.bat

### ❌ 转换后模型显示不正常

**解决方法**：
1. 检查原始 XML 文件是否正确导出
2. 某些复杂飞船可能包含不支持的部件类型

### ❌ 找不到 Output 文件夹的文件

**解决方法**：
- 转换完成后，在程序中输入 **Y** 可以直接打开 Output 文件夹
- 或者手动在文件资源管理器中打开 `Output` 文件夹

## 📦 打开 USD 文件

转换后的 `.usda` 文件可以在以下软件中打开：

| 软件 | 方法 |
|------|------|
| **Blender** | 可能需要安装 USD 插件，可以直接吧文件拖入窗口|
| **Maya** | 文件 → 导入 → USD |
| **Unity** | 直接拖入 Assets 文件夹 |
| **Unreal** | 启用 USD 插件 → 导入 |

## 📝 技术说明

本项目采用模块化流水线架构：

```
XML → parser.py → generator.py → normal_calculator.py → merger.py → USD
         ↓              ↓                  ↓
   materials.json  cache_gen/*.json  cache_normals/*.json
```

- **parser.py**: 解析游戏 XML 文件
- **generator.py**: 生成 3D 网格顶点数据
- **normal_calculator.py**: 计算表面法线（让模型看起来平滑）
- **merger.py**: 合并所有部件并导出 USD 文件
- **pipeline.py**: 流水线主控，协调所有模块

## 📄 许可证

MIT License - 自由使用，欢迎改进！
