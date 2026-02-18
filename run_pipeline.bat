@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title SimpleRockets 2 to USD Converter (Pipeline Mode)

cd /d "%~dp0"

:: 设置 Python 路径包含本地依赖
set "PYTHONPATH=%~dp0deps;%PYTHONPATH%"

set "DEFAULT_FILE=Test-Juno2OBJ.xml"

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python
    pause
    exit /b 1
)

:: 检查 numpy
python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安装 numpy...
    pip install numpy
    if errorlevel 1 (
        echo [错误] 安装 numpy 失败
        pause
        exit /b 1
    )
    echo [完成] numpy 安装成功
    echo.
)

:: 检测 USD 依赖
python -c "from pxr import Usd, UsdGeom" >nul 2>&1
if errorlevel 1 (
    echo [提示] 未检测到 USD 库，正在自动安装...
    echo.
    
    if not exist "deps" mkdir "deps"
    
    echo [步骤 1/3] 正在下载 usd-core...
    pip download usd-core -d temp_deps --only-binary :all:
    if errorlevel 1 (
        echo [错误] 下载 usd-core 失败，请检查网络连接
        pause
        exit /b 1
    )
    
    echo [步骤 2/3] 正在解压到 deps 目录...
    for %%F in (temp_deps\usd_core-*.whl) do (
        powershell -Command "Expand-Archive -Path '%%F' -DestinationPath 'deps' -Force"
        if errorlevel 1 (
            echo [错误] 解压失败，尝试使用 Python 解压...
            python -c "import zipfile; zipfile.ZipFile('%%F').extractall('deps')"
        )
    )
    
    echo [步骤 3/3] 清理临时文件...
    rmdir /s /q temp_deps 2>nul
    
    :: 验证安装
    python -c "from pxr import Usd, UsdGeom" >nul 2>&1
    if errorlevel 1 (
        echo [错误] USD 库安装失败，请手动安装
        echo [提示] 方法1: 运行命令 pip install usd-core 后复制 site-packages/pxr 到 deps/
        echo [提示] 方法2: 访问 https://github.com/PixarAnimationStudios/OpenUSD 查看安装指南
        pause
        exit /b 1
    )
    
    echo [完成] USD 库安装成功
    echo.
)

if not exist "Input" mkdir "Input"
if not exist "Output" mkdir "Output"

:MAIN_LOOP
cls
echo ============================================
echo  SimpleRockets 2 to USD Converter
echo  [Pipeline Mode - 模块化流水线]
echo ============================================
echo.

echo --------------------------------------------
echo [Input 文件夹中的 XML 文件]
echo --------------------------------------------

set "FILE_COUNT=0"
for %%F in ("Input\*.xml") do (
    set /a FILE_COUNT+=1
    echo   - %%~nxF
)

if %FILE_COUNT%==0 goto NO_FILES
goto START_CONVERT

:NO_FILES
echo   (暂无 XML 文件)
echo.
echo [提示] 请将 XML 文件放入 Input 文件夹
echo.
set /p "OPEN_FOLDER=是否打开 Input 文件夹? (直接回车=Y/N): "
if "!OPEN_FOLDER!"=="" set "OPEN_FOLDER=Y"
if /i "!OPEN_FOLDER!"=="Y" (
    explorer "Input"
    echo.
    echo 请在放入 XML 文件后按任意键继续...
    pause >nul
    goto MAIN_LOOP
)
exit /b 0

:START_CONVERT
echo --------------------------------------------
echo.

if exist "Input\%DEFAULT_FILE%" (
    echo [默认] 直接回车使用: %DEFAULT_FILE%
)
echo.

set /p "USER_INPUT=请输入要转换的文件名: "

if "!USER_INPUT!"=="" (
    if exist "Input\%DEFAULT_FILE%" (
        set "INPUT_FILE=%DEFAULT_FILE%"
        set "OUTPUT_NAME=Test-Juno2OBJ.usda"
    ) else (
        echo.
        echo [错误] 默认文件不存在
        pause
        goto MAIN_LOOP
    )
) else (
    set "INPUT_FILE=!USER_INPUT!"
    for %%A in ("!INPUT_FILE!") do set "FILE_EXT=%%~xA"
    if /i not "!FILE_EXT!"==".xml" set "INPUT_FILE=!INPUT_FILE!.xml"
    
    if not exist "Input\!INPUT_FILE!" (
        echo.
        echo [错误] 文件不存在: Input\!INPUT_FILE!
        pause
        goto MAIN_LOOP
    )
    
    for %%B in ("!INPUT_FILE!") do set "OUTPUT_NAME=%%~nB.usda"
)

echo.
echo ============================================
echo [信息] 输入文件: Input\!INPUT_FILE!
echo [信息] 输出文件: Output\!OUTPUT_NAME!
echo ============================================
echo.

echo [执行] python pipeline.py "!INPUT_FILE!" "!OUTPUT_NAME!"
echo.

python pipeline.py "!INPUT_FILE!" "!OUTPUT_NAME!"

if errorlevel 1 (
    echo.
    echo [错误] 转换失败
    pause
    goto MAIN_LOOP
)

echo.
echo [完成] 转换成功！
echo.
set /p "OPEN_OUTPUT=是否打开 Output 文件夹? (直接回车=Y/N): "
if "!OPEN_OUTPUT!"=="" set "OPEN_OUTPUT=Y"
if /i "!OPEN_OUTPUT!"=="Y" explorer "Output"

goto MAIN_LOOP

endlocal
