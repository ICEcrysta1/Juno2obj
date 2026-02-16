@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title SimpleRockets 2 to OBJ Converter

cd /d "%~dp0"

set "DEFAULT_FILE=Test-Juno2OBJ.xml"

python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python
    pause
    exit /b 1
)

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

if not exist "Input" mkdir "Input"
if not exist "Output" mkdir "Output"

:MAIN_LOOP
cls
echo ============================================
echo  SimpleRockets 2 to OBJ Converter
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
goto HAS_FILES

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

:HAS_FILES
echo --------------------------------------------
echo.

if exist "Input\%DEFAULT_FILE%" (
    echo [默认] %DEFAULT_FILE% (直接回车使用默认)
)
echo.

set /p "USER_INPUT=请输入要转换的文件名: "

if "!USER_INPUT!"=="" (
    if exist "Input\%DEFAULT_FILE%" (
        set "INPUT_FILE=%DEFAULT_FILE%"
        set "OUTPUT_NAME=Test-Juno2OBJ.obj"
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
    
    for %%B in ("!INPUT_FILE!") do set "OUTPUT_NAME=%%~nB.obj"
)

echo.
echo ============================================
echo [信息] 输入文件: Input\!INPUT_FILE!
echo [信息] 输出文件: Output\!OUTPUT_NAME!
echo ============================================
echo.

python sr2_to_obj.py "!INPUT_FILE!" "!OUTPUT_NAME!"

if errorlevel 1 (
    echo.
    echo [错误] 转换失败
    pause
    exit /b 1
)

echo.
echo [完成] 转换成功！
echo.
set /p "OPEN_OUTPUT=是否打开 Output 文件夹? (直接回车=Y/N): "
if "!OPEN_OUTPUT!"=="" set "OPEN_OUTPUT=Y"
if /i "!OPEN_OUTPUT!"=="Y" explorer "Output"

endlocal
