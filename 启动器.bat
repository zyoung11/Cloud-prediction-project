@echo off
setlocal

REM 检查是否已经最小化运行
if "%1"=="minimized" (
    shift
) else (
    start /min cmd /c "%~dpnx0" minimized & exit
)

REM 检查是否安装了 Python
REM 使用 for 循环捕获 python --version 的输出
for /f "delims=" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
REM 如果 PYTHON_VERSION 为空，说明 Python 未安装或不在系统路径中
if "%PYTHON_VERSION%"=="" (
    echo 未找到 Python 请确保已安装并添加到系统路径中
    pause
    exit /b
) else (
    echo Python 版本: %PYTHON_VERSION%
)

REM 运行 app.py
REM 使用 || 运算符捕获 python 命令的错误
python "%~dp0app.py" || (
    echo 运行 app.py 失败，请检查 app.py 文件是否存在且无语法错误
    pause
    exit /b
)
exit