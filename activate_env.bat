@echo off
cd /d "%~dp0"
call C:\Users\Admin\miniforge3\Scripts\activate.bat pytorch
echo.
echo ============================================================
echo  Metal Defect Detection - Environment Ready
echo ============================================================
echo  Conda env  : pytorch
echo  Project    : %CD%
echo  Python     :
where python | findstr miniforge
echo.
echo  Run application: python main.py
echo ============================================================
echo.
cmd /k
