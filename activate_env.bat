@echo off
REM Activate pytorch environment before any project command
call mamba activate pytorch
echo pytorch env activated | python -c "import sys; print(f'Python: {sys.executable}')"
