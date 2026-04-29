@echo on
REM Print torch config + xpu device count. Required env: CONDA_ENV.

if "%CONDA_ENV%"=="" set CONDA_ENV=windows_ut
call "C:\ProgramData\miniforge3\Scripts\activate.bat"
call conda activate %CONDA_ENV%
python -c "import torch; print(torch.__config__.show())"
python -c "import torch; print(torch.__config__.parallel_info())"
python -c "import torch; print(torch.xpu.device_count())"
