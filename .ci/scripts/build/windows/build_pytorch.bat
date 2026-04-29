@echo on
REM Build PyTorch XPU on Windows. Replaces the inline "Build Pytorch XPU"
REM cmd block in _windows_ut.yml.
REM
REM Required env:
REM   CONDA_ENV       conda env created by prepare_pytorch.bat
REM   MAX_JOBS        parallel build slots (default: 32)
REM   TORCH_XPU_ARCH_LIST  default: mtl-h,bmg,lnl-m
REM   PYTORCH_DIR     pytorch source dir relative to GITHUB_WORKSPACE (default: ..\pytorch)
REM   BUILD_LOG       log file (default: build_torch_wheel_log.log under pytorch)

if "%CONDA_ENV%"=="" set CONDA_ENV=windows_ci
if "%MAX_JOBS%"=="" set MAX_JOBS=32
if "%TORCH_XPU_ARCH_LIST%"=="" set TORCH_XPU_ARCH_LIST=mtl-h,bmg,lnl-m
if "%PYTORCH_DIR%"=="" set PYTORCH_DIR=..\pytorch
if "%BUILD_LOG%"=="" set BUILD_LOG=build_torch_wheel_log.log

call "C:\ProgramData\miniforge3\Scripts\activate.bat"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\env\vars.bat"
call "C:\Program Files (x86)\Intel\oneAPI\ocloc\latest\env\vars.bat"
call "C:\Program Files (x86)\Intel\oneAPI\pti\latest\env\vars.bat"
call conda activate %CONDA_ENV%

cd %PYTORCH_DIR%
pip install -r requirements.txt
pip install cmake setuptools clang-format
pip install mkl-static mkl-include
set USE_STATIC_MKL=1
copy "%CONDA_PREFIX%\Library\bin\libiomp*5md.dll" .\torch\lib
copy "%CONDA_PREFIX%\Library\bin\uv.dll" .\torch\lib
if defined CMAKE_PREFIX_PATH (
    set CMAKE_PREFIX_PATH="%CONDA_PREFIX%\Library;%CMAKE_PREFIX_PATH%"
) else (
    set CMAKE_PREFIX_PATH="%CONDA_PREFIX%\Library"
)
python setup.py clean

python setup.py bdist_wheel > %BUILD_LOG% 2>&1
set EXIT_CODE=%errorlevel%
if %EXIT_CODE% neq 0 (
    echo "[INFO] Build failed with exit code %EXIT_CODE%"
    exit /b %EXIT_CODE%
)
echo "[INFO] Build Successfully"

echo "[INFO] begin to install torch whls"
for /r %%i in (dist\torch*.whl) do (
    set TORCH_WHL=%%i
)
echo "[INFO] the torch version is %TORCH_WHL%"
python -m pip install %TORCH_WHL%

REM Avoid potential conflicts between hard-coded MKL and the Torch XPU wheel
powershell -Command "(Get-Content '.ci\docker\requirements-ci.txt') | Where-Object { $_ -notmatch 'mkl' } | Set-Content '.ci\docker\requirements-ci.txt'"
pip install -r .ci\docker\requirements-ci.txt
