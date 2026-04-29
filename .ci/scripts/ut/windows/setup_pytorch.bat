@echo on
REM Recreate a clean conda env for UT and clone pytorch + torch-xpu-ops on it.
REM Mirrors the inline "Setup Python Environment" cmd block of the ut_test
REM job in _windows_ut.yml.
REM
REM Required env:
REM   PYTHON_VERSION       e.g. 3.10
REM   CONDA_ENV            conda env name (default: windows_ut)
REM   PYTORCH_REPO         resolved by build/windows/_resolve_specs.bat
REM   PYTORCH_COMMIT       resolved by build/windows/_resolve_specs.bat
REM   TORCH_XPU_OPS_REPO   resolved by build/windows/_resolve_specs.bat
REM   TORCH_XPU_OPS_COMMIT resolved by build/windows/_resolve_specs.bat
REM   GITHUB_EVENT_NAME    'pull_request' triggers local copy

if "%PYTHON_VERSION%"=="" set PYTHON_VERSION=3.10
if "%CONDA_ENV%"=="" set CONDA_ENV=windows_ut

echo "C:\ProgramData\miniforge3\Scripts" >> "%GITHUB_PATH%"
echo "C:\ProgramData\miniforge3\Library\bin" >> "%GITHUB_PATH%"
call "C:\ProgramData\miniforge3\Scripts\activate.bat"
rmdir /s /q "C:\ProgramData\miniforge3\envs\%CONDA_ENV%"
call conda clean -ay
call conda remove --all -y -n %CONDA_ENV%
call conda create -n %CONDA_ENV% python=%PYTHON_VERSION% -y
call conda activate %CONDA_ENV%
pip install pyyaml requests pytest-timeout pytest pytest-xdist
call conda install -y libuv
call conda install -y rust

cd ..
if exist "pytorch" (
    rmdir /s /q pytorch
)

git clone %PYTORCH_REPO% pytorch
cd pytorch && git checkout %PYTORCH_COMMIT%
pip install -r .ci\docker\requirements-ci.txt

if "%TORCH_XPU_OPS_COMMIT%"=="pinned" (
    echo "Don't replace torch-xpu-ops!"
) else (
    echo "Replace torch-xpu-ops!"
    cd third_party
    if exist "torch-xpu-ops" (
        rmdir /s /q torch-xpu-ops
    )
    cd ..
    if "%GITHUB_EVENT_NAME%"=="pull_request" (
        Xcopy ..\torch-xpu-ops third_party\torch-xpu-ops /E/H/Y/F/I
    ) else (
        git clone "%TORCH_XPU_OPS_REPO%" "third_party\torch-xpu-ops"
        cd "third_party\torch-xpu-ops"
        git checkout "%TORCH_XPU_OPS_COMMIT%"
        cd ../..
    )
    powershell -Command "(Get-Content caffe2/CMakeLists.txt) -replace 'checkout --quiet \${TORCH_XPU_OPS_COMMIT}', 'log -n 1' | Set-Content caffe2/CMakeLists.txt"
)
