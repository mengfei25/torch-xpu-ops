@echo on
REM Prepare a clean PyTorch source checkout in the parent directory and
REM optionally swap in the caller's torch-xpu-ops checkout. Mirrors the
REM previous inline "Prepare Stock Pytorch" cmd block in _windows_ut.yml.
REM
REM Required env:
REM   PYTHON_VERSION       e.g. 3.10
REM   CONDA_ENV            conda env name to (re)create (e.g. windows_ci)
REM   PYTORCH_REPO         resolved by _resolve_specs.bat
REM   PYTORCH_COMMIT       resolved by _resolve_specs.bat
REM   TORCH_XPU_OPS_REPO   resolved by _resolve_specs.bat
REM   TORCH_XPU_OPS_COMMIT resolved by _resolve_specs.bat
REM   GITHUB_EVENT_NAME    'pull_request' triggers local copy from ..\torch-xpu-ops
REM   INSTALL_BUILD_DEPS   when '1', install build-time pip/conda deps (default: 1)

if "%CONDA_ENV%"=="" set CONDA_ENV=windows_ci
if "%PYTHON_VERSION%"=="" set PYTHON_VERSION=3.10
if "%INSTALL_BUILD_DEPS%"=="" set INSTALL_BUILD_DEPS=1

echo "C:\ProgramData\miniforge3\Scripts" >> "%GITHUB_PATH%"
echo "C:\ProgramData\miniforge3\Library\bin" >> "%GITHUB_PATH%"
call "C:\ProgramData\miniforge3\Scripts\activate.bat"
rmdir /s /q "C:\ProgramData\miniforge3\envs\%CONDA_ENV%"
call conda clean -ay
call conda remove --all -y -n %CONDA_ENV%
call conda create -n %CONDA_ENV% python=%PYTHON_VERSION% cmake ninja -y
call conda activate %CONDA_ENV%
cd ..
if exist "pytorch" (
  rmdir /s /q pytorch
)

git clone %PYTORCH_REPO% pytorch
cd pytorch && git checkout %PYTORCH_COMMIT%

if "%INSTALL_BUILD_DEPS%"=="1" (
    pip install pyyaml requests pytest-timeout
    call conda install -y libuv
    call conda install -y rust
)

git config --system core.longpaths true
git config --global core.symlinks true
git config --global core.fsmonitor false
powershell -Command "Set-ItemProperty -Path 'HKLM:\\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1"
git status
git show -s
git submodule sync && git submodule update --init --recursive

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
