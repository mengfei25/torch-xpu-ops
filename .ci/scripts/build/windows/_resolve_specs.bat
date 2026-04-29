@echo on
REM Resolve PyTorch and torch-xpu-ops repo/commit pairs from caller-provided
REM PYTORCH and TORCH_XPU_OPS spec strings, then write them as PYTORCH_REPO,
REM PYTORCH_COMMIT, TORCH_XPU_OPS_REPO and TORCH_XPU_OPS_COMMIT to GITHUB_ENV.
REM
REM Required env:
REM   PYTORCH        spec: "branch", "repo@branch", or "https://...@commit"
REM   TORCH_XPU_OPS  spec: "branch", "repo@branch", or "https://...@commit"
REM   GITHUB_ENV     path to the GitHub environment file

if "%PYTORCH%"=="" exit /b 2
if "%TORCH_XPU_OPS%"=="" exit /b 2

set PYTORCH_REPO=
set PYTORCH_COMMIT=
set TORCH_XPU_OPS_REPO=
set TORCH_XPU_OPS_COMMIT=

echo %PYTORCH% | findstr /C:"https://" >nul
if %errorlevel% equ 0 (
    echo %PYTORCH% | findstr "@" >nul
    if %errorlevel% equ 0 (
        for /f "tokens=1,2 delims=@" %%a in ("%PYTORCH%") do (
            set PYTORCH_REPO=%%a
            set PYTORCH_COMMIT=%%b
        )
    ) else (
        set PYTORCH_REPO=%PYTORCH%
        set PYTORCH_COMMIT=main
    )
) else (
    set PYTORCH_REPO=https://github.com/pytorch/pytorch.git
    set PYTORCH_COMMIT=%PYTORCH%
)

echo %TORCH_XPU_OPS% | findstr /C:"https://" >nul
if %errorlevel% equ 0 (
    echo %TORCH_XPU_OPS% | findstr "@" >nul
    if %errorlevel% equ 0 (
        for /f "tokens=1,2 delims=@" %%a in ("%TORCH_XPU_OPS%") do (
            set TORCH_XPU_OPS_REPO=%%a
            set TORCH_XPU_OPS_COMMIT=%%b
        )
    ) else (
        set TORCH_XPU_OPS_REPO=%TORCH_XPU_OPS%
        set TORCH_XPU_OPS_COMMIT=main
    )
) else (
    set TORCH_XPU_OPS_REPO=https://github.com/intel/torch-xpu-ops.git
    set TORCH_XPU_OPS_COMMIT=%TORCH_XPU_OPS%
)

echo PYTORCH_REPO=%PYTORCH_REPO%
echo PYTORCH_COMMIT=%PYTORCH_COMMIT%
echo TORCH_XPU_OPS_REPO=%TORCH_XPU_OPS_REPO%
echo TORCH_XPU_OPS_COMMIT=%TORCH_XPU_OPS_COMMIT%

if not "%GITHUB_ENV%"=="" (
    >> "%GITHUB_ENV%" echo PYTORCH_REPO=%PYTORCH_REPO%
    >> "%GITHUB_ENV%" echo PYTORCH_COMMIT=%PYTORCH_COMMIT%
    >> "%GITHUB_ENV%" echo TORCH_XPU_OPS_REPO=%TORCH_XPU_OPS_REPO%
    >> "%GITHUB_ENV%" echo TORCH_XPU_OPS_COMMIT=%TORCH_XPU_OPS_COMMIT%
)
