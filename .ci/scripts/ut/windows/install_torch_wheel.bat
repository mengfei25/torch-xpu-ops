@echo on
REM Activate vcvars+oneAPI and install the Windows torch wheel that was
REM downloaded to %WHEEL_DIR%.
REM
REM Required env:
REM   CONDA_ENV   conda env name (default: windows_ut)
REM   WHEEL_DIR   directory containing the .whl artifact

if "%CONDA_ENV%"=="" set CONDA_ENV=windows_ut
if "%WHEEL_DIR%"=="" set WHEEL_DIR=%GITHUB_WORKSPACE%\wheel_artifact

call "C:\ProgramData\miniforge3\Scripts\activate.bat"
call conda activate %CONDA_ENV%
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\env\vars.bat"
call "C:\Program Files (x86)\Intel\oneAPI\ocloc\latest\env\vars.bat"
call "C:\Program Files (x86)\Intel\oneAPI\pti\latest\env\vars.bat"

set TORCH_WHL=
for /r "%WHEEL_DIR%" %%f in (*.whl) do set "TORCH_WHL=%%f"
if "%TORCH_WHL%"=="" (
    echo ::error::No .whl found under %WHEEL_DIR%
    exit /b 1
)
pip install "%TORCH_WHL%"
