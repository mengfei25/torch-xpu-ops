@echo on
REM Activate vcvars + oneAPI runtime and the conda env. Sourced by every
REM run_*.bat in this directory.
REM
REM Required env:
REM   CONDA_ENV   conda env name (default: windows_ut)
REM Optional env:
REM   ONEAPI_VARS  when '0', skip oneAPI vars (default: 1)

if "%CONDA_ENV%"=="" set CONDA_ENV=windows_ut
if "%ONEAPI_VARS%"=="" set ONEAPI_VARS=1

call "C:\ProgramData\miniforge3\Scripts\activate.bat"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if "%ONEAPI_VARS%"=="1" (
    call "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\env\vars.bat"
    call "C:\Program Files (x86)\Intel\oneAPI\ocloc\latest\env\vars.bat"
    call "C:\Program Files (x86)\Intel\oneAPI\pti\latest\env\vars.bat"
)
call conda activate %CONDA_ENV%
