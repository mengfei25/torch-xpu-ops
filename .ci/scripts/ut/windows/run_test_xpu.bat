@echo on
REM Run the test_xpu UT on Windows. (Currently disabled by caller; kept for parity.)
REM Required env: CONDA_ENV, GITHUB_WORKSPACE.

set ONEAPI_VARS=0
call %~dp0_activate.bat
cd ..\pytorch\third_party\torch-xpu-ops\test\xpu\
python run_test_win_with_skip_mtl.py

if not exist "%GITHUB_WORKSPACE%\ut_log" mkdir "%GITHUB_WORKSPACE%\ut_log"
copy test_xpu.xml %GITHUB_WORKSPACE%\ut_log /Y
