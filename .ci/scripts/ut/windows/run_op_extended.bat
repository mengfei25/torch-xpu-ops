@echo on
REM Run the OP-extended UT suite on Windows.
REM Required env: CONDA_ENV, GITHUB_WORKSPACE.

call %~dp0_activate.bat
set PYTORCH_TEST_WITH_SLOW=1
cd ..\pytorch\third_party\torch-xpu-ops\test\xpu\extended\
python run_test_with_skip_mtl.py

if not exist "%GITHUB_WORKSPACE%\ut_log" mkdir "%GITHUB_WORKSPACE%\ut_log"
copy op_extended.xml %GITHUB_WORKSPACE%\ut_log /Y
