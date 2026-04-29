@echo on
REM Run the OP UT suite on Windows.
REM Required env: CONDA_ENV, GITHUB_WORKSPACE.

call %~dp0_activate.bat
set PYTORCH_TEST_WITH_SLOW=1
set PYTORCH_ENABLE_XPU_FALLBACK=1
set PYTEST_ADDOPTS=-v --timeout 600 --timeout_method=thread --max-worker-restart 1000000 -n 1
cd ..\pytorch\third_party\torch-xpu-ops\test\xpu\
python run_test_with_windows_nighltly.py

if not exist "%GITHUB_WORKSPACE%\ut_log" mkdir "%GITHUB_WORKSPACE%\ut_log"
for /r . %%f in (test*.xml op_ut_with_*.xml) do move "%%f" "%GITHUB_WORKSPACE%\ut_log\" >nul 2>&1 || echo "File move completed"
