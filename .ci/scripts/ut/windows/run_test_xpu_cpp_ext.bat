@echo on
REM Run the AOT SYCL CPP-extension UT on Windows.
REM Required env: CONDA_ENV, GITHUB_WORKSPACE.

call %~dp0_activate.bat
cd ..\pytorch\third_party\torch-xpu-ops\test\xpu\
python -m pytest test_cpp_extensions_aot_xpu.py -v --junit-xml=test_cpp_extensions_aot_xpu.xml

if not exist "%GITHUB_WORKSPACE%\ut_log" mkdir "%GITHUB_WORKSPACE%\ut_log"
copy test_cpp_extensions_aot_xpu.xml %GITHUB_WORKSPACE%\ut_log /Y
