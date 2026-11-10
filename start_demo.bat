@echo off
REM Starts the web dashboard for the demo and opens it in the browser.
REM Usage: double-click, or run  start_demo.bat  from this folder.
cd /d "%~dp0web" || exit /b 1

if not exist "checkpoints\best_model_attn_sen2naip.pt" (
  echo Missing web\checkpoints\best_model_attn_sen2naip.pt
  pause & exit /b 1
)
if not exist "checkpoints\best_model_attn_gan_sen2naip.pt" (
  echo Missing web\checkpoints\best_model_attn_gan_sen2naip.pt
  pause & exit /b 1
)

set PY=..\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo Starting dashboard at http://localhost:8000  (Ctrl+C to stop)
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8000"
"%PY%" -m uvicorn src.server:app --host 127.0.0.1 --port 8000
pause
