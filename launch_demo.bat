@echo off
REM ============================================================
REM  NEXUS live demonstration launcher (Windows, no Docker)
REM  1. Starts FastAPI backend  (:8000) in a minimized window
REM  2. Starts Next.js frontend (:3000) in a minimized window
REM  3. Waits for both health checks
REM  4. Opens the dashboard in the default browser
REM  5. Runs a real 100-step episode (seed 42) via the live API
REM     and saves it to results\live_demo_100.json
REM  Double-click this file, or run:  launch_demo.bat
REM  To stop the demo later, close the two minimized windows
REM  titled "NEXUS backend :8000" and "NEXUS frontend :3000".
REM ============================================================
setlocal
cd /d "%~dp0"
title NEXUS live demo launcher

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Missing .venv - run scripts\setup_demo.ps1 first.
  pause
  exit /b 1
)
if not exist "frontend\node_modules" (
  echo [ERROR] Missing frontend\node_modules - run scripts\setup_demo.ps1 first.
  pause
  exit /b 1
)
if not exist "frontend\.next\BUILD_ID" (
  echo [SETUP] Building frontend once, please wait...
  pushd frontend
  call npm run build
  popd
)

echo [1/5] Starting backend  http://localhost:8000 ...
start "NEXUS backend :8000" /min cmd /c "cd /d %~dp0 && .venv\Scripts\python.exe -m uvicorn nexus.api.main:app --port 8000"

echo [2/5] Starting frontend http://localhost:3000 ...
start "NEXUS frontend :3000" /min cmd /c "cd /d %~dp0frontend && npm run start"

echo [3/5] Waiting for servers...
powershell -NoProfile -Command "$ok=$false; for($i=0;$i -lt 60;$i++){ try{ $b=(Invoke-WebRequest -UseBasicParsing http://localhost:8000/health -TimeoutSec 2).StatusCode; $f=(Invoke-WebRequest -UseBasicParsing http://localhost:3000 -TimeoutSec 2).StatusCode; if($b -eq 200 -and $f -eq 200){$ok=$true;break} }catch{}; Start-Sleep -Seconds 1 }; if(-not $ok){ Write-Host 'Servers did not come up in 60s.'; exit 1 }"
if errorlevel 1 (
  echo [ERROR] Servers failed to start. Check the two minimized windows.
  pause
  exit /b 1
)

echo [4/5] Opening dashboard in your browser...
start "" "http://localhost:3000/marketplace"

echo [5/5] Running live 100-step episode (seed 42) - creating actual results...
".venv\Scripts\python.exe" scripts\run_live_episode.py
if errorlevel 1 (
  echo [ERROR] Episode run failed - is the backend healthy at :8000?
  pause
  exit /b 1
)

echo.
echo ============================================================
echo  DEMO READY
echo  Dashboard : http://localhost:3000/marketplace  (already open)
echo  Backend   : http://localhost:8000/health
echo  Results   : results\live_demo_100.json
echo  Press any key to close THIS launcher window.
echo  The servers keep running in their minimized windows.
echo ============================================================
pause
endlocal
