# NEXUS local demo — no Docker required.
# Starts FastAPI (:8000) + Next.js (:3000) detached and opens the dashboard.
# First run only: .\scripts\setup_demo.ps1 (or follow README "Setup").
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

function Test-Port([int]$port) {
    try { (Invoke-WebRequest -UseBasicParsing "http://localhost:$port" -TimeoutSec 2) | Out-Null; $true }
    catch { $false }
}

if (-not (Test-Path "$root\.venv\Scripts\python.exe")) {
    Write-Host "Missing .venv — run .\scripts\setup_demo.ps1 first." -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path "$root\frontend\node_modules")) {
    Write-Host "Missing frontend\node_modules — run .\scripts\setup_demo.ps1 first." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Port 8000)) {
    Start-Process -FilePath "$root\.venv\Scripts\python.exe" `
        -ArgumentList "-m", "uvicorn", "nexus.api.main:app", "--port", "8000" `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput "$root\results\backend.log" `
        -RedirectStandardError "$root\results\backend.err.log"
}
if (-not (Test-Port 3000)) {
    if (-not (Test-Path "$root\frontend\.next")) {
        Push-Location "$root\frontend"; npm run build; Pop-Location
    }
    Start-Process -FilePath "npm.cmd" -ArgumentList "run", "start" `
        -WorkingDirectory "$root\frontend" -WindowStyle Hidden `
        -RedirectStandardOutput "$root\results\frontend.log" `
        -RedirectStandardError "$root\results\frontend.err.log"
}

for ($i = 0; $i -lt 20; $i++) {
    if ((Test-Port 8000) -and (Test-Port 3000)) { break }
    Start-Sleep -Seconds 1
}
Write-Host "Backend  http://localhost:8000/health"
Write-Host "Dashboard http://localhost:3000  (opening...)"
Start-Process "http://localhost:3000"
