# One-time setup for the local demo (Windows, Python 3.12 + Node 20+).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root

if (-not (Test-Path ".venv")) {
    py -3.12 -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps

Push-Location frontend
npm install
npm run build
Pop-Location

Write-Host "Setup done. Start the demo with: .\scripts\run_demo.ps1"
Pop-Location
