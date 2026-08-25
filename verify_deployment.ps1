# verify_deployment.ps1 — Pre-deployment Performance & Integrity Verification
$ErrorActionPreference = "Stop"

Write-Host "=== Starting PhysioPulse Expert Deployment Verification ===" -ForegroundColor Cyan

# 1. Unit Tests
Write-Host "[1/3] Running Unit Tests..." -ForegroundColor Yellow
pytest tests/test_runtime_paths.py
if ($LASTEXITCODE -ne 0) { Write-Error "Unit tests failed!"; exit 1 }

# 2. Performance SLA Guard
Write-Host "[2/3] Checking Performance SLAs..." -ForegroundColor Yellow
python performance_guard.py
if ($LASTEXITCODE -ne 0) { Write-Error "Performance SLA violation detected!"; exit 1 }

# 3. Model Benchmark Report
Write-Host "[3/3] Generating Detailed Performance Report..." -ForegroundColor Yellow
python benchmark.py
if ($LASTEXITCODE -ne 0) { Write-Error "Benchmark suite failed!"; exit 1 }

Write-Host "✓ DEPLOYMENT VERIFICATION PASSED" -ForegroundColor Green
Write-Host "Model meets all accuracy and hardware SLA requirements."
