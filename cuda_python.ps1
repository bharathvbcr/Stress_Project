param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = $env:STRESS_CUDA_PYTHON
if (-not $python) {
    $python = "C:\conda-data\envs\cuda_torch_env\python.exe"
}

if (-not (Test-Path $python)) {
    Write-Error "CUDA Python interpreter not found at $python"
    exit 1
}

# Verify that the configured interpreter can actually import cuda-python bindings
& $python -c "from cuda.bindings import driver, runtime" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "The configured Python interpreter at $python cannot import 'cuda.bindings'."
    Write-Warning "CUDA Python fast-path support will be disabled at runtime and will fall back to standard PyTorch/CPU copying."
}

if (-not $env:STRESS_TORCH_DEVICE) {
    $env:STRESS_TORCH_DEVICE = "cuda"
}

if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"
}
else {
    $env:PYTHONPATH = $repoRoot
}

& $python @Args
exit $LASTEXITCODE
