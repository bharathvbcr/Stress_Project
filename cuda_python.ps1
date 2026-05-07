param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$python = "C:\conda-data\envs\cuda_torch_env\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "CUDA Python interpreter not found at $python"
    exit 1
}

& $python @Args
exit $LASTEXITCODE
