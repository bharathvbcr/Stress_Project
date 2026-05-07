param(
    [string]$RepoRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\')
$indexPath = Join-Path $root "docs/repo-map.md"

if (-not (Test-Path -LiteralPath $indexPath)) {
    throw "Expected repository index not found: $indexPath"
}

$mapLines = Get-Content -LiteralPath $indexPath
$pathLines = foreach($line in $mapLines) {
    if($line -match '^- `([^`]+)`\s*$') {
        $matches[1]
    }
}

$filePaths = $pathLines | Where-Object { $_ -ne $null -and $_.Trim() } | Select-Object -Unique

function Get-Ownership($path) {
    if($path -like "timesfm/v1/*" -or $path -like "timesfm/timesfm-forecasting/examples/*") { return "Legacy / archive" }
    if($path -in @("README.md","docs/repo-map.md","docs/repo-map.json","docs/repo-map-refresh.ps1")) { return "Documentation" }
    if($path -like "conf/*") { return "Data ingestion / dataset construction" }
    if($path -in @("main.py","models.py","run_pipeline.py","run_pipeline_timesfm.py","run_lightning.py","training.py","tuning.py","timesfm_wrapper.py","lightning_data.py","lightning_module.py")) { return "Modeling / training" }
    if($path -in @("data_loader.py","data_pipeline.py","data_splitting.py","preprocessing.py","windowing.py")) { return "Data ingestion / dataset construction" }
    if($path -in @("feature_engineering.py","feature_extraction.py","signal_processing.py","utils.py","pytorch_datasets.py")) { return "Feature engineering / signal processing" }
    if($path -in @("api.py","evaluation.py","validation.py","benchmark.py","sampling.py","visualization.py","losses.py","export_trt.py")) { return "Evaluation / inference" }
    if($path -like "timesfm/AGENTS.md" -or $path -like "timesfm/README.md" -or $path -like "timesfm/pyproject.toml" -or $path -like "timesfm/src/*" -or $path -like "timesfm/timesfm-forecasting/SKILL.md" -or $path -like "timesfm/timesfm-forecasting/references/*" -or $path -like "timesfm/timesfm-forecasting/scripts/*" -or $path -like "timesfm/.gitattributes" -or $path -like "timesfm/.gitignore" -or $path -like "timesfm/requirements.txt" -or $path -like "timesfm/.github/*" -or $path -like "timesfm/LICENSE") { return "TimesFM package / packaging" }
    if($path -in @("AGENTS.md","CLAUDE.md",".claude/settings.local.json",".devcouncil/config.yaml",".devcouncil/state.sqlite",".gitignore","RESUME_STAR_POINTS.md")) { return "Project governance / meta" }
    if($path -like "tests/*") { return "Tests" }
    if($path -like "docs/*") { return "Documentation" }
    return "Shared utilities / orchestration"
}

function Get-Category($path) {
    $ext = [System.IO.Path]::GetExtension($path)
    switch($ext.ToLowerInvariant()) {
        ".py" { return "python" }
        ".yaml" { return "yaml" }
        ".yml" { return "yaml" }
        ".json" { return "json" }
        ".md" { return "markdown" }
        ".ipynb" { return "notebook" }
        ".png" { return "image" }
        ".gif" { return "image" }
        ".csv" { return "data" }
        ".sh" { return "shell" }
        ".ps1" { return "shell" }
        ".toml" { return "config" }
        ".txt" { return "text" }
        ".lock" { return "lock" }
        ".sqlite" { return "database" }
        ".db" { return "database" }
        ".sqlite3" { return "database" }
        default { if([string]::IsNullOrWhiteSpace($ext)) { return "file" }; return "other" }
    }
}

$entries = New-Object System.Collections.Generic.List[psobject]

foreach($rel in $filePaths) {
    $full = Join-Path $root $rel
    if(-not (Test-Path -LiteralPath $full)) { continue }
    $normPath = $rel.Replace('\','/')
    $dir = Split-Path -Path $normPath -Parent
    $dir = $dir.Replace('\','/')
    if([string]::IsNullOrWhiteSpace($dir)) { $dir = "." }

    $entries.Add([PSCustomObject]@{
        path = $normPath
            absolute_path = (Resolve-Path -LiteralPath $full).Path.Replace('\','/')
        directory = $dir
        name = [System.IO.Path]::GetFileName($normPath)
        extension = [System.IO.Path]::GetExtension($normPath)
        category = Get-Category $normPath
        ownership = Get-Ownership $normPath
    })
}

$dirs = $entries | Sort-Object path | Group-Object directory | Sort-Object Name | ForEach-Object {
    [ordered]@{
        directory = $_.Name
        file_count = $_.Count
        files = $_.Group | Sort-Object path | ForEach-Object { $_.path }
    }
}

$map = [ordered]@{
    schema_version = "stressproject-repo-map-json/1.0"
    generated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
    generated_by = "repo-map-refresh.ps1"
    repo_root = $root
    source_index = "docs/repo-map.md"
    canonical = $true
    file_count = $entries.Count
    include_filters = @("exclude outputs","exclude scratch","exclude __pycache__",".venv",".git")
    ownership_matrix = @(
        [ordered]@{ name = "data"; files = @("data_loader.py","data_pipeline.py","data_splitting.py","preprocessing.py","windowing.py","conf/*") }
        [ordered]@{ name = "feature"; files = @("feature_engineering.py","feature_extraction.py","signal_processing.py","utils.py","pytorch_datasets.py") }
        [ordered]@{ name = "modeling"; files = @("main.py","models.py","run_pipeline.py","run_pipeline_timesfm.py","run_lightning.py","training.py","tuning.py","timesfm_wrapper.py","lightning_data.py","lightning_module.py") }
        [ordered]@{ name = "inference"; files = @("api.py","evaluation.py","validation.py","benchmark.py","sampling.py","visualization.py","losses.py","export_trt.py") }
        [ordered]@{ name = "documentation"; files = @("README.md","docs/*") }
        [ordered]@{ name = "tests"; files = @("tests/*") }
        [ordered]@{ name = "project_governance"; files = @("AGENTS.md","CLAUDE.md",".claude/*",".devcouncil/*",".gitignore","RESUME_STAR_POINTS.md") }
        [ordered]@{ name = "shared_utilities"; files = @("config.json","cuda_python.ps1","convert_to_hf.py","dashboard.py","dvc_init.py","widget_setup.py","Baseline_Calibration_for_Stress_Response.ipynb","gitnexus-analyze.err.log","gitnexus-analyze.out.log","pytest.ini","requirements.txt") }
        [ordered]@{ name = "timesfm"; files = @("timesfm/AGENTS.md","timesfm/README.md","timesfm/pyproject.toml","timesfm/src/timesfm/*","timesfm/timesfm-forecasting/SKILL.md","timesfm/timesfm-forecasting/references/*","timesfm/timesfm-forecasting/scripts/*","timesfm/.gitattributes","timesfm/.github/*","timesfm/.gitignore","timesfm/LICENSE","timesfm/requirements.txt") }
        [ordered]@{ name = "legacy"; files = @("timesfm/v1/**","timesfm/timesfm-forecasting/examples/**") }
    )
    call_dependency_anchors = [ordered]@{
        main_to_modeling = @("main.py","utils.py","data_pipeline.py","models.py","lightning_module.py")
        inference_api_flow = @("api.py","initialize_model_state","get_model","/predict")
        training_paths = @("run_pipeline_timesfm.py","run_pipeline.py","run_lightning.py")
    }
    high_coupling_candidates = @("utils.py","models.py","preprocessing.py","data_pipeline.py","training.py","lightning_module.py","run_pipeline_timesfm.py")
    directories = $dirs
    files = @($entries | Sort-Object path)
}

$jsonPath = Join-Path $root "docs/repo-map.json"
Set-Content -Encoding UTF8 -Path $jsonPath -Value ($map | ConvertTo-Json -Depth 10)
Write-Output "repo-map.json updated: $jsonPath ($($entries.Count) files)"
