$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $projectRoot "data\visual_vae_run_20260812"
$corpusRoot = Join-Path $runRoot "corpus_full"
$manifestPath = Join-Path $runRoot "split_manifest_full.json"
$collectionReport = Join-Path $runRoot "collection_report_full.json"
$checkpointPath = Join-Path $runRoot "visual_mark_cvae_full.pt"
$trainingReport = Join-Path $runRoot "training_report_full.json"
$panelPath = Join-Path $runRoot "prediction_panel_full.png"
$logPath = Join-Path $runRoot "overnight_console.log"

New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
Set-Location -LiteralPath $projectRoot
$env:PYTHONUNBUFFERED = "1"

Write-Host "Resuming registered visual corpus collection, then training the visual VAE."
Write-Host "Keep this terminal open and keep the computer connected to AC power."
Write-Host "Run directory: $runRoot"
Write-Host "Log: $logPath"

python -u -m active_painter.visual_collect `
    --output-dir $corpusRoot `
    --manifest $manifestPath `
    --report $collectionReport `
    --trajectories 96 `
    --workers 3 `
    --resume `
    --max-transitions-per-trajectory 8 `
    --max-worker-seconds 21600 `
    --canvas-size 64 `
    --spatial-grid-size 16 `
    --torch-threads 2 `
    --seed 424242 2>&1 | Tee-Object -FilePath $logPath -Append
$collectionExitCode = $LASTEXITCODE
if ($collectionExitCode -ne 0) {
    throw "Visual corpus collection failed with exit code $collectionExitCode. See $logPath."
}

python -u -m active_painter.visual_vae_train `
    --manifest $manifestPath `
    --checkpoint $checkpointPath `
    --report $trainingReport `
    --panel $panelPath `
    --resume `
    --epochs 80 `
    --batch-size 16 `
    --learning-rate 0.0002 `
    --patch-size 64 `
    --latent-dim 16 `
    --base-channels 24 `
    --condition-channels 16 `
    --importance-samples 8 `
    --prior-samples 8 `
    --panel-every-epochs 5 `
    --torch-threads 4 `
    --device auto `
    --seed 2718 2>&1 | Tee-Object -FilePath $logPath -Append
$trainingExitCode = $LASTEXITCODE
if ($trainingExitCode -ne 0) {
    throw "Visual VAE training failed with exit code $trainingExitCode. See $logPath."
}

Write-Host "Overnight visual VAE run completed successfully."
Write-Host "Training report: $trainingReport"
Write-Host "Prediction panel: $panelPath"
