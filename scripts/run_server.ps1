<#
.SYNOPSIS
    Start the JevBERT PoC server on 127.0.0.1:8765.

.DESCRIPTION
    The three things that have to be true before the server will start, in order:

      1. an API key in .env (`python -m jevbert init-env`) - there is no
         unauthenticated mode, and a key shorter than 32 characters is refused;
      2. the pinned model revision in models/ with hashes recorded in the manifest
         (`python -m jevbert fetch-model`) - the server never downloads weights;
      3. the manifest's limits within the configured limits.

    Any of them missing is a startup refusal with a message saying which, not a server
    that comes up and fails per request.

.EXAMPLE
    ./scripts/run_server.ps1
    ./scripts/run_server.ps1 -Port 9000
#>
[CmdletBinding()]
param(
    [string]$Config = "configs/jevbert.poc.yaml",
    [string]$ServerHost = "127.0.0.1",
    [int]$Port = 8765,
    [string]$LogLevel = "info"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".env")) {
    Write-Host "No .env found. Generating an API key ..." -ForegroundColor Yellow
    uv run python -m jevbert init-env
}

$modelDir = "models/MoritzLaurer--bge-m3-zeroshot-v2.0"
if (-not (Test-Path $modelDir)) {
    Write-Host "Model weights are not present. Fetching the pinned revision ..." -ForegroundColor Yellow
    uv run python -m jevbert fetch-model
    if ($LASTEXITCODE -ne 0) { throw "fetch-model failed with exit code $LASTEXITCODE" }
}

Write-Host "Starting JevBERT on http://${ServerHost}:${Port} (config: $Config)" -ForegroundColor Green
Write-Host "  readiness: http://${ServerHost}:${Port}/readyz   (503 until the bundle has warmed up)"
uv run python -m jevbert serve --config $Config --host $ServerHost --port $Port --log-level $LogLevel
