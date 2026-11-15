<#
.SYNOPSIS
  T17 one-command reproduction entry point (Windows PowerShell wrapper).

.DESCRIPTION
  Thin wrapper around scripts/run_all.py; both wrappers expose the same stages
  and safety rules.  Training stages require -AllowTrain and a -GpuLeaseId;
  every GPU stage requires -GpuLeaseId; existing artifacts are skipped unless a
  force switch is given; -DryRun executes nothing (no logs, no results, no model
  load).

.EXAMPLE
  pwsh scripts/run_all.ps1 -ListStages

.EXAMPLE
  pwsh scripts/run_all.ps1 -Stage check -DryRun

.EXAMPLE
  pwsh scripts/run_all.ps1 -Stage deploy

.EXAMPLE
  pwsh scripts/run_all.ps1 -Stage all -DryRun

.EXAMPLE
  pwsh scripts/run_all.ps1 -Stage eval-sft -GpuLeaseId GPU-LEASE-...

.EXAMPLE
  pwsh scripts/run_all.ps1 -Stage eval-sft -CpuEval
#>
[CmdletBinding()]
param(
  [string[]]$Stage = @(),
  [switch]$DryRun,
  [switch]$ListStages,
  [switch]$AllowTrain,
  [switch]$AllowModel,
  [string]$GpuLeaseId,
  [switch]$CpuEval,
  [switch]$ForceRetrain,
  [switch]$ForceMerge,
  [switch]$ForceRerun,
  [switch]$KeepGoing,
  [string]$LogRoot,
  [string]$Python
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$driver = Join-Path $projectRoot 'scripts/run_all.py'

if (-not (Test-Path $driver)) {
  Write-Error "run_all.ps1: driver not found: $driver"
  exit 2
}

$pythonBin = if ($Python) { $Python } elseif ($env:PYTHON) { $env:PYTHON } else { 'python' }
$pythonCommand = Get-Command $pythonBin -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
  Write-Error "run_all.ps1: python executable '$pythonBin' not found (use -Python or set `$env:PYTHON)"
  exit 2
}

$pyArgs = New-Object System.Collections.Generic.List[string]
foreach ($item in $Stage) {
  if ($item -and $item.Trim().Length -gt 0) {
    $pyArgs.Add('--stage')
    $pyArgs.Add($item)
  }
}
if ($DryRun) { $pyArgs.Add('--dry-run') }
if ($ListStages) { $pyArgs.Add('--list-stages') }
if ($AllowTrain) { $pyArgs.Add('--allow-train') }
if ($AllowModel) { $pyArgs.Add('--allow-model') }
if ($GpuLeaseId) { $pyArgs.Add('--gpu-lease-id'); $pyArgs.Add($GpuLeaseId) }
if ($CpuEval) { $pyArgs.Add('--cpu-eval') }
if ($ForceRetrain) { $pyArgs.Add('--force-retrain') }
if ($ForceMerge) { $pyArgs.Add('--force-merge') }
if ($ForceRerun) { $pyArgs.Add('--force-rerun') }
if ($KeepGoing) { $pyArgs.Add('--keep-going') }
if ($LogRoot) { $pyArgs.Add('--log-root'); $pyArgs.Add($LogRoot) }
if ($Python) { $pyArgs.Add('--python'); $pyArgs.Add($Python) }

& $pythonBin $driver @pyArgs
exit $LASTEXITCODE
