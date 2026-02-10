Param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Remove-Target {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        return
    }

    if ($DryRun) {
        Write-Host "[DRY-RUN] remove $Path"
    } else {
        Remove-Item -Recurse -Force $Path
        Write-Host "removed $Path"
    }
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent

Get-ChildItem -Path $root -Recurse -Directory -Force -Filter "__pycache__" |
    ForEach-Object { Remove-Target -Path $_.FullName }

Get-ChildItem -Path $root -Recurse -Directory -Force |
    Where-Object { $_.Name -eq ".ipynb_checkpoints" -or $_.Name -eq "build" -or $_.Name -like "*.egg-info" } |
    ForEach-Object { Remove-Target -Path $_.FullName }

Get-ChildItem -Path $root -Recurse -File -Include "*.pyc","*.pyo","*.lprof" |
    ForEach-Object { Remove-Target -Path $_.FullName }

Write-Host "cleanup finished"
