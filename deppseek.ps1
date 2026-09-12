<#
    DeppSeek launcher for PowerShell.

    Exists alongside deppseek.cmd because a .cmd shim re-enters cmd.exe, which
    mangles quoting on arguments containing characters PowerShell treats
    normally. Calling the interpreter directly preserves them.
#>
$ErrorActionPreference = "Stop"
$Home_ = $PSScriptRoot
$VenvPython = Join-Path $Home_ ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    & $VenvPython -m deppseek.cli @args
} else {
    Write-Host "[deppseek] No virtual environment found. Run setup.ps1 first." -ForegroundColor Yellow
    & python -m deppseek.cli @args
}
exit $LASTEXITCODE
