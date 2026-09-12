<#
.SYNOPSIS
    Install DeppSeek into a virtual environment on Windows.

.DESCRIPTION
    Picks an interpreter, creates a venv, installs the package, and reports what
    is and is not available. It deliberately prefers a Python that the MATLAB
    Engine API can bind to, because that choice decides whether MATLAB variables
    persist cheaply between tool calls or every call pays full MATLAB startup.

    MathWorks pins each engine release to one MATLAB release and caps the
    interpreter version. For MATLAB R2026a the engine is matlabengine 26.1.x,
    which declares python_requires ">=3.9, <3.14". A Python 3.14 install cannot
    load it at all.

.PARAMETER Python
    Full path to a specific python.exe to build the venv from.

.PARAMETER NoMatlab
    Skip MATLAB detection and the engine install.

.PARAMETER Full
    Install every optional extra (dashboard, units, notebooks, MCP).

.EXAMPLE
    .\setup.ps1
.EXAMPLE
    .\setup.ps1 -Full
.EXAMPLE
    .\setup.ps1 -Python "C:\Users\aryam\AppData\Local\Programs\Python\Python313\python.exe"
#>

[CmdletBinding()]
param(
    [string]$Python = "",
    [switch]$NoMatlab,
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"

# The interpreter range the current MATLAB engine binds to.
$EngineMin = [version]"3.9"
$EngineMaxExclusive = [version]"3.14"
$MinSupported = [version]"3.11"

function Write-Step { param($m) Write-Host "`n== $m" -ForegroundColor Cyan }
function Write-Good { param($m) Write-Host "   $m" -ForegroundColor Green }
function Write-Warn { param($m) Write-Host "   $m" -ForegroundColor Yellow }
function Write-Bad  { param($m) Write-Host "   $m" -ForegroundColor Red }
function Write-Info { param($m) Write-Host "   $m" -ForegroundColor Gray }

Write-Host ""
Write-Host "  DeppSeek setup" -ForegroundColor Cyan
Write-Host "  --------------" -ForegroundColor DarkCyan

# ---------------------------------------------------------------------------
# 1. Find candidate interpreters
# ---------------------------------------------------------------------------
Write-Step "Looking for Python interpreters"

function Get-PythonVersion {
    param([string]$Exe)
    try {
        $raw = & $Exe -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $raw) { return $null }
        return [version]$raw.Trim()
    } catch { return $null }
}

$candidates = @()

# The py launcher knows about every registered install.
if (Get-Command py -ErrorAction SilentlyContinue) {
    $listed = & py -0p 2>$null
    foreach ($line in $listed) {
        if ($line -match '([A-Za-z]:\\[^\s].*python\.exe)') {
            $candidates += $Matches[1]
        }
    }
}
foreach ($name in @("python", "python3")) {
    $found = Get-Command $name -ErrorAction SilentlyContinue
    if ($found) { $candidates += $found.Source }
}
$candidates = $candidates | Sort-Object -Unique | Where-Object { Test-Path $_ }

if ($candidates.Count -eq 0) {
    Write-Bad "No Python found. Install Python 3.13 from https://www.python.org/downloads/"
    exit 1
}

$interpreters = @()
foreach ($exe in $candidates) {
    $v = Get-PythonVersion $exe
    if ($null -eq $v) { continue }
    $engineOk = ($v -ge $EngineMin) -and ($v -lt $EngineMaxExclusive)
    $interpreters += [pscustomobject]@{
        Path = $exe; Version = $v; EngineCompatible = $engineOk
        Supported = ($v -ge $MinSupported)
    }
    $note = if ($engineOk) { "MATLAB engine OK" } else { "no MATLAB engine" }
    Write-Info ("{0,-8} {1,-18} {2}" -f $v, $note, $exe)
}

$usable = $interpreters | Where-Object { $_.Supported }
if ($usable.Count -eq 0) {
    Write-Bad "No interpreter at $MinSupported or newer. Install Python 3.13."
    exit 1
}

# ---------------------------------------------------------------------------
# 2. Choose one
# ---------------------------------------------------------------------------
Write-Step "Choosing an interpreter"

if ($Python) {
    if (-not (Test-Path $Python)) { Write-Bad "Not found: $Python"; exit 1 }
    $chosen = [pscustomobject]@{
        Path = $Python; Version = (Get-PythonVersion $Python)
        EngineCompatible = $false; Supported = $true
    }
    $chosen.EngineCompatible = ($chosen.Version -ge $EngineMin) -and ($chosen.Version -lt $EngineMaxExclusive)
    Write-Good "Using the interpreter you specified: $($chosen.Version) at $Python"
} else {
    # Prefer the newest engine-compatible interpreter; fall back to the newest
    # usable one. The MATLAB warm workspace is worth more than a point release.
    $engineReady = $usable | Where-Object { $_.EngineCompatible } | Sort-Object Version -Descending
    if ($engineReady) {
        $chosen = $engineReady[0]
        Write-Good "Using Python $($chosen.Version), which the MATLAB engine supports."
    } else {
        $chosen = ($usable | Sort-Object Version -Descending)[0]
        Write-Warn "Using Python $($chosen.Version). No installed interpreter supports the"
        Write-Warn "MATLAB engine (needs >=$EngineMin and <$EngineMaxExclusive)."
        Write-Info "MATLAB will still work, but every call starts a fresh MATLAB and pays"
        Write-Info "its startup cost. Variables persist via a saved workspace file."
        Write-Info ""
        Write-Info "To get the warm workspace, install Python 3.13 and re-run:"
        Write-Info "    winget install Python.Python.3.13"
        Write-Info "    .\setup.ps1"
    }
    Write-Info $chosen.Path
}

# ---------------------------------------------------------------------------
# 3. Virtual environment
# ---------------------------------------------------------------------------
Write-Step "Creating the virtual environment"

if (Test-Path $VenvPath) {
    $existing = Get-PythonVersion (Join-Path $VenvPath "Scripts\python.exe")
    if ($existing -ne $chosen.Version) {
        Write-Warn "Existing .venv is Python $existing but $($chosen.Version) was chosen; recreating."
        Remove-Item -Recurse -Force $VenvPath
    } else {
        Write-Good "Reusing the existing .venv (Python $existing)."
    }
}
if (-not (Test-Path $VenvPath)) {
    & $chosen.Path -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { Write-Bad "venv creation failed."; exit 1 }
    Write-Good "Created $VenvPath"
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

# ---------------------------------------------------------------------------
# 4. Install
# ---------------------------------------------------------------------------
Write-Step "Installing DeppSeek"

& $VenvPython -m pip install --quiet --upgrade pip setuptools wheel
$extras = if ($Full) { ".[all]" } else { "." }
& $VenvPython -m pip install --quiet -e $extras
if ($LASTEXITCODE -ne 0) { Write-Bad "Install failed."; exit 1 }
Write-Good "Installed $(if ($Full) { 'with all optional extras' } else { 'core dependencies' })"
if (-not $Full) {
    Write-Info "Optional extras are not installed. For the dashboard, unit checking,"
    Write-Info "notebooks and MCP, re-run with -Full."
}

# ---------------------------------------------------------------------------
# 5. MATLAB
# ---------------------------------------------------------------------------
if (-not $NoMatlab) {
    Write-Step "Checking MATLAB"

    $matlabExe = $null
    $onPath = Get-Command matlab -ErrorAction SilentlyContinue
    if ($onPath) {
        $matlabExe = $onPath.Source
    } else {
        # A default MATLAB install does not add itself to PATH, so this is the
        # normal case rather than the exception.
        $roots = @("C:\Program Files\MATLAB", "C:\Program Files (x86)\MATLAB")
        foreach ($root in $roots) {
            if (-not (Test-Path $root)) { continue }
            $releases = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending
            foreach ($release in $releases) {
                $exe = Join-Path $release.FullName "bin\matlab.exe"
                if (Test-Path $exe) { $matlabExe = $exe; break }
            }
            if ($matlabExe) { break }
        }
    }

    if (-not $matlabExe) {
        Write-Warn "MATLAB not found. run_matlab will report that when called."
    } else {
        Write-Good "MATLAB: $matlabExe"
        if (-not $onPath) {
            Write-Info "It is not on PATH, but DeppSeek locates it under Program Files."
        }

        if ($chosen.EngineCompatible) {
            Write-Info "Installing the MATLAB Engine API for a warm workspace..."
            & $VenvPython -m pip install --quiet matlabengine 2>$null
            $probe = & $VenvPython -c "import matlab.engine; print('ok')" 2>$null
            if ($probe -match "ok") {
                Write-Good "Warm MATLAB workspace enabled: variables persist between calls."
            } else {
                Write-Warn "The engine did not install cleanly. This usually means the"
                Write-Warn "matlabengine version does not match your MATLAB release."
                Write-Info "Install it from your MATLAB tree instead:"
                $engineDir = Join-Path (Split-Path (Split-Path $matlabExe)) "extern\engines\python"
                Write-Info "    cd `"$engineDir`""
                Write-Info "    `"$VenvPython`" -m pip install ."
            }
        } else {
            Write-Warn "Python $($chosen.Version) cannot load the MATLAB engine."
            Write-Info "MATLAB works through the batch path: each call starts MATLAB fresh,"
            Write-Info "but variables persist via a saved workspace file."
        }
    }
}

# ---------------------------------------------------------------------------
# 6. API key
# ---------------------------------------------------------------------------
Write-Step "API key"

if ($env:DEEPSEEK_API_KEY) {
    Write-Good "DEEPSEEK_API_KEY is set in this session."
} else {
    $persisted = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "User")
    if ($persisted) {
        Write-Good "DEEPSEEK_API_KEY is set for your user account."
        Write-Info "It will be picked up by new PowerShell windows."
    } else {
        Write-Warn "DEEPSEEK_API_KEY is not set. Set it for this window:"
        Write-Info '    $env:DEEPSEEK_API_KEY = "sk-your-key"'
        Write-Info "Or persist it for your account:"
        Write-Info '    [Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "sk-your-key", "User")'
    }
}

# ---------------------------------------------------------------------------
# 7. Done
# ---------------------------------------------------------------------------
Write-Step "Ready"

Write-Host ""
Write-Host "  Run it:" -ForegroundColor White
Write-Host "    .\deppseek.cmd --doctor" -ForegroundColor Gray
Write-Host "    .\deppseek.cmd" -ForegroundColor Gray
Write-Host '    .\deppseek.cmd "why does case3 diverge?"' -ForegroundColor Gray
Write-Host ""
Write-Host "  To use it from any directory, add this folder to your PATH:" -ForegroundColor White
Write-Host "    `$p = [Environment]::GetEnvironmentVariable('Path','User')" -ForegroundColor DarkGray
Write-Host "    [Environment]::SetEnvironmentVariable('Path', `"`$p;$ProjectRoot`", 'User')" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Start with --doctor: it reports anything missing and what it costs you." -ForegroundColor DarkGray
Write-Host ""
