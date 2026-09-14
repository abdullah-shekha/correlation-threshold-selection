<#
    colreg -- Windows setup (PowerShell).

        cd E:\Data_science\colreg
        powershell -ExecutionPolicy Bypass -File .\setup.ps1

    setup.bat does the same thing and is the more reliable of the two; use this
    only if you prefer PowerShell.

    Written for Windows PowerShell 5.1 compatibility, which means deliberately
    avoiding three things that work in PowerShell 7 but not in 5.1:
      * the call operator applied to a method-call expression, e.g.
        `& $s.Split(" ")[0] $s.Split(" ")[1]` -- 5.1 parses the trailing tokens
        in argument mode and the statement falls apart;
      * String.Split(String, Int32), which only exists on .NET Core, not the
        .NET Framework 4.x that 5.1 runs on;
      * Python f-strings inside double-quoted PowerShell strings, where `{` and
        `$` collide with PowerShell's own syntax.
    Keep it that way when editing.
#>

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Step($msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

# --- Python -----------------------------------------------------------------
Write-Host "colreg setup, package version 2.0" -ForegroundColor Cyan

Step "Finding Python"

# Launcher command and its argument are kept as separate variables from the
# start, so nothing ever needs splitting later.
$pyExe = $null
$pyArg = $null

foreach ($pair in @(@("py", "-3.12"), @("py", "-3.11"), @("python", ""))) {
    $exe = $pair[0]
    $arg = $pair[1]
    try {
        if ($arg -eq "") {
            $out = & $exe --version 2>&1
        }
        else {
            $out = & $exe $arg --version 2>&1
        }
        if ($LASTEXITCODE -eq 0) {
            $pyExe = $exe
            $pyArg = $arg
            Write-Host "Using: $exe $arg  ($out)"
            break
        }
    }
    catch {
        # Launcher or interpreter absent; try the next candidate.
    }
}

if ($null -eq $pyExe) {
    Write-Host "ERROR: No Python found on PATH." -ForegroundColor Red
    Write-Host 'Install Python 3.12 from python.org and tick "Add python.exe to PATH".'
    exit 1
}

# --- venv -------------------------------------------------------------------
Step "Creating virtual environment (.venv)"

$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (Test-Path $vpy) {
    Write-Host ".venv already exists, reusing it"
}
else {
    if ($pyArg -eq "") {
        & $pyExe -m venv .venv
    }
    else {
        & $pyExe $pyArg -m venv .venv
    }
}

if (-not (Test-Path $vpy)) {
    Write-Host "ERROR: could not create the virtual environment at .venv" -ForegroundColor Red
    exit 1
}

# --- dependencies -----------------------------------------------------------
Step "Installing dependencies"

& $vpy -m pip install --upgrade pip --quiet
& $vpy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed. Check your connection and re-run." -ForegroundColor Red
    exit 1
}

# No f-string here: braces and dollar signs inside a double-quoted PowerShell
# string are PowerShell syntax, not Python syntax.
$verScript = "import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))"
$ver = & $vpy -c $verScript
Write-Host "Interpreter in venv: Python $ver"

& $vpy -c "import numba" 2>&1 | Out-Null
$numbaOk = ($LASTEXITCODE -eq 0)

if ($numbaOk) {
    Write-Host "numba OK -- fast path active" -ForegroundColor Green
}
else {
    Write-Host "WARNING: numba is not installed." -ForegroundColor Yellow
    Write-Host "         The study runs roughly 250x slower without it." -ForegroundColor Yellow
    Write-Host "         Numba needs Python 3.11 or 3.12; yours is $ver." -ForegroundColor Yellow
}

# --- folders ----------------------------------------------------------------
Step "Creating working folders"

foreach ($d in @("data\cache", "results\shards", "results\figures")) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    Write-Host "  $d"
}

# --- verification -----------------------------------------------------------
Step "Running scikit-learn parity suite"

& $vpy -m pytest tests -q
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: PARITY TESTS FAILED." -ForegroundColor Red
    Write-Host "Stop here. The estimators disagree with scikit-learn, and a study"
    Write-Host "built on that is not worth running. Send the pytest output."
    exit 1
}

Step "Benchmarking this machine"
& $vpy scripts\benchmark.py

Step "Setup complete"

Write-Host ""
Write-Host "Activate the environment in each new shell:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then, in order:"
Write-Host "    python scripts\verify_corpus.py"
Write-Host "    python scripts\pilot.py"
Write-Host "    python scripts\run_study.py --stage main --limit 20"
Write-Host "    python scripts\run_study.py --stage main --jobs 6"
Write-Host "    python scripts\run_study.py --collect"
Write-Host ""
Write-Host "run_study.py is resumable: Ctrl-C is safe, re-run the same command."
