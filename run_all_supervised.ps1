# Full 5-fold, fixed-seed (42) queue: our data 3-class, our data binary, then the 5 public datasets.
# Each step is restarted after a crash (a sticky CUDA error kills the process; finished folds are skipped on restart,
# the interrupted fold is retrained from scratch). A step ends when it exits cleanly with no unfinished failed fold,
# or after 20 attempts.
$b = $PSScriptRoot; $py = $(if ($env:TUSEB_PYTHON) { $env:TUSEB_PYTHON } else { "python" })
$steps = @(
    @{ name = 'ours_3class'; dir = $b; args = '02_train_cv.py'; res = "$b\results\trained_models"; log = "$b\results\logs" },
    @{ name = 'ours_binary'; dir = $b; args = '02_train_cv.py --task binary'; res = "$b\results\binary\trained_models"; log = "$b\results\binary\logs" },
    @{ name = 'public'; dir = "$b\public"; args = '02_train_public.py --datasets OCD3_2class OCD2_2class NDO_2class NDO_3class NDO_4class';
       res = "$b\public\results_public\*\trained_models"; log = "$b\public\results_public\logs" }
)
$sup = "$b\supervisor.log"
foreach ($s in $steps) {
    New-Item -ItemType Directory -Force $s.log | Out-Null
    for ($i = 1; $i -le 20; $i++) {
        "$(Get-Date -Format s) $($s.name) attempt $i" | Out-File -Append -Encoding utf8 $sup
        Push-Location $s.dir
        cmd /c "`"$py`" -W ignore -u $($s.args) >> `"$($s.log)\train_stdout.log`" 2>> `"$($s.log)\train_stderr.log`""
        $code = $LASTEXITCODE; Pop-Location
        $failed = @(Get-ChildItem "$($s.res)\*\fold_*\error.txt" -ErrorAction SilentlyContinue |
            Where-Object { -not (Test-Path (Join-Path $_.DirectoryName 'done.json')) }).Count
        "$(Get-Date -Format s) $($s.name) exit $code, unfinished-with-error $failed" | Out-File -Append -Encoding utf8 $sup
        if ($code -eq 0 -and $failed -eq 0) { break }
        Start-Sleep 60
    }
}
"$(Get-Date -Format s) queue finished" | Out-File -Append -Encoding utf8 $sup
