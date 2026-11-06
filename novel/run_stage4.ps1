# Stage 4 (decided 2026-11-03): Design A (A5 = KD + SupCon, A6 = + cRW) on the larger public datasets,
# with the same IN-12k init as stage 3 (reference A2b). OCD2 first (fast), then the three NDO tasks.
# Restarted after a crash (finished folds are skipped), up to 20 attempts. Log: stage4.log
$b = $PSScriptRoot; $py = $(if ($env:TUSEB_PYTHON) { $env:TUSEB_PYTHON } else { "python" })
$log = "$b\stage4.log"; New-Item -ItemType Directory -Force "$b\logs" | Out-Null
"$(Get-Date -Format s) starting stage 4" | Out-File -Append -Encoding utf8 $log
$tasks = 'OCD2_2class NDO_2class NDO_3class NDO_4class'
Push-Location $b
for ($i = 1; $i -le 20; $i++) {
    "$(Get-Date -Format s) stage4 attempt $i" | Out-File -Append -Encoding utf8 $log
    cmd /c "`"$py`" -W ignore -u train_novel.py --tasks $tasks --rows A5 A6 >> `"$b\logs\stage4_stdout.log`" 2>> `"$b\logs\stage4_stderr.log`""
    $code = $LASTEXITCODE
    $failed = @(Get-ChildItem "$b\results\*\A[56]\fold_*\error.txt" -ErrorAction SilentlyContinue |
        Where-Object { -not (Test-Path (Join-Path $_.DirectoryName 'done.json')) }).Count
    "$(Get-Date -Format s) stage4 exit $code, unfinished-with-error $failed" | Out-File -Append -Encoding utf8 $log
    if ($code -eq 0 -and $failed -eq 0) { break }
    Start-Sleep 60
}
Pop-Location
"$(Get-Date -Format s) stage 4 finished" | Out-File -Append -Encoding utf8 $log
