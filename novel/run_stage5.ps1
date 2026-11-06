# Stage 5 (decided 2026-11-03): final single-model candidates A10 (DINOv3 init + KD + SupCon) and A11 (+ cRW)
# on all 7 tasks. Waits for stage 4 (PID in stage4_pid.txt). Small tasks first, then OCD2 and NDO.
# Restarted after a crash (finished folds are skipped), up to 20 attempts. Log: stage5.log
$b = $PSScriptRoot; $py = $(if ($env:TUSEB_PYTHON) { $env:TUSEB_PYTHON } else { "python" })
$log = "$b\stage5.log"; New-Item -ItemType Directory -Force "$b\logs" | Out-Null
$prev = Get-Content "$b\stage4_pid.txt" -ErrorAction SilentlyContinue
"$(Get-Date -Format s) stage 5 waiting for stage 4 (PID $prev)" | Out-File -Append -Encoding utf8 $log
while ($prev -and (Get-Process -Id $prev -ErrorAction SilentlyContinue)) { Start-Sleep 60 }
"$(Get-Date -Format s) starting stage 5" | Out-File -Append -Encoding utf8 $log
$tasks = 'ours_3class ours_binary OCD3_2class OCD2_2class NDO_2class NDO_3class NDO_4class'
Push-Location $b
for ($i = 1; $i -le 20; $i++) {
    "$(Get-Date -Format s) stage5 attempt $i" | Out-File -Append -Encoding utf8 $log
    cmd /c "`"$py`" -W ignore -u train_novel.py --tasks $tasks --rows A10 A11 >> `"$b\logs\stage5_stdout.log`" 2>> `"$b\logs\stage5_stderr.log`""
    $code = $LASTEXITCODE
    $failed = @(Get-ChildItem "$b\results\*\A1[01]\fold_*\error.txt" -ErrorAction SilentlyContinue |
        Where-Object { -not (Test-Path (Join-Path $_.DirectoryName 'done.json')) }).Count
    "$(Get-Date -Format s) stage5 exit $code, unfinished-with-error $failed" | Out-File -Append -Encoding utf8 $log
    if ($code -eq 0 -and $failed -eq 0) { break }
    Start-Sleep 60
}
Pop-Location
"$(Get-Date -Format s) stage 5 finished" | Out-File -Append -Encoding utf8 $log
