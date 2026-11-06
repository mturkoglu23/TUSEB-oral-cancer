# Stage 3 (decided 2026-11-02): Design A teacher distillation on the small tasks, after the B/C ablations.
# Waits for the stage-2 queue (PID in stage2_pid.txt), then trains A4c (+ DINOv3 ViT-L feature KD), A5 (+ SupCon)
# and A6 (+ cRW = full Design A) with the IN-12k init on ours_3class, ours_binary and OCD3_2class; reference A2b.
# Restarted after a crash (finished folds are skipped), up to 20 attempts. Log: stage3.log
$b = $PSScriptRoot; $py = $(if ($env:TUSEB_PYTHON) { $env:TUSEB_PYTHON } else { "python" })
$log = "$b\stage3.log"; New-Item -ItemType Directory -Force "$b\logs" | Out-Null
$prev = Get-Content "$b\stage2_pid.txt"
"$(Get-Date -Format s) waiting for stage-2 queue PID $prev" | Out-File -Append -Encoding utf8 $log
while (Get-Process -Id $prev -ErrorAction SilentlyContinue) { Start-Sleep 300 }
"$(Get-Date -Format s) starting stage 3" | Out-File -Append -Encoding utf8 $log
# 2026-11-03 (user): the B/C ablations on ours_binary and OCD3 are skipped; stage 2 was stopped after B1-C4 on
# ours_3class, so its remaining ours_3class rows (C5 C6 C7 C9) run here first.
$steps = @(@('ours_3class', 'C5 C6 C7 C9'), @('ours_3class ours_binary OCD3_2class', 'A4c A5 A6'))
Push-Location $b
foreach ($st in $steps) { for ($i = 1; $i -le 20; $i++) {
    "$(Get-Date -Format s) stage3 [$($st[1])] attempt $i" | Out-File -Append -Encoding utf8 $log
    cmd /c "`"$py`" -W ignore -u train_novel.py --tasks $($st[0]) --rows $($st[1]) >> `"$b\logs\stage3_stdout.log`" 2>> `"$b\logs\stage3_stderr.log`""
    $code = $LASTEXITCODE
    $failed = @(Get-ChildItem "$b\results\*\[AC]*\fold_*\error.txt" -ErrorAction SilentlyContinue |
        Where-Object { -not (Test-Path (Join-Path $_.DirectoryName 'done.json')) }).Count
    "$(Get-Date -Format s) stage3 exit $code, unfinished-with-error $failed" | Out-File -Append -Encoding utf8 $log
    if ($code -eq 0 -and $failed -eq 0) { break }
    Start-Sleep 60
} }
Pop-Location
"$(Get-Date -Format s) stage 3 finished" | Out-File -Append -Encoding utf8 $log
