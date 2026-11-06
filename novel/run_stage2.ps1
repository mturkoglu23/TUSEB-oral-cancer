# Stage 2 of the novel designs (decided 2026-11-02: Designs B and C before the rest of Design A).
# Waits for the stage-1 queue (PID in stage1_pid.txt), then trains on our two tasks and OCD3:
#   2a: full designs first (B4 = full ZoomNeXt, C8 = full ProtoFormer), 2b: their ablation rows.
# B0 = C0 = A2b (IN-12k init, A1 recipe), already trained in stage 1.
# Restarted after a crash (finished folds are skipped), up to 20 attempts. Log: stage2.log
# -NoWait: run alongside stage 1 instead of after it (the small tasks fit next to the NDO runs on the GPU).
param([switch]$NoWait)
$b = $PSScriptRoot; $py = $(if ($env:TUSEB_PYTHON) { $env:TUSEB_PYTHON } else { "python" })
$log = "$b\stage2.log"; New-Item -ItemType Directory -Force "$b\logs" | Out-Null
$prev = Get-Content "$b\stage1_pid.txt"
if (-not $NoWait) {
    "$(Get-Date -Format s) waiting for stage-1 queue PID $prev" | Out-File -Append -Encoding utf8 $log
    while (Get-Process -Id $prev -ErrorAction SilentlyContinue) { Start-Sleep 300 }
}
"$(Get-Date -Format s) starting stage 2 (NoWait=$NoWait)" | Out-File -Append -Encoding utf8 $log
$tasks = 'ours_3class ours_binary OCD3_2class'
$steps = [ordered]@{ '2a' = 'B4 C8'; '2b' = 'B1 B2 B2s B3 B5 B6 B7 C1 C2 C3 C4 C5 C6 C7 C9' }
Push-Location $b
foreach ($s in $steps.Keys) {
    for ($i = 1; $i -le 20; $i++) {
        "$(Get-Date -Format s) stage$s attempt $i" | Out-File -Append -Encoding utf8 $log
        cmd /c "`"$py`" -W ignore -u train_novel.py --tasks $tasks --rows $($steps[$s]) >> `"$b\logs\stage2_stdout.log`" 2>> `"$b\logs\stage2_stderr.log`""
        $code = $LASTEXITCODE
        $failed = @(Get-ChildItem "$b\results\*\[BC]*\fold_*\error.txt" -ErrorAction SilentlyContinue |
            Where-Object { -not (Test-Path (Join-Path $_.DirectoryName 'done.json')) }).Count
        "$(Get-Date -Format s) stage$s exit $code, unfinished-with-error $failed" | Out-File -Append -Encoding utf8 $log
        if ($code -eq 0 -and $failed -eq 0) { break }
        Start-Sleep 60
    }
}
Pop-Location
"$(Get-Date -Format s) stage 2 finished" | Out-File -Append -Encoding utf8 $log
