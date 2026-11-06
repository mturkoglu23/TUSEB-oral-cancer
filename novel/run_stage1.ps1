# Stage 1 of the novel designs: waits for the benchmark queue (PID in ../supervisor_pid.txt) to finish, caches the
# DINOv3 ViT-L teacher embeddings for later Design-A rows, then trains rows A1 A2a A2b A2c on all 7 tasks.
# Restarted after a crash (finished folds are skipped), up to 20 attempts. Log: stage1.log
$b = $PSScriptRoot; $py = $(if ($env:TUSEB_PYTHON) { $env:TUSEB_PYTHON } else { "python" })
$log = "$b\stage1.log"; New-Item -ItemType Directory -Force "$b\logs" | Out-Null
$bench = Get-Content "$b\..\supervisor_pid.txt"
"$(Get-Date -Format s) waiting for benchmark queue PID $bench" | Out-File -Append -Encoding utf8 $log
while (Get-Process -Id $bench -ErrorAction SilentlyContinue) { Start-Sleep 300 }
"$(Get-Date -Format s) benchmark finished; starting stage 1" | Out-File -Append -Encoding utf8 $log
$tasks = 'ours_3class ours_binary OCD3_2class OCD2_2class NDO_2class NDO_3class NDO_4class'
Push-Location $b
cmd /c "`"$py`" -W ignore -u cache_teacher.py --teacher dinov3_vitl --tasks $tasks >> `"$b\logs\stage1_stdout.log`" 2>> `"$b\logs\stage1_stderr.log`""
for ($i = 1; $i -le 20; $i++) {
    "$(Get-Date -Format s) stage1 attempt $i" | Out-File -Append -Encoding utf8 $log
    cmd /c "`"$py`" -W ignore -u train_novel.py --tasks $tasks --rows A1 A2a A2b A2c >> `"$b\logs\stage1_stdout.log`" 2>> `"$b\logs\stage1_stderr.log`""
    $code = $LASTEXITCODE
    $failed = @(Get-ChildItem "$b\results\*\A*\fold_*\error.txt" -ErrorAction SilentlyContinue |
        Where-Object { -not (Test-Path (Join-Path $_.DirectoryName 'done.json')) }).Count
    "$(Get-Date -Format s) stage1 exit $code, unfinished-with-error $failed" | Out-File -Append -Encoding utf8 $log
    if ($code -eq 0 -and $failed -eq 0) { break }
    Start-Sleep 60
}
Pop-Location
"$(Get-Date -Format s) stage 1 finished" | Out-File -Append -Encoding utf8 $log
