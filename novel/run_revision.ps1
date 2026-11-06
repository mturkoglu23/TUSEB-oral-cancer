# Experiments requested by the internal review (2026-11-03). Starts when the seed queue (run_seeds.ps1, PID in
# seeds_pid.txt) has finished. Finished folds are skipped on restart. Log: revision.log
#  1. NDO patient-grouped (NDO_4class_pg): benchmark ConvNeXtTiny + A2b, A6, and the factorial rows A5, A4c, A4s, A2r
#  2. our data, factorial: A4s (SupCon only), A5x (KD+SupCon without extra NDO images), A7 (A6 without them), A2r (cRW only)
#  3. our data, nested CV: inner 4-fold loop for A1, A2b, A5, A6 inside each outer fold (selection without the test fold)
#  4. NDO_3class_pg, NDO_2class_pg: benchmark ConvNeXtTiny + A2b, A6
#  5. seed robustness of the strong control A2b (seeds 43, 44)
$b = $PSScriptRoot; $cv = Split-Path $b -Parent; $py = $(if ($env:TUSEB_PYTHON) { $env:TUSEB_PYTHON } else { "python" })
$log = "$b\revision.log"; New-Item -ItemType Directory -Force "$b\logs" | Out-Null
$seedPid = [int](Get-Content "$b\seeds_pid.txt" | Select-Object -First 1)
"$(Get-Date -Format s) waiting for seed queue PID $seedPid" | Out-File -Append -Encoding utf8 $log
while (-not (Select-String -Quiet -Path "$b\seeds.log" -Pattern 'seed runs finished') -and
       (Get-Process -Id $seedPid -ErrorAction SilentlyContinue)) { Start-Sleep 120 }
"$(Get-Date -Format s) starting revision runs" | Out-File -Append -Encoding utf8 $log

function Run($dir, $argline, $tag) {
    for ($i = 1; $i -le 5; $i++) {
        "$(Get-Date -Format s) $tag attempt $i" | Out-File -Append -Encoding utf8 $log
        Push-Location $dir
        cmd /c "`"$py`" -W ignore -u $argline >> `"$b\logs\revision_stdout.log`" 2>> `"$b\logs\revision_stderr.log`""
        $code = $LASTEXITCODE
        Pop-Location
        "$(Get-Date -Format s) $tag exit $code" | Out-File -Append -Encoding utf8 $log
        if ($code -eq 0) { return }
        Start-Sleep 60
    }
}

Run "$cv\public" "02_train_public.py --datasets NDO_4class_pg --models ConvNeXtTiny" "pg4 bench"
Run "$b" "train_novel.py --tasks NDO_4class_pg --rows A2b A6 A5 A4c A4s A2r" "pg4 novel"
Run "$b" "train_novel.py --tasks ours_3class --rows A4s A5x A7 A2r" "ours factorial"
foreach ($o in 0..4) { Run "$b" "train_novel.py --tasks ours_3class --rows A1 A2b A5 A6 --nested $o" "nested outer $o" }
Run "$cv\public" "02_train_public.py --datasets NDO_3class_pg NDO_2class_pg --models ConvNeXtTiny" "pg3/pg2 bench"
Run "$b" "train_novel.py --tasks NDO_3class_pg NDO_2class_pg --rows A2b A6" "pg3/pg2 novel"
foreach ($s in 43, 44) {
    Run "$b" "train_novel.py --tasks ours_3class OCD3_2class OCD2_2class NDO_2class NDO_3class NDO_4class --rows A2b --seed $s" "A2b seed $s"
}
"$(Get-Date -Format s) revision runs finished" | Out-File -Append -Encoding utf8 $log
