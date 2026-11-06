# Seed robustness (decided 2026-11-03): the proposed model (A6) and the benchmark ConvNeXtTiny re-trained with
# seeds 43 and 44 on the same folds (seed 42 = the original runs). Outputs: novel/results/<task>/A6_s<seed>/,
# results/seeds/s<seed>/ (+ binary/), public/results_public/<dataset>/seeds/s<seed>/. Finished folds are skipped on restart.
# Log: seeds.log
$b = $PSScriptRoot; $cv = Split-Path $b -Parent; $py = $(if ($env:TUSEB_PYTHON) { $env:TUSEB_PYTHON } else { "python" })
$log = "$b\seeds.log"; New-Item -ItemType Directory -Force "$b\logs" | Out-Null
"$(Get-Date -Format s) starting seed runs" | Out-File -Append -Encoding utf8 $log

function Run($dir, $argline, $tag) {
    for ($i = 1; $i -le 10; $i++) {
        "$(Get-Date -Format s) $tag attempt $i" | Out-File -Append -Encoding utf8 $log
        Push-Location $dir
        cmd /c "`"$py`" -W ignore -u $argline >> `"$b\logs\seeds_stdout.log`" 2>> `"$b\logs\seeds_stderr.log`""
        $code = $LASTEXITCODE
        Pop-Location
        "$(Get-Date -Format s) $tag exit $code" | Out-File -Append -Encoding utf8 $log
        if ($code -eq 0) { return }
        Start-Sleep 60
    }
}

foreach ($s in 43, 44) {
    Run $cv "02_train_cv.py --models ConvNeXtTiny --seed $s" "s$s bench ours 3class"
    Run $cv "02_train_cv.py --models ConvNeXtTiny --task binary --seed $s" "s$s bench ours binary"
    Run "$b" "train_novel.py --tasks ours_3class OCD3_2class OCD2_2class --rows A6 --seed $s" "s$s A6 small"
    Run "$cv\public" "02_train_public.py --models ConvNeXtTiny --seed $s" "s$s bench public"
    Run "$b" "train_novel.py --tasks NDO_2class NDO_3class NDO_4class --rows A6 --seed $s" "s$s A6 NDO"
}
"$(Get-Date -Format s) seed runs finished" | Out-File -Append -Encoding utf8 $log
