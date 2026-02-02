---
name: run-benchmark
description: ベンチマークを実行し結果を報告する
user-invocable: true
allowed-tools: Bash, Read
---

# Run Benchmark

ベンチマークスイートを実行して結果を報告してください。

## 手順

1. まず `python -m pytest tests/ -v` でテストが通ることを確認
2. ユーザーが引数を指定した場合はそれを使い、指定がなければデフォルトで実行:
   ```
   python -m app.run_benchmark --alpha 0.0 0.5 1.0
   ```
3. `outputs/benchmark.csv` と `outputs/benchmark.md` を読み取る
4. 結果を以下の観点でユーザーに報告:
   - 各ソルバーのL2/L∞誤差
   - 実行時間
   - 最良ソルバーとその理由
   - 前回との差異（もしあれば）
