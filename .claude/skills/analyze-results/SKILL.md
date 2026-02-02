---
name: analyze-results
description: ベンチマーク結果を詳細分析する
user-invocable: true
allowed-tools: Bash, Read, Glob
---

# Analyze Results

outputs/ にある最新のベンチマーク結果を詳細に分析してください。

## 手順

1. `outputs/benchmark.csv` を読み取る
2. 以下の観点で分析:
   - **精度比較**: 各ソルバーのL2/L∞誤差をα値ごとに比較
   - **速度比較**: 実行時間の比較
   - **特徴量分析**: max_abs_gradient, energy_content等の物理的意味
   - **非線形性の影響**: αが大きくなると各ソルバーの精度がどう変化するか
   - **トレードオフ**: 精度 vs 速度のパレートフロント
3. 改善提案:
   - グリッド解像度の推奨値
   - 各αに対する最適ソルバー
   - 精度向上のためのパラメータ調整案
4. 結果を日本語でわかりやすくまとめる
