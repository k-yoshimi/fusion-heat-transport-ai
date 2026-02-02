---
name: refine-reference
description: リファレンス解の精度を検証・改善する
user-invocable: true
allowed-tools: Bash, Read, Edit, Write
---

# Refine Reference

リファレンス解の精度を検証し、必要に応じて改善してください。

## 手順

1. `app/run_benchmark.py` の `compute_reference()` を読む
2. 現在の設定を確認（グリッド倍率、時間刻み倍率）
3. 収束性テスト: 倍率を変えて結果がどれだけ変わるか確認
   ```python
   # Pythonスクリプトで4x, 8x, 16xの結果を比較
   ```
4. 結果を報告:
   - 現在のリファレンスの推定精度
   - より高精度にするための推奨設定
   - 計算コストとのトレードオフ
5. ユーザーの承認があれば `compute_reference()` を更新
6. HISTORY.md を更新
