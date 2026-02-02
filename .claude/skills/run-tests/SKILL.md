---
name: run-tests
description: テストを実行し失敗があれば修正する
user-invocable: true
allowed-tools: Bash, Read, Edit, Write, Glob, Grep
---

# Run Tests

テストスイートを実行し、失敗があれば原因を調査して修正してください。

## 手順

1. `python -m pytest tests/ -v` を実行
2. 全テストパスなら結果を報告して終了
3. 失敗がある場合:
   - 失敗したテストのエラーメッセージを分析
   - 関連するソースコードを読む
   - 修正案をユーザーに説明してから修正
   - 再度テスト実行で確認
4. HISTORY.md にバグ修正内容を追記
