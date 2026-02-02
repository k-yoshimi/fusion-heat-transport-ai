---
name: add-solver
description: 新しいソルバーをプロジェクトに追加する
user-invocable: true
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Add Solver

新しい数値ソルバーをプロジェクトに追加してください。

## 手順

1. ユーザーにソルバーの種類・手法を確認（例: 陽的FDM, MOL, FEM等）
2. `solvers/base.py` の `SolverBase` を読み、インターフェースを確認
3. 適切なサブディレクトリを作成（例: `solvers/fem/`）
4. `SolverBase` を継承した新ソルバーを実装:
   - `name` クラス属性を設定
   - `solve(T0, r, dt, t_end, alpha) -> T_history` を実装
   - r=0 の特異点を L'Hôpital で処理
   - Neumann BC (r=0), Dirichlet BC (r=1) を適用
5. `tests/test_solvers.py` にテストを追加:
   - 基本動作テスト（温度が拡散で下がるか）
   - 境界条件テスト
   - 非線形(alpha>0)テスト
6. `app/run_benchmark.py` の `run()` にソルバーを追加
7. `python -m pytest tests/ -v` で全テストパスを確認
8. HISTORY.md を更新
