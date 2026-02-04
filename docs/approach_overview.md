# Fusion Heat Transport PDE Benchmark: アプローチ概要

## 1. プロジェクト概要

本プロジェクトは、核融合プラズマにおける1次元径方向熱輸送方程式を対象に、複数の数値解法（FDM, FEM, FVM, Spectral, PINN）を比較・評価するベンチマークフレームワークです。

### 対象方程式

```
∂T/∂t = (1/r) ∂/∂r (r χ ∂T/∂r)
```

ここで、非線形拡散係数 χ(|T'|) は：
- |T'| > 0.5 のとき: χ = (|T' - 0.5|)^α + 0.1
- |T'| ≤ 0.5 のとき: χ = 0.1

α パラメータが非線形性の強さを制御します。

### r=0 特異点の処理

中心 r=0 でロピタルの定理を適用：
```
(1/r)∂r(rχ∂T/∂r) → 2χ∂²T/∂r²  (r→0)
```

## 2. 3フェーズ構成

### Phase 1: 基本ベンチマーク（完了）

**目的**: 各ソルバーの基本性能評価

**成果物**:
- 8種類のソルバー実装
- 精度・速度の基本比較
- 5種類の初期条件プロファイル

### Phase 2: 物理ベース選択と最適化（完了）

**目的**: 問題特性に基づくソルバー自動選択

**成果物**:
- `policy/physics_selector.py` - 物理特徴量ベースの選択器
- `policy/optimizer.py` - パラメータ最適化（dt, nr）
- `policy/stability.py` - 安定性制約の管理

**着眼点**:
1. **物理特徴量の抽出**: max_chi, chi_ratio, problem_stiffness
2. **安定性制約の明示化**: CFL条件、α上限
3. **多目的最適化**: 精度と速度のトレードオフ

### Phase 3: マルチエージェント改善サイクル（今回実装）

**目的**: 自動化された手法改善ループ

**成果物**:
- `docs/analysis/pareto_analyzer.py` - パレート解析
- `docs/analysis/improvement_agents.py` - 改善エージェント群
- `docs/analysis/method_improvement_cycle.py` - サイクル管理

## 3. マルチエージェント改善サイクルの詳細

### 3.1 ワークフロー（7フェーズ）

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Pareto Analysis                                    │
│  └─ 各ソルバーで(alpha, nr, dt)スイープ → パレートフロント   │
│                     ↓                                        │
│  Phase 2: Bottleneck Analysis                                │
│  └─ 安定性、精度、速度、カバレッジのギャップ特定            │
│                     ↓                                        │
│  Phase 3: Proposal Generation                                │
│  └─ parameter_tuning / algorithm_tweak 提案生成             │
│                     ↓                                        │
│  Phase 4: Multi-Agent Evaluation                             │
│  └─ 4観点（精度・速度・安定性・複雑度）からの評価           │
│                     ↓                                        │
│  Phase 5: Human Review (Optional)                            │
│  └─ インタラクティブな承認/却下                             │
│                     ↓                                        │
│  Phase 6: Implementation                                     │
│  └─ 承認提案の適用                                          │
│                     ↓                                        │
│  Phase 7: Report & Archive                                   │
│  └─ レポート生成、状態保存                                  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 エージェント構成

| Agent | 役割 | 着眼点 |
|-------|------|--------|
| **ParetoAnalysisAgent** | パレートフロント計算 | error vs time の2目的最適化 |
| **BottleneckAnalysisAgent** | ボトルネック特定 | 4カテゴリ（stability, accuracy_gap, speed_gap, coverage_gap） |
| **ProposalGenerationAgent** | 改善提案生成 | ボトルネックに対応した具体策 |
| **EvaluationAgent** | 多観点評価 | 重み付き合議（精度1.5、安定性1.2、速度1.0、複雑度0.8） |
| **ReportAgent** | レポート生成 | 構造化マークダウン |

### 3.3 パレート解析の着眼点

**パレート最適性判定**:
```python
def _is_pareto_dominated(candidate, others):
    """点 A が点 B に支配されるのは、B が全目的で A 以上かつ
    少なくとも1目的で A より良い場合"""
    for other in others:
        if other[0] <= candidate[0] and other[1] <= candidate[1]:
            if other[0] < candidate[0] or other[1] < candidate[1]:
                return True
    return False
```

**評価指標**:
- L2誤差（円筒座標重み付き）
- 計算時間（wall time）
- 安定性（NaN、発散の有無）

### 3.4 ボトルネック検出の着眼点

1. **安定性問題** (severity: high/medium)
   - 条件: stability_rate < 90%
   - 対策: adaptive dt, パラメータ制約

2. **精度ギャップ** (severity: medium)
   - 条件: worst_error > best_error × 100
   - 対策: 解像度向上、高次スキーム

3. **速度ギャップ** (severity: low)
   - 条件: slowest_time > fastest_time × 100
   - 対策: 実装最適化、並列化

4. **カバレッジギャップ** (severity: low)
   - 条件: 単一ソルバーが80%以上を支配
   - 対策: ニッチ探索、パラメータチューニング

### 3.5 提案タイプ

| Type | 説明 | 自動実装 |
|------|------|----------|
| `parameter_tuning` | パラメータ調整 | 可能 |
| `algorithm_tweak` | アルゴリズム改良 | スケッチ提供 |
| `new_solver` | 新規ソルバー | スケッチ提供 |

## 4. ソルバー追加の流れ

### 4.1 手順

1. `solvers/` 配下にソルバークラス作成
2. `SolverBase` を継承、`solve()` メソッド実装
3. `app/run_benchmark.py` の `SOLVERS` リストに追加
4. `policy/stability.py` に安定性情報追加
5. テスト追加・実行

### 4.2 実装チェックリスト

- [ ] r=0 特異点処理（L'Hôpital）
- [ ] 境界条件: T'(0)=0, T(1)=0
- [ ] 非線形χの計算
- [ ] 出力形状: (nt+1, nr)
- [ ] scipy banded solver使用（高速化）

## 5. 高速化の着眼点

### 5.1 一般的な高速化手法

| 手法 | 適用場面 | 期待効果 |
|------|----------|----------|
| **banded solver** | 三重対角行列 | 10-100倍 |
| **事前計算** | 定数行列 | 2-5倍 |
| **Numba JIT** | 内部ループ | 10-50倍 |
| **ベクトル化** | numpy操作 | 5-20倍 |

### 5.2 ソルバー別の高速化ポイント

**FDM系**:
- `scipy.linalg.solve_banded()` で三重対角ソルブ
- 係数行列の再利用

**Spectral系**:
- FFT/DCTの効率的使用
- 変換行列の事前計算

**FEM系**:
- 疎行列フォーマット（CSR）
- アセンブリの最適化

**PINN系**:
- バッチサイズ調整
- GPU活用（PyTorch）

## 6. スラッシュコマンド一覧

Claude Code で使用可能なスラッシュコマンド（スキル）:

| コマンド | 説明 | 使用例 |
|----------|------|--------|
| `/add-solver` | 新規ソルバーを追加 | "FEM P3要素を追加" |
| `/speedup-solver` | ソルバーの高速化分析と最適化 | "implicit_fdmを高速化" |
| `/improvement-cycle` | マルチエージェント改善サイクル実行 | "3サイクル実行" |
| `/pareto-analysis` | パレートフロント解析 | "全ソルバーを解析" |
| `/run-benchmark` | ベンチマーク実行 | "alpha=1.5でテスト" |
| `/run-tests` | テスト実行と修正 | "ソルバーテスト実行" |
| `/analyze-results` | 結果の詳細分析 | "最新結果を分析" |
| `/refine-reference` | 参照解の精度検証 | "精度を確認" |

## 7. CLI コマンド一覧

### 基本ベンチマーク
```bash
make benchmark                    # 標準ベンチマーク実行
python -m app.run_benchmark --alpha 1.5 --nr 101 --dt 0.0005
```

### ML選択器
```bash
python -m app.run_benchmark --use-ml-selector --alpha 1.5
python -m app.run_benchmark --physics-selector --optimize-params
```

### 改善サイクル
```bash
python docs/analysis/method_improvement_cycle.py --cycles 3
python docs/analysis/method_improvement_cycle.py --resume
python docs/analysis/method_improvement_cycle.py --interactive
python docs/analysis/method_improvement_cycle.py --report
```

### パレート解析
```bash
python docs/analysis/pareto_analyzer.py                    # 全ソルバー
python docs/analysis/method_improvement_cycle.py --phase pareto  # フェーズ単独
```

### テスト
```bash
make test                         # 全テスト
python -m pytest tests/ -k "solver" -v  # ソルバーテストのみ
```

## 8. ディレクトリ構造

```
fusion-heat-transport-ai/
├── app/
│   └── run_benchmark.py          # CLIエントリーポイント
├── solvers/
│   ├── base.py                   # SolverBase ABC
│   ├── fdm/                      # 有限差分法
│   │   ├── implicit.py           # 陰解法FDM
│   │   ├── compact4.py           # 4次コンパクト差分
│   │   └── imex.py               # IMEX法
│   ├── fem/
│   │   └── p2_fem.py             # P2有限要素法
│   ├── fvm/
│   │   └── cell_centered.py      # セル中心FVM
│   ├── spectral/
│   │   ├── cosine.py             # コサイン変換
│   │   └── chebyshev.py          # チェビシェフ
│   └── pinn/
│       └── stub.py               # PINNスタブ
├── policy/
│   ├── select.py                 # ルールベース選択
│   ├── physics_selector.py       # 物理ベース選択
│   ├── optimizer.py              # パラメータ最適化
│   ├── stability.py              # 安定性制約
│   └── train.py                  # ML学習
├── docs/analysis/
│   ├── pareto_analyzer.py        # パレート解析
│   ├── improvement_agents.py     # 改善エージェント
│   └── method_improvement_cycle.py # サイクル管理
├── data/
│   ├── pareto_fronts/            # パレートフロントJSON
│   ├── cycle_reports/            # サイクルレポート
│   └── improvement_history.json  # 履歴
└── tests/
    └── test_*.py                 # テストファイル
```

## 9. 今後の拡張方向

1. **新ソルバー追加候補**
   - 適応格子法 (AMR)
   - マルチグリッド法
   - 高次WENO/ENO

2. **高速化候補**
   - GPU並列化（CuPy/PyTorch）
   - MPI分散計算
   - JIT最適化（Numba）

3. **解析拡張候補**
   - 2D/3D拡張
   - 時間変動境界条件
   - マルチスケール解析
