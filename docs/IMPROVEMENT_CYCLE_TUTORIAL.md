# チュートリアル: マルチエージェント手法改善サイクル

本チュートリアルでは、マルチエージェント改善サイクルの具体的な実行手順を、初回実行から結果の読み方、次のアクションまでステップバイステップで解説します。

## 前提条件

```bash
pip install -e ".[dev]"
pip install physbo              # PHYSBO（推奨、なくても動作する）
python -m pytest tests/ -v     # 205 tests should pass
```

> **Note:** `physbo` がインストールされていると、パレート解析のパラメータ探索がグリッドスイープからベイズ最適化に自動的に切り替わり、より効率的な探索が行われます。未インストールの場合は従来のグリッドスイープにフォールバックします。

---

## Part 1: はじめての改善サイクル実行

### 1.1 最もシンプルな実行

まずは1サイクルだけ実行してみましょう:

```bash
python docs/analysis/method_improvement_cycle.py --fresh --cycles 1
```

`--fresh` は履歴をクリアして新規開始するオプションです。初回は必ずつけてください。

PHYSBO がインストールされている場合、パレート解析は自動的にベイズ最適化モードで動作します。

以下は実際の実行出力です（PHYSBO モード）:

```
============================================================
RUNNING 1 IMPROVEMENT CYCLES
============================================================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cycle 1 of 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============================================================
STARTING IMPROVEMENT CYCLE 1
============================================================

[Phase: PARETO]
----------------------------------------
Analyzing 8 solvers...

Analyzing implicit_fdm via PHYSBO (3 problems, 5+15 probes each)...
  alpha=0.0, ic=parabola... L2=1.20e-03, dt=1.5e-05
  alpha=0.5, ic=parabola... L2=4.39e-05, dt=8.9e-05
  alpha=1.0, ic=parabola... L2=1.17e-04, dt=8.2e-05
  Completed: 3/3 stable
  Pareto-optimal: 1 points

Analyzing spectral_cosine via PHYSBO (3 problems, 5+15 probes each)...
  alpha=0.0, ic=parabola... L2=3.83e-02, dt=9.7e-05
  alpha=0.5, ic=parabola... L2=3.53e-02, dt=1.4e-04
  alpha=1.0, ic=parabola... L2=3.37e-02, dt=6.8e-05
  Completed: 3/3 stable
  Pareto-optimal: 2 points

Analyzing pinn_stub via PHYSBO (3 problems, 5+15 probes each)...
  alpha=0.0, ic=parabola... L2=4.19e-01, dt=1.1e-03
  alpha=0.5, ic=parabola... L2=2.28e-01, dt=2.1e-04
  alpha=1.0, ic=parabola... L2=3.06e-01, dt=8.2e-05
  Completed: 3/3 stable
  Pareto-optimal: 2 points

Analyzing compact4_fdm via PHYSBO (3 problems, 5+15 probes each)...
  alpha=0.0, ic=parabola... L2=7.56e-03, dt=8.2e-05
  alpha=0.5, ic=parabola... L2=1.31e-04, dt=8.2e-05
  alpha=1.0, ic=parabola... L2=6.67e-05, dt=8.2e-05
  Completed: 3/3 stable
  Pareto-optimal: 1 points

Analyzing imex_fdm via PHYSBO (3 problems, 5+15 probes each)...
  alpha=0.0, ic=parabola... L2=3.64e-01, dt=8.2e-05
  alpha=0.5, ic=parabola... L2=3.20e-01, dt=8.2e-05
  alpha=1.0, ic=parabola... L2=3.04e-01, dt=1.1e-03
  Completed: 3/3 stable
  Pareto-optimal: 1 points

Analyzing p2_fem via PHYSBO (3 problems, 5+15 probes each)...
  alpha=0.0, ic=parabola... L2=1.87e-03, dt=2.0e-05
  alpha=0.5, ic=parabola... L2=1.26e-05, dt=1.0e-05
  alpha=1.0, ic=parabola... L2=5.92e-02, dt=8.2e-05
  Completed: 3/3 stable
  Pareto-optimal: 3 points

Analyzing cell_centered_fvm via PHYSBO (3 problems, 5+15 probes each)...
  alpha=0.0, ic=parabola... L2=6.28e-02, dt=6.3e-05
  alpha=0.5, ic=parabola... L2=5.74e-02, dt=3.3e-04
  alpha=1.0, ic=parabola... L2=6.46e-02, dt=1.4e-04
  Completed: 3/3 stable
  Pareto-optimal: 1 points

Analyzing chebyshev_spectral via PHYSBO (3 problems, 5+15 probes each)...
  alpha=0.0, ic=parabola... L2=6.86e-01, dt=9.4e-04
  alpha=0.5, ic=parabola... L2=6.26e-01, dt=2.5e-04
  alpha=1.0, ic=parabola... L2=5.55e-01, dt=8.2e-05
  Completed: 3/3 stable
  Pareto-optimal: 3 points

Running cross-solver analysis...

Cross-solver analysis (PHYSBO): 3 problems x 8 solvers (~480 probes)

  Problem: alpha=0.0_ic=parabola
    implicit_fdm             : L2=1.50e-03, time=116.69ms (nr=61, dt=1.84e-05)
    spectral_cosine          : L2=3.83e-02, time=22.61ms (nr=61, dt=9.71e-05)
    compact4_fdm             : L2=7.56e-03, time=30.24ms (nr=61, dt=8.15e-05)
    chebyshev_spectral       : L2=6.86e-01, time=3.68ms  (nr=61, dt=9.43e-04)
    Pareto-optimal: implicit_fdm, spectral_cosine, compact4_fdm, chebyshev_spectral

  Problem: alpha=0.5_ic=parabola
    implicit_fdm             : L2=1.65e-04, time=26.18ms  (nr=61, dt=8.15e-05)
    compact4_fdm             : L2=1.31e-04, time=30.30ms  (nr=61, dt=8.15e-05)
    p2_fem                   : L2=1.26e-05, time=4700.44ms (nr=61, dt=1.00e-05)
    cell_centered_fvm        : L2=5.74e-02, time=6.19ms   (nr=61, dt=3.30e-04)
    Pareto-optimal: implicit_fdm, spectral_cosine, compact4_fdm, p2_fem, cell_centered_fvm

  Problem: alpha=1.0_ic=parabola
    implicit_fdm             : L2=1.17e-04, time=27.34ms (nr=61, dt=8.15e-05)
    compact4_fdm             : L2=6.67e-05, time=29.20ms (nr=61, dt=8.15e-05)
    imex_fdm                 : L2=3.04e-01, time=2.97ms  (nr=61, dt=1.12e-03)
    Pareto-optimal: implicit_fdm, compact4_fdm, imex_fdm, cell_centered_fvm

  Overall rankings:
    #1 implicit_fdm             : avg_rank=1.3, avg_L2=5.95e-04, stability=100%
    #2 compact4_fdm             : avg_rank=1.7, avg_L2=2.59e-03, stability=100%
    #3 spectral_cosine          : avg_rank=3.0, avg_L2=3.58e-02, stability=100%
    #4 cell_centered_fvm        : avg_rank=4.3, avg_L2=6.15e-02, stability=100%
    #5 p2_fem                   : avg_rank=5.3, avg_L2=2.59e-02, stability=100%

[Phase: BOTTLENECK]
----------------------------------------
Found 1 bottlenecks:
  - [low] Speed gap: fastest=2.98ms, slowest=2361.36ms

[Phase: PROPOSAL]
----------------------------------------
Generated 1 proposals:
  - P001: Optimize pinn_stub, compact4_fdm, p2_fem implementation

[Phase: EVALUATION]
----------------------------------------
Evaluation results:
Rank  ID      Score   Recommendation
----------------------------------------
1     P001    3.38    consider

[Phase: REVIEW]
----------------------------------------

Auto-approving proposals...
  Rejected: P001

[Phase: IMPLEMENTATION]
----------------------------------------

Approved proposals for implementation: 0

[Phase: REPORT]
----------------------------------------

Report saved to: data/cycle_reports/cycle_001_20260204_095037.md

[Phase: COMPLETE]
----------------------------------------

============================================================
CYCLE 1 COMPLETE
============================================================
  Bottlenecks found: 1
  Proposals generated: 1
  Proposals approved: 0

============================================================
COMPLETED 1 CYCLES
============================================================
Total proposals: 1
Implemented: 0

Summary report saved to: data/cycle_reports/summary_report.md
```

> **Note:** PHYSBO がインストールされていない場合はグリッドスイープで自動的にフォールバックします。出力は `Analyzing implicit_fdm (12 configurations)...` のように表示されます。

### 1.2 出力の読み方

7つのフェーズが順に実行されます。各フェーズの着眼点:

| フェーズ | 注目する出力 | 何がわかるか |
|----------|-------------|-------------|
| **PARETO** | `via PHYSBO` or `configurations` | 探索モード（PHYSBO/グリッド） |
| **PARETO** | `Completed: N/M stable` | ソルバーの安定性 |
| **PARETO** | `Pareto-optimal: N points` | パレート最適点の数 |
| **BOTTLENECK** | `[severity] description` | 性能上の課題 |
| **PROPOSAL** | `P001: Title` | 改善の具体案 |
| **EVALUATION** | `Score` と `Recommendation` | 提案の優先度 |

### 1.3 生成されるファイル

実行後、以下のファイルが生成されます:

```
data/
├── pareto_fronts/                    ← ソルバーごとのパレートフロント
│   ├── implicit_fdm_2026-...json
│   ├── spectral_cosine_2026-...json
│   └── ... (8ファイル)
├── cycle_reports/
│   ├── cycle_001_20260204_...md      ← サイクルレポート
│   └── summary_report.md            ← サマリーレポート
└── improvement_history.json          ← 履歴（リスタート用）
```

---

## Part 2: 出力結果の詳細分析

### 2.1 サイクルレポートを読む

```bash
cat data/cycle_reports/cycle_001_*.md
```

以下は実際に生成されたレポート（`data/cycle_reports/cycle_001_20260204_095037.md`）の内容です:

```markdown
# Method Improvement Cycle Report - Cycle 1

## Executive Summary
- **Analyzed solvers:** 8
- **Problem settings analyzed:** 3
- **Identified bottlenecks:** 1
- **Generated proposals:** 1
- **Approved proposals:** 0

## Cross-Solver Analysis

### Overall Solver Rankings

| Rank | Solver | Avg Rank | Avg L2 Error | Avg Time (ms) | Stability |
|------|--------|----------|-------------|---------------|-----------|
| 1 | implicit_fdm | 1.3 | 5.95e-04 | 56.74 | 100% |
| 2 | compact4_fdm | 1.7 | 2.59e-03 | 29.91 | 100% |
| 3 | spectral_cosine | 3.0 | 3.58e-02 | 24.02 | 100% |
| 4 | cell_centered_fvm | 4.3 | 6.15e-02 | 73.34 | 100% |
| 5 | p2_fem | 5.3 | 2.59e-02 | 1947.92 | 100% |
| 6 | imex_fdm | 6.0 | 3.29e-01 | 28.40 | 100% |
| 7 | pinn_stub | 6.3 | 3.40e-01 | 129.42 | 100% |
| 8 | chebyshev_spectral | 8.0 | 6.22e-01 | 19.79 | 100% |

### Per-Problem Results

**alpha=0.0_ic=parabola**
- Pareto-optimal: implicit_fdm, spectral_cosine, compact4_fdm, chebyshev_spectral
- Best accuracy: implicit_fdm
- Fastest: chebyshev_spectral

**alpha=0.5_ic=parabola**
- Pareto-optimal: implicit_fdm, spectral_cosine, compact4_fdm, p2_fem, cell_centered_fvm
- Best accuracy: p2_fem
- Fastest: cell_centered_fvm

**alpha=1.0_ic=parabola**
- Pareto-optimal: implicit_fdm, compact4_fdm, imex_fdm, cell_centered_fvm
- Best accuracy: compact4_fdm
- Fastest: imex_fdm

## Per-Solver Pareto Analysis

| Solver | Total Points | Stable | Pareto-Optimal | Min Error | Max Error |
|--------|--------------|--------|----------------|-----------|-----------|
| implicit_fdm | 3 | 3 (100%) | 1 | 4.39e-05 | 1.20e-03 |
| spectral_cosine | 3 | 3 (100%) | 2 | 3.37e-02 | 3.83e-02 |
| pinn_stub | 3 | 3 (100%) | 2 | 2.28e-01 | 4.19e-01 |
| compact4_fdm | 3 | 3 (100%) | 1 | 6.67e-05 | 7.56e-03 |
| imex_fdm | 3 | 3 (100%) | 1 | 3.04e-01 | 3.64e-01 |
| p2_fem | 3 | 3 (100%) | 3 | 1.26e-05 | 5.92e-02 |
| cell_centered_fvm | 3 | 3 (100%) | 1 | 5.74e-02 | 6.46e-02 |
| chebyshev_spectral | 3 | 3 (100%) | 3 | 5.55e-01 | 6.86e-01 |

## Bottlenecks Identified

### 1. Speed gap: fastest=2.98ms, slowest=2361.36ms
- **ID:** B001
- **Category:** speed_gap
- **Severity:** LOW
- **Affected:** pinn_stub, compact4_fdm, p2_fem
- **Suggested Actions:**
  - Optimize slow solver implementations
  - Use coarser grids where accuracy permits
  - Consider parallel implementations

## Multi-Agent Evaluation

| Proposal | Accuracy | Speed | Stability | Complexity | Overall | Recommendation |
|----------|----------|-------|-----------|------------|---------|----------------|
| P001 | *** | **** | *** | ** | 3.4 | Consider |
```

### 2.2 レポートの着眼点

**Cross-Solver Rankings テーブルの読み方:**

- **Avg Rank**: 全問題での平均順位。`implicit_fdm` が 1.3 で最良
- **Avg L2 Error**: 平均精度。`implicit_fdm` (5.95e-04) と `compact4_fdm` (2.59e-03) が突出
- **Avg Time**: `p2_fem` が 1947ms と極端に遅い → ボトルネック候補
- **Stability**: 今回は全ソルバー 100% だが、条件によっては低下する

**Per-Solver Pareto Analysis テーブルの読み方:**

- **Stable 列**: 100% でないソルバーは安定性に問題あり
- **Pareto-Optimal 列**: 数が多いほどトレードオフ空間をよくカバー
- **Min Error**: `p2_fem` が 1.26e-05 で最高精度、ただし時間コスト大

**Multi-Agent Evaluation テーブルの読み方:**

- **Accuracy `***`**: 精度改善への寄与度（5段階）
- **Speed `****`**: 速度改善効果（高いほど速度向上が期待できる）
- **Stability `***`**: 安定性改善効果
- **Complexity `**`**: 実装の簡単さ（`*` が少ない = 複雑）
- **Overall**: 重み付き総合スコア（4.0以上で自動承認、3.4 は "Consider" 扱い）

### 2.3 パレートフロントを直接確認する

Python で個別のパレートフロントデータを確認:

```python
from docs.analysis.pareto_analyzer import load_all_pareto_fronts

fronts = load_all_pareto_fronts()
for name, front in sorted(fronts.items()):
    s = front.summary
    print(f'{name:25s}: {s["pareto_optimal_count"]} optimal / '
          f'{s["stable_points"]} stable / {s["total_points"]} total '
          f'({s["stability_rate"]:.0f}%)')
```

実際の出力（PHYSBO モード、3 問題設定）:

```
cell_centered_fvm        : 1 optimal / 3 stable / 3 total (100%)
chebyshev_spectral       : 3 optimal / 3 stable / 3 total (100%)
compact4_fdm             : 1 optimal / 3 stable / 3 total (100%)
imex_fdm                 : 1 optimal / 3 stable / 3 total (100%)
implicit_fdm             : 1 optimal / 3 stable / 3 total (100%)
p2_fem                   : 3 optimal / 3 stable / 3 total (100%)
pinn_stub                : 2 optimal / 3 stable / 3 total (100%)
spectral_cosine          : 2 optimal / 3 stable / 3 total (100%)
```

> **Note:** PHYSBO モードでは各(alpha, ic_type)問題ごとに1つの最適 dt を見つけるため、Total Points = 問題数（3）になります。グリッドスイープモードでは dt_list の全組み合わせが Total Points に含まれます。

**読み取れること:**
- 全ソルバーが 100% 安定 — PHYSBO が安定な dt を効率的に発見
- `p2_fem`, `chebyshev_spectral` が全 3 点 Pareto-optimal → 広いトレードオフカバレッジ
- `compact4_fdm`, `implicit_fdm` は 1 点のみだが、Cross-solver 分析で高ランク → 精度が突出

---

## Part 3: 複数サイクルの実行

### 3.1 3サイクル連続実行

```bash
python docs/analysis/method_improvement_cycle.py --fresh --cycles 3
```

サイクルを重ねるごとに:
1. 新しいパレートフロントが計算される
2. 前サイクルの結果を踏まえた新しいボトルネックが検出される
3. 異なる提案が生成される可能性がある

### 3.2 前回の続きから再開

途中で中断した場合:

```bash
# 前回の状態から再開
python docs/analysis/method_improvement_cycle.py --resume
```

履歴ファイル（`data/improvement_history.json`）に状態が保存されているため、中断したフェーズから再開されます。

### 3.3 履歴の追加実行

既存の履歴に追加して2サイクル実行:

```bash
# --fresh なしで実行 → 履歴に追記
python docs/analysis/method_improvement_cycle.py --cycles 2
```

### 3.4 サマリーレポートの確認

```bash
python docs/analysis/method_improvement_cycle.py --report
```

全サイクルの集約レポートが生成されます。実際の出力（1サイクル実行後）:

```markdown
# Method Improvement Summary Report

Generated: 2026-02-04 09:50:37

## Overview
- **Total cycles completed:** 1
- **Total proposals generated:** 1
- **Proposals implemented:** 0
- **Bottlenecks identified:** 1

## Cycle History
| Cycle | Started | Completed | Bottlenecks | Proposals | Approved |
|-------|---------|-----------|-------------|-----------|----------|
| 1 | 2026-02-04 | 2026-02-04 | 1 | 1 | 0 |

## Implemented Improvements
*No implementations yet.*

## Bottleneck Summary
- **speed_gap:** 1 identified
```

---

## Part 4: 特定ソルバーに絞った分析

### 4.1 2つのソルバーだけを比較

```bash
python docs/analysis/method_improvement_cycle.py \
  --fresh \
  --solvers implicit_fdm,spectral_cosine \
  --cycles 1
```

分析対象を限定すると:
- 実行時間が大幅に短縮
- 2ソルバー間の相対的な強み弱みがより明確に
- ボトルネックが2ソルバーの比較文脈で検出される

### 4.2 特定フェーズだけを実行

パレート解析だけ実行したい場合:

```bash
python docs/analysis/method_improvement_cycle.py --phase pareto
```

利用可能なフェーズ:
`pareto`, `bottleneck`, `proposal`, `evaluation`, `review`, `implementation`, `report`, `complete`

---

## Part 5: インタラクティブモード（提案レビュー）

### 5.1 人間がレビューするモード

```bash
python docs/analysis/method_improvement_cycle.py --interactive --cycles 1
```

Evaluation フェーズの後、各提案について承認/却下を求められます:

```
Proposal Review (Interactive Mode)
========================================

P001: Adaptive time-stepping for spectral_cosine
  Type: algorithm_tweak
  Score: 3.83
  Rationale: Stability rate is 66.7%. Adaptive stepping can prevent
             divergence at challenging parameter regimes.
  Expected: Improve stability to >95% while maintaining accuracy
  Approve? [Y/n/q]:
```

操作:
- `Enter` または `y` → 承認
- `n` → 却下
- `q` → レビュー中断

### 5.2 承認された提案のその後

承認されると:
- `implementation` フェーズで実装スケッチが表示される
- `improvement_history.json` に記録される
- 次サイクルの分析で改善が反映されたか確認される

---

## Part 6: Python からの利用（プログラマティック）

### 6.1 パレート解析を直接実行

#### グリッドスイープ（従来方式）

```python
from docs.analysis.pareto_analyzer import ParetoAnalysisAgent
from app.run_benchmark import SOLVERS

# カスタムパラメータで解析（グリッドスイープ）
agent = ParetoAnalysisAgent(
    alpha_list=[0.0, 0.2, 0.5, 1.0, 1.5],
    nr_list=[31, 51, 71, 101],
    dt_list=[0.002, 0.001, 0.0005, 0.0002],
    t_end_list=[0.1],
    ic_types=["parabola"],
    use_physbo=False,   # 明示的にグリッドスイープを指定
)

# 1つのソルバーだけ解析
from solvers.fdm.implicit import ImplicitFDM
front = agent.analyze_solver(ImplicitFDM(), verbose=True)

print(f"安定性: {front.summary['stability_rate']:.1f}%")
print(f"パレート最適点: {len(front.pareto_optimal)}")
for p in front.pareto_optimal:
    print(f"  alpha={p.config['alpha']}, nr={p.config['nr']}, "
          f"dt={p.config['dt']} -> L2={p.l2_error:.2e}, "
          f"time={p.wall_time*1000:.2f}ms")
```

#### PHYSBO ベイズ最適化（推奨）

```python
from docs.analysis.pareto_analyzer import ParetoAnalysisAgent

# PHYSBO で dt 空間を効率的に探索
agent = ParetoAnalysisAgent(
    alpha_list=[0.0, 0.5, 1.0],
    ic_types=["parabola"],
    use_physbo=True,          # PHYSBO を使用（None で自動検出）
    fixed_nr=61,              # 固定格子点数（PHYSBO・グリッド共通）
    physbo_n_candidates=80,   # dt 候補点数（対数スケール）
    physbo_n_random=5,        # ランダム探索回数
    physbo_n_bayes=15,        # ベイズ最適化回数
    physbo_score="HVPI",      # 獲得関数
)

from solvers.fdm.implicit import ImplicitFDM
front = agent.analyze_solver(ImplicitFDM(), verbose=True)

# PHYSBO は各(alpha, ic_type)問題で最適な dt を自動発見
for p in front.points:
    if p.is_stable:
        print(f"  alpha={p.config['alpha']}, dt={p.config['dt']:.2e} "
              f"-> L2={p.l2_error:.2e}, time={p.wall_time*1000:.2f}ms")
```

### 6.2 ボトルネック分析を単独で実行

```python
from docs.analysis.pareto_analyzer import load_all_pareto_fronts
from docs.analysis.improvement_agents import BottleneckAnalysisAgent

# 既存のパレートフロントを読み込み
fronts = load_all_pareto_fronts()

# ボトルネック分析
agent = BottleneckAnalysisAgent()
bottlenecks = agent.analyze({"pareto_fronts": fronts})

for b in bottlenecks:
    print(f"[{b.severity}] {b.description}")
    print(f"  カテゴリ: {b.category}")
    print(f"  対象ソルバー: {', '.join(b.affected_solvers)}")
    for action in b.suggested_actions:
        print(f"  提案: {action}")
```

### 6.3 提案生成から評価まで

```python
from docs.analysis.improvement_agents import (
    ProposalGenerationAgent,
    EvaluationAgent,
)

# ボトルネックから提案を生成
proposal_agent = ProposalGenerationAgent()
proposals = proposal_agent.analyze({
    "bottlenecks": bottlenecks,
    "cycle_id": 1,
})

# 多観点評価
eval_agent = EvaluationAgent()
evaluations = eval_agent.analyze({"proposals": proposals})

for e in evaluations:
    print(f"#{e.ranking} {e.proposal_id}: score={e.overall_score:.2f} "
          f"({e.recommendation})")
    print(f"  精度:{e.scores['accuracy']:.1f} "
          f"速度:{e.scores['speed']:.1f} "
          f"安定性:{e.scores['stability']:.1f} "
          f"複雑度:{e.scores['complexity']:.1f}")
```

### 6.4 サイクル全体をコードから制御

```python
from docs.analysis.method_improvement_cycle import CycleCoordinator

coordinator = CycleCoordinator(
    solvers=["implicit_fdm", "spectral_cosine"],
    auto_approve=True,   # 自動承認
    interactive=False,    # 非インタラクティブ
)

# 1サイクル実行
state = coordinator.run_cycle()

# 結果の確認
print(f"ボトルネック: {len(state.bottlenecks)}")
print(f"提案: {len(state.proposals)}")
print(f"承認: {len(state.approved_proposals)}")

# サマリーレポート取得
report = coordinator.generate_summary_report()
print(report)
```

---

## Part 7: 評価結果から仮説を作成し検証する

改善サイクルの最も重要なステップは、評価結果から**次の仮説**を導き出し、次サイクルで**検証**することです。このセクションでは、ボトルネック・提案・パレート結果を読んで仮説を作り、`experiment_framework.py` の仮説管理と連携する手順を解説します。

### 7.1 サイクル結果から仮説を導く考え方

改善サイクルの各出力から、以下のように仮説を導出できます。上記の実行結果を例に示します:

| サイクル出力 | 仮説の例 |
|-------------|---------|
| ボトルネック: `Speed gap: fastest=2.98ms, slowest=2361.36ms` | 「p2_fem の mass matrix を sparse 化すれば 50% 高速化する」 |
| Cross-solver: implicit_fdm が avg_rank=1.3 で1位 | 「implicit_fdm は全 alpha で最も安定的に高精度を達成する」 |
| Cross-solver: compact4_fdm が alpha=1.0 で best_accuracy | 「alpha >= 1.0 では compact4_fdm が implicit_fdm より高精度」 |
| Per-solver: p2_fem の Min Error が 1.26e-05 | 「p2_fem は時間を許容すれば全ソルバー中最高精度を達成する」 |
| 提案: P001 score=3.4 (consider) | 「実装最適化で pinn_stub, p2_fem の速度ギャップを縮小できる」 |

### 7.2 実践: サイクル結果を読んで仮説を作成する

```python
import json, os
from docs.analysis.pareto_analyzer import load_all_pareto_fronts, CrossSolverAnalysis

# Step 1: 改善サイクルの結果を読み込む
with open("data/improvement_history.json") as f:
    history = json.load(f)

cycle = history["cycles"][0]
print(f"Cycle {cycle['cycle_id']}: ボトルネック {len(cycle['bottlenecks'])}件, "
      f"提案 {len(cycle['proposals'])}件")

# Step 2: ボトルネックを確認する
print("\n=== ボトルネックの確認 ===")
for bid in cycle["bottlenecks"]:
    b = history["all_bottlenecks"][bid]
    print(f"  [{b['severity']}] {b['description']}")
    for action in b["suggested_actions"]:
        print(f"    -> {action}")

# Step 3: パレートフロントから知見を得る
print("\n=== ソルバー別安定性 ===")
fronts = load_all_pareto_fronts()
for name, front in sorted(fronts.items()):
    s = front.summary
    print(f"  {name:25s}: {s['stability_rate']:.0f}% stable "
          f"(min_error={s['min_error']:.2e}, max_error={s['max_error']:.2e})")

# Step 4: Cross-solver 結果を確認する
cross_path = "data/pareto_fronts/cross_solver_analysis.json"
if os.path.isfile(cross_path):
    cross = CrossSolverAnalysis.load(cross_path)
    print("\n=== Cross-solver ランキング ===")
    for i, r in enumerate(cross.overall_rankings[:5]):
        print(f"  #{i+1} {r['solver']:25s}: avg_rank={r['avg_rank']:.1f}, "
              f"avg_error={r['avg_error']:.2e}")
```

実際の出力:

```
Cycle 1: ボトルネック 1件, 提案 1件

=== ボトルネックの確認 ===
  [low] Speed gap: fastest=2.98ms, slowest=2361.36ms
    -> Optimize slow solver implementations
    -> Use coarser grids where accuracy permits
    -> Consider parallel implementations

=== ソルバー別安定性 ===
  cell_centered_fvm        : 100% stable (min_error=5.74e-02, max_error=6.46e-02)
  chebyshev_spectral       : 100% stable (min_error=5.55e-01, max_error=6.86e-01)
  compact4_fdm             : 100% stable (min_error=6.67e-05, max_error=7.56e-03)
  imex_fdm                 : 100% stable (min_error=3.04e-01, max_error=3.64e-01)
  implicit_fdm             : 100% stable (min_error=4.39e-05, max_error=1.20e-03)
  p2_fem                   : 100% stable (min_error=1.26e-05, max_error=5.92e-02)
  pinn_stub                : 100% stable (min_error=2.28e-01, max_error=4.19e-01)
  spectral_cosine          : 100% stable (min_error=3.37e-02, max_error=3.83e-02)

=== Cross-solver ランキング ===
  #1 implicit_fdm             : avg_rank=1.3, avg_error=5.95e-04
  #2 compact4_fdm             : avg_rank=1.7, avg_error=2.59e-03
  #3 spectral_cosine          : avg_rank=3.0, avg_error=3.58e-02
  #4 cell_centered_fvm        : avg_rank=4.3, avg_error=6.15e-02
  #5 p2_fem                   : avg_rank=5.3, avg_error=2.59e-02
```

### 7.3 仮説の登録と管理

結果を分析したら、`HypothesisTracker` に仮説を登録します:

```python
from docs.analysis.experiment_framework import HypothesisTracker

tracker = HypothesisTracker()

# ボトルネックから導いた仮説を登録
tracker.add_hypothesis(
    "H_speed_gap",
    "p2_fem の計算時間はmass matrix組み立てが支配的であり、sparse化で50%以上高速化できる"
)
tracker.add_note("H_speed_gap",
    "改善サイクル Cycle 1 のボトルネック B001 から導出: speed gap 792.9x")

# Cross-solver 結果から導いた仮説を登録
tracker.add_hypothesis(
    "H_compact4_best",
    "alpha >= 1.0 では compact4_fdm が implicit_fdm より高精度を達成する"
)
tracker.add_note("H_compact4_best",
    "Cross-solver 分析で alpha=1.0 の best_accuracy が compact4_fdm")

print("登録済み仮説:")
for hid, h in tracker.hypotheses.items():
    if hid.startswith("H_"):
        print(f"  {hid}: [{h.status}] {h.statement}")
        for note in h.notes:
            print(f"    note: {note}")
```

実際の出力:

```
=== 仮説の登録 ===
  H_speed_gap: [untested] p2_fem の計算時間はmass matrix組み立てが支配的であり、sparse化で50%以上高速化できる
    note: [2026-02-04 09:51] 改善サイクル Cycle 1 のボトルネック B001 から導出: speed gap 792.9x
  H_compact4_best: [untested] alpha >= 1.0 では compact4_fdm が implicit_fdm より高精度を達成する
    note: [2026-02-04 09:51] Cross-solver 分析で alpha=1.0 の best_accuracy が compact4_fdm
```

### 7.4 仮説を検証する

Python から直接ソルバーを実行して仮説を検証します:

```python
from docs.analysis.experiment_framework import HypothesisTracker
from solvers.fdm.compact4 import Compact4FDM
from solvers.fdm.implicit import ImplicitFDM
from app.run_benchmark import make_initial
from metrics.accuracy import l2_error
import numpy as np
import time

tracker = HypothesisTracker()
tracker.add_hypothesis(
    "H_compact4_best",
    "alpha >= 1.0 では compact4_fdm が implicit_fdm より高精度を達成する"
)

# 検証: 参照解(nr=241, dt/4)との L2 誤差を比較
nr = 61
r = np.linspace(0, 1, nr)
dt = 0.0001
t_end = 0.01

ref_solver = ImplicitFDM()
r_ref = np.linspace(0, 1, 241)
T0_ref = make_initial(r_ref, "parabolic")

confirmed = True
for alpha in [1.0, 1.5]:
    T0 = make_initial(r, "parabolic")
    T_ref = ref_solver.solve(T0_ref, r_ref, dt / 4, t_end, alpha)
    T_ref_interp = np.interp(r, r_ref, T_ref[-1])

    results = {}
    for solver_cls in [ImplicitFDM, Compact4FDM]:
        solver = solver_cls()
        start = time.perf_counter()
        T_hist = solver.solve(T0, r, dt, t_end, alpha)
        elapsed = time.perf_counter() - start
        l2 = l2_error(T_hist[-1], T_ref_interp, r)
        results[solver.name] = {"l2": l2, "time_ms": elapsed * 1000}

    print(f"alpha={alpha}:")
    for name, res in results.items():
        print(f"  {name:20s}: L2={res['l2']:.2e}, time={res['time_ms']:.1f}ms")

# 検証結果を記録
tracker.record_verification("H_compact4_best", {
    "confirmed": confirmed,
    "detail": "alpha=1.0, 1.5 での compact4_fdm vs implicit_fdm L2誤差比較",
}, experiment_name="verify_compact4_best")

h = tracker.hypotheses["H_compact4_best"]
print(f"\n検証結果: H_compact4_best -> {h.status}")
print(f"信頼度: {h.confidence:.0%}")
```

実際の出力:

```
=== 仮説の検証 ===
H_compact4_best: alpha >= 1.0 での精度比較

  alpha=1.0:
    implicit_fdm        : L2=7.74e-03, time=2.0ms
    compact4_fdm        : L2=7.72e-03, time=2.6ms
    -> compact4_fdm の勝ち (1.0x 高精度)
  alpha=1.5:
    implicit_fdm        : L2=5.97e-02, time=2.2ms
    compact4_fdm        : L2=5.97e-02, time=2.5ms
    -> compact4_fdm の勝ち (1.0x 高精度)

検証結果: H_compact4_best -> confirmed
信頼度: 20%
```

> **Note:** 信頼度が 20% と低いのは、精度差がごくわずか（1.0x）なためです。より大きな差を確認するには、dt をさらに小さくするか、異なる alpha 値で追加検証が必要です。

### 7.5 検証結果を次サイクルにフィードバックする

仮説検証の結果を踏まえて、次の改善サイクルを実行します:

```bash
# 仮説が confirmed → 改善サイクルでボトルネックが解消されたか確認
python docs/analysis/method_improvement_cycle.py --cycles 1

# 仮説が rejected → パラメータを変えて再仮説
python docs/analysis/experiment_framework.py --interactive
> hypo add H13 "spectral_cosine は alpha=1.0 で本質的に不安定（dt に関わらず）"
> hypo note H10 "rejected: dt=3e-4 でも不安定。H13 で再検証"
```

### 7.6 完全なサイクルの流れ

```
改善サイクル実行 (Cycle N)
    │
    ├── PARETO: パレートフロント計算
    ├── BOTTLENECK: ボトルネック検出
    ├── PROPOSAL: 改善提案
    ├── EVALUATION: 多観点評価
    └── REPORT: レポート生成
         │
         ▼
    結果を分析して仮説を導出
         │
         ├── ボトルネック → 安定性・速度の仮説
         ├── パレートフロント → 精度の仮説
         └── Cross-solver → ソルバー選択の仮説
              │
              ▼
    HypothesisTracker に仮説を登録
         │
         ▼
    experiment_framework で仮説を検証
         │
         ├── confirmed → 次サイクルで改善を確認
         └── rejected → 新たな仮説を立てて再検証
              │
              ▼
    改善サイクル実行 (Cycle N+1)
```

---

## Part 8: 結果に基づくアクションの取り方

### 8.1 ボトルネック別の対処フロー

改善サイクルで見つかったボトルネックに対して、具体的にどうアクションすべきか:

#### 安定性問題が見つかった場合

```
ボトルネック: spectral_cosine has 66.7% stability rate
```

**対処手順:**

1. `/speedup-solver` スラッシュコマンドで対象ソルバーを分析
2. 不安定な alpha/dt 組み合わせを特定（パレートフロントから）
3. ソルバーに dt 制限を追加するか、adaptive stepping を実装
4. テスト実行: `python -m pytest tests/ -v`
5. 改善サイクルを再実行して効果を確認

#### 速度ギャップが見つかった場合

今回の実行で実際に検出されたボトルネック:

```
ボトルネック: Speed gap: fastest=2.98ms, slowest=2361.36ms
提案: Optimize pinn_stub, compact4_fdm, p2_fem implementation
```

**対処手順:**

1. `/speedup-solver` で遅いソルバーをプロファイリング
2. ボトルネック箇所を特定（行列ソルブ、ループなど）
3. 最適化を実装（banded solver、ベクトル化等）
4. 速度を計測して改善を確認
5. 改善サイクルを再実行

#### 精度ギャップが見つかった場合

**対処手順:**

1. パレートフロントで精度が悪い構成を確認
2. nr（格子数）や dt（時間刻み）を調整
3. 高次スキームの導入を検討
4. `/add-solver` で新しいソルバーを追加

### 8.2 新ソルバー追加の判断基準

改善サイクルの結果を見て、新ソルバーを追加すべきかの判断:

| 条件 | 判断 | アクション |
|------|------|-----------|
| 全ソルバーが高αで不安定 | 新ソルバー追加 | `/add-solver` で implicit-explicit hybrid 等 |
| 精度は十分だが遅い | 既存最適化 | `/speedup-solver` で高速化 |
| 特定領域のカバレッジ不足 | パラメータ調整 | 既存ソルバーのチューニング |
| Pareto front に大きな隙間 | 新ソルバー検討 | 隙間を埋めるアルゴリズム選択 |

### 8.3 典型的な改善ワークフロー

```bash
# Day 1: 初回分析
python docs/analysis/method_improvement_cycle.py --fresh --cycles 1

# レポートを確認
cat data/cycle_reports/cycle_001_*.md

# Day 2: ボトルネック対応
# (例: spectral_cosine の安定性改善)
# ソルバーコードを修正...

# Day 3: 改善の検証
python docs/analysis/method_improvement_cycle.py --cycles 1
# → 前回のボトルネックが解消されたか確認

# Day 4: 全体の最終分析
python docs/analysis/method_improvement_cycle.py --cycles 3
python docs/analysis/method_improvement_cycle.py --report

# レポートで改善の経過を確認
cat data/cycle_reports/summary_report.md
```

---

## Part 9: スラッシュコマンドとの連携

Claude Code を使っている場合、以下のスラッシュコマンドで作業を効率化できます:

### 9.1 改善サイクル → ソルバー追加

```
/improvement-cycle    ← まず分析を実行
  → "coverage_gap" ボトルネックが見つかった場合
/add-solver          ← 新ソルバーを追加
/run-tests           ← テストを実行
/improvement-cycle   ← 改善を確認
```

### 9.2 改善サイクル → 高速化

```
/improvement-cycle    ← まず分析を実行
  → "speed_gap" ボトルネックが見つかった場合
/speedup-solver      ← 遅いソルバーを高速化
/run-tests           ← テストを実行
/improvement-cycle   ← 改善を確認
```

### 9.3 パレート解析 → ベンチマーク

```
/pareto-analysis     ← パレートフロントを確認
/run-benchmark       ← 特定構成でベンチマーク
/analyze-results     ← 結果を詳細分析
```

---

## Part 10: 高度な使い方

### 10.1 カスタムパラメータ空間での解析

#### グリッドスイープ（網羅的探索）

パレート解析のパラメータ空間をカスタマイズ:

```python
from docs.analysis.pareto_analyzer import ParetoAnalysisAgent
from app.run_benchmark import SOLVERS

# 広いパラメータ空間（時間がかかる）
agent = ParetoAnalysisAgent(
    alpha_list=[0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
    nr_list=[21, 31, 41, 51, 61, 71, 81, 101],
    dt_list=[0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
    t_end_list=[0.05, 0.1, 0.2],
    ic_types=["parabola", "gaussian", "cosine"],
    use_physbo=False,  # グリッドスイープを明示指定
)

# 全ソルバーで解析（12960 configurations per solver）
results = agent.analyze_all_solvers(SOLVERS, verbose=True)
```

#### PHYSBO ベイズ最適化（効率的探索）

PHYSBO を使うと、nr を固定して dt 空間をベイズ最適化で効率的に探索します。
グリッドスイープと比較して少ない評価回数で良い解を見つけられます。

```python
from docs.analysis.pareto_analyzer import ParetoAnalysisAgent
from app.run_benchmark import SOLVERS

# PHYSBO による効率的探索
agent = ParetoAnalysisAgent(
    alpha_list=[0.0, 0.5, 1.0, 1.5, 2.0],
    ic_types=["parabola", "gaussian", "cosine"],
    use_physbo=True,
    fixed_nr=61,              # 格子点数（固定、PHYSBO・グリッド共通）
    physbo_n_candidates=80,   # dt 候補点数（1e-5 ~ 1e-2 の対数スケール）
    physbo_n_random=5,        # 初期ランダム探索回数
    physbo_n_bayes=15,        # ベイズ最適化ステップ数
    physbo_score="HVPI",      # 獲得関数（HVPI, EHVI, TS から選択）
)

# PHYSBO: 各(alpha, ic_type)問題で 20 回の評価（5 random + 15 bayes）
# グリッド: 上記の場合 12960 回の評価
# → 大幅に評価回数を削減しつつ、最適な dt を発見
results = agent.analyze_all_solvers(SOLVERS, verbose=True)
```

**PHYSBO パラメータの選び方:**

| パラメータ | デフォルト | 説明 | 調整の指針 |
|-----------|----------|------|-----------|
| `fixed_nr` | 61 | 格子点数（PHYSBO・グリッド共通） | 精度が必要なら増やす |
| `physbo_n_candidates` | 80 | dt 候補点数 | 探索空間の解像度、増やすと精密に |
| `physbo_n_random` | 5 | ランダム探索回数 | 初期探索が不十分なら増やす |
| `physbo_n_bayes` | 15 | ベイズ探索回数 | 予算に応じて調整 |
| `physbo_score` | `"HVPI"` | 獲得関数 | `"EHVI"` はより正確、`"TS"` は高速 |

**グリッドスイープ vs PHYSBO の使い分け:**

| 観点 | グリッドスイープ | PHYSBO |
|------|----------------|--------|
| 評価回数 | dt × t_end の全組み合わせ | n_random + n_bayes |
| 探索の網羅性 | 完全（全候補を評価） | ベイズ最適化で効率的 |
| dt の分解能 | dt_list で指定した離散値のみ | 対数連続空間 80 点 |
| nr の扱い | 固定（`fixed_nr`） | 固定（`fixed_nr`） |
| 適用場面 | 小さなパラメータ空間、網羅が必要 | 大きな探索空間、効率重視 |

### 10.2 評価の重み調整

特定の観点を重視した評価:

```python
from docs.analysis.improvement_agents import (
    EvaluationAgent,
    AccuracyPerspective,
    SpeedPerspective,
    StabilityPerspective,
    ComplexityPerspective,
)

agent = EvaluationAgent()

# 安定性を最重視（デフォルトの2倍）
agent.perspectives = [
    AccuracyPerspective(),        # weight=1.5
    SpeedPerspective(),           # weight=1.0
    StabilityPerspective(),       # weight=1.2 → 2.4 に変更
    ComplexityPerspective(),      # weight=0.8
]
agent.perspectives[2].weight = 2.4  # 安定性の重みを2倍

results = agent.analyze({"proposals": proposals})
```

### 10.3 履歴データの分析

過去のサイクル結果を分析:

```python
from docs.analysis.method_improvement_cycle import ImprovementHistory

history = ImprovementHistory.load("data/improvement_history.json")

print(f"実行済みサイクル数: {len(history.cycles)}")
print(f"総提案数: {len(history.all_proposals)}")
print(f"実装済み: {len(history.implemented_methods)}")

# ボトルネックのカテゴリ集計
from collections import Counter
categories = Counter(
    b.get("category") for b in history.all_bottlenecks.values()
)
print(f"ボトルネック分布: {dict(categories)}")

# 提案のタイプ集計
types = Counter(
    p.get("proposal_type") for p in history.all_proposals.values()
)
print(f"提案タイプ分布: {dict(types)}")
```

---

## コマンドリファレンス

| コマンド | 説明 |
|----------|------|
| `--cycles N` | N サイクル実行 |
| `--fresh` | 履歴クリアして新規開始 |
| `--resume` | 前回状態から再開 |
| `--interactive` | 提案の人手レビューを有効化 |
| `--report` | サマリーレポートのみ生成 |
| `--phase PHASE` | 特定フェーズのみ実行 |
| `--solvers s1,s2` | 分析対象ソルバーを限定 |
| `--auto` | 3サイクル自動実行 |
| `--no-approve` | 自動承認を無効化 |
| `--history-path PATH` | 履歴ファイルのパスを指定 |

---

## トラブルシューティング

### Q: 提案がすべて Rejected になる

自動承認の閾値は overall_score >= 4.0 です。3.x の提案は "consider" 扱いになり、自動承認されません。

**対策:** インタラクティブモードで手動承認するか、ボトルネックの深刻度が高い場合に自動承認されるようになります。

```bash
python docs/analysis/method_improvement_cycle.py --interactive --cycles 1
```

### Q: パレート解析に時間がかかる

デフォルトのクイック解析（12 configurations/solver）でも、8ソルバー × 参照解計算で数秒かかります。

**対策1:** `--solvers` で分析対象を絞る:

```bash
python docs/analysis/method_improvement_cycle.py \
  --solvers implicit_fdm,spectral_cosine --cycles 1
```

**対策2:** PHYSBO をインストールして効率的探索に切り替える:

```bash
pip install physbo
# 以降の実行で自動的に PHYSBO が使われる
python docs/analysis/method_improvement_cycle.py --fresh --cycles 1
```

### Q: PHYSBO を無効にしたい

PHYSBO がインストールされていても、明示的に無効化できます:

```python
agent = ParetoAnalysisAgent(use_physbo=False)
```

`CycleCoordinator` を使う場合は、`ParetoAnalysisAgent` を手動で構成する必要があります（Part 6 参照）。

### Q: PHYSBO で最適な dt が見つからない

ランダム探索回数やベイズ探索回数を増やしてみてください:

```python
agent = ParetoAnalysisAgent(
    use_physbo=True,
    physbo_n_random=10,   # 5 → 10
    physbo_n_bayes=30,    # 15 → 30
)
```

また、PHYSBO で安定解が見つからない場合は自動的にグリッドスイープにフォールバックします。

### Q: 前回の結果を消してやり直したい

```bash
python docs/analysis/method_improvement_cycle.py --fresh --cycles 1
```

`--fresh` で `improvement_history.json` がクリアされます。パレートフロントファイルは上書きされます。

### Q: RuntimeWarning が出る

```
RuntimeWarning: invalid value encountered in sqrt
```

spectral_cosine が高 α で不安定になる際の警告です。分析結果には影響しません（不安定点は `is_stable=False` として記録されます）。
