# マルチエージェント手法改善サイクル — 技術ノート

## 1. 概要

本プロジェクトでは、核融合プラズマの熱輸送PDEベンチマークにおいて、8つの数値ソルバー（FDM, FEM, FVM, Spectral, PINN）の性能を自動的に解析・改善するエージェントパイプラインを構築した。

**目的**: ソルバー間の精度・速度・安定性のトレードオフを自動的にパレート解析し、ボトルネックを特定し、改善提案を生成・評価する閉ループ改善サイクルを実現する。

> **注記: 実行モデルについて**
>
> 本システムは「マルチエージェント」と称しているが、実行モデルは**逐次パイプライン**である。
> 各エージェントは前段の出力を入力として必要とするため、Phase 1→2→3→...→7 の順に
> 直列実行される（`CycleCoordinator.run_cycle()` がPHASESリストを`for`ループで走査）。
> 並行実行・交渉・競合解決といった一般的なマルチエージェント機構は含まない。
>
> 「マルチエージェント」という名称は、**役割分担された複数の専門エージェントが協調する**
> という設計パターンを指しており、並行性を意味するものではない。
>
> 唯一の並列化可能ポイントは EvaluationAgent 内の4つの Perspective である（後述）。

**主要モジュール**:
- `docs/analysis/pareto_analyzer.py` — パレート解析エージェント + PHYSBO統合
- `docs/analysis/improvement_agents.py` — ボトルネック検出・提案生成・評価・レポートエージェント
- `docs/analysis/method_improvement_cycle.py` — サイクル全体の制御（CycleCoordinator）
- `docs/analysis/experiment_framework.py` — 仮説管理・実験実行・解析

## 2. エージェント一覧

### 2.1 ParetoAnalysisAgent（パレート解析）

**役割**: パラメータスイープを実行し、各ソルバーの精度 vs 速度パレートフロントを計算する。

**入力**:
- ソルバーインスタンス群
- パラメータ空間（alpha, nr, dt, t_end, ic_types）

**出力**:
- `ParetoFront` — ソルバーごとの安定点・パレート最適点
- `CrossSolverAnalysis` — ソルバー横断比較（ランキング、勝利回数、カバレッジギャップ）

**主要メソッド**:
| メソッド | 説明 |
|---------|------|
| `analyze_solver()` | ソルバー単体のパレート解析（PHYSBO or グリッド） |
| `run_quick_analysis()` | 縮小パラメータ空間での高速解析 |
| `run_quick_cross_solver()` | ソルバー横断比較 |
| `_find_best_for_problem_physbo()` | PHYSBOベイズ最適化によるdt探索 |

### 2.2 BottleneckAnalysisAgent（ボトルネック検出）

**役割**: パレートフロントとソルバー横断結果から性能ボトルネックを自動検出する。

**入力**: ParetoFront辞書, CrossSolverAnalysis
**出力**: `Bottleneck[]`（深刻度: high/medium/low）

**検出カテゴリ**:

| カテゴリ | 説明 | 閾値 |
|---------|------|------|
| `stability` | 安定性率が低い | < 90% |
| `accuracy_gap` | ソルバー間の精度差が大きい | — |
| `speed_gap` | 実行速度差が大きい | 100倍以上 |
| `coverage_gap` | 単一ソルバーがパレートフロントを支配 | > 80% |
| `no_stable_solver` | 安定なソルバーが存在しない問題 | 0% |
| `solver_dominance` | 1つのソルバーが大半の問題で勝利 | > 80% |
| `cross_solver_accuracy_gap` | 最良ソルバーでも精度不足 | L2 > 0.5 |
| `solver_instability` | 多くの問題でソルバーが失敗 | < 80% |

### 2.3 ProposalGenerationAgent（改善提案生成）

**役割**: ボトルネックリストから具体的な改善提案を自動生成する。

**入力**: `Bottleneck[]`, cycle_id
**出力**: `MethodProposal[]`（タイプ: parameter_tuning / algorithm_tweak / new_solver）

**提案パターン**:
- **安定性ボトルネック** → 適応タイムステッピング、dtの制約条件
- **精度ボトルネック** → 解像度の向上（nr増加）
- **速度ボトルネック** → ベクトル化、バンドソルバー最適化
- **カバレッジボトルネック** → 弱いソルバーのニッチ探索
- **安定ソルバー不在** → 適応リトライ＋新規IMEXソルバー提案

### 2.4 EvaluationAgent + 4つの視点（多観点評価）

**役割**: 提案を4つの独立した視点から重み付きスコアリングで評価する。

**入力**: `MethodProposal[]`
**出力**: `EvaluationResult[]`（スコア0-5、推薦: approve/consider/reject）

| 視点 | 重み | 高評価の提案タイプ |
|-----|------|-----------------|
| AccuracyPerspective | 1.5 | 解像度・適応手法 |
| SpeedPerspective | 1.0 | 最適化・パラメータ調整 |
| StabilityPerspective | 1.2 | 安定性重視の提案 |
| ComplexityPerspective | 0.8 | パラメータ調整(4.5) > アルゴリズム(2.5) > 新ソルバー(1.5) |

**判定基準**:
- `approve`: 加重平均スコア ≥ 4.0
- `consider`: 加重平均スコア ≥ 3.0
- `reject`: 加重平均スコア < 3.0

**現在の実装**: 各Perspectiveはルールベース（文字列マッチング + 提案タイプ判定）で、
計算コストは数マイクロ秒程度。4視点を`for`ループで逐次実行している。

**並列化の可能性**: 4つのPerspectiveは互いに独立（入力を共有するが出力は独立）なため、
パイプライン中で唯一並列化が構造的に可能な箇所である。ただし、現在のルールベース実装は
極めて軽量であり、並列化のオーバーヘッドがスコアリング自体のコストを上回る。
将来的にPerspectiveがLLM呼び出しやシミュレーション実行を伴う重い処理に拡張される場合は、
`concurrent.futures.ThreadPoolExecutor`（I/Oバウンド）や `ProcessPoolExecutor`（CPUバウンド）
での並列化が有効になる。MPIは単一マシン内のこのユースケースには過剰であり、
分散クラスタ上で各Perspectiveが異なるシミュレーションを実行する場合にのみ適切となる。

### 2.5 ReportAgent（レポート合成）

**役割**: サイクル全体の結果をマークダウンレポートとして合成する。

**出力セクション**:
1. エグゼクティブサマリー（ソルバー数、ボトルネック数、提案数、承認数）
2. ソルバー横断ランキングテーブル
3. ソルバー別パレート解析結果
4. ボトルネック一覧（深刻度・提案アクション付き）
5. 提案と多観点評価スコアテーブル
6. 次サイクルへの推奨事項

### 2.6 HypothesisTracker + ExperimentRunner（仮説管理・検証）

**HypothesisTracker**:
- 仮説を登録・追跡（id, statement, status, confidence）
- ステータス遷移: `untested` → `confirmed` / `rejected` / `inconclusive`
- 検証履歴をJSON永続化（`data/hypotheses_memo.json`）

**ExperimentRunner**:
- 実験設定に基づきソルバーを実行（参照解は4倍解像度ImplicitFDM）
- 結果をCSVデータベースに蓄積（`data/experiments.csv`）
- 19カラム: experiment_name, alpha, nr, dt, t_end, solver, l2_error, wall_time, is_stable, ...

**定義済み実験**:
| 実験名 | 目的 |
|--------|------|
| `stability_map` | alpha × dt 空間の安定性マップ |
| `ic_comparison` | 初期条件による最適ソルバーの変化 |
| `pinn_comparison` | PINN vs FDM/Spectral比較 |
| `linear_regime` | 線形領域（\|dT/dr\| < 0.5）での性能比較 |
| `fine_sweep` | 網羅的パラメータスイープ |

## 3. データフロー図（逐次パイプライン）

全フェーズは**逐次実行**される。各エージェントは前段の出力に依存するため並行実行不可。

```
  ┌────────────────────────────────────────────────────────────────┐
  │                   CycleCoordinator                            │
  │            (method_improvement_cycle.py)                       │
  │   for phase in PHASES: run_phase(phase, state)  ← 逐次ループ  │
  └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    Phase 1  ┌──────────────┐
             │  Pareto      │  出力: ParetoFront{}, CrossSolverAnalysis
             │  Analysis    │  state.phase = "bottleneck" に遷移
             │  Agent       │
             └──────┬───────┘
                    │ (前段の出力を入力として渡す)
                    ▼
    Phase 2  ┌──────────────────┐
             │  Bottleneck      │  入力: ParetoFront{}, CrossSolverAnalysis
             │  Analysis Agent  │  出力: Bottleneck[]
             └──────┬───────────┘
                    │
                    ▼
    Phase 3  ┌──────────────┐
             │  Proposal    │  入力: Bottleneck[]
             │  Generation  │  出力: MethodProposal[]
             │  Agent       │
             └──────┬───────┘
                    │
                    ▼
    Phase 4  ┌──────────────┐   ┌──────────────────────────────────────┐
             │  Evaluation  │   │  4 Perspectives (逐次, 並列化可能*): │
             │  Agent       │──▶│  Accuracy(1.5) + Speed(1.0)          │
             │              │   │  + Stability(1.2) + Complexity(0.8)  │
             └──────┬───────┘   └──────────────────────────────────────┘
                    │           * 現在はforループ。将来の重い処理拡張時に
                    │             concurrent.futuresで並列化可能
                    ▼
    Phase 5-6 ┌──────────────┐
              │  Review &    │  approve (≥4.0) / consider (≥3.0) / reject
              │  Implement   │  → 実装スケッチ提示
              └──────┬───────┘
                     │
                     ▼
    Phase 7  ┌──────────────┐
             │  Report      │  → data/cycle_reports/cycle_NNN.md
             │  Agent       │  → improvement_history.json 更新
             └──────────────┘

  ─── 別系統（独立実行） ─────────────────────────────────────

  ┌──────────────┐     ┌──────────────────┐
  │  Hypothesis  │◀───▶│  Experiment      │
  │  Tracker     │     │  Runner          │
  │              │     │                  │
  │ 仮説登録     │     │ 実験実行         │
  │ 検証記録     │     │ 参照解計算       │
  │ 信頼度更新   │     │ CSV蓄積         │
  └──────────────┘     └──────────────────┘
  data/hypotheses_      data/experiments.csv
  memo.json
```

## 4. PHYSBO統合 — ベイズ最適化によるパラメータ探索

### 位置づけ

ParetoAnalysisAgent内に統合。グリッドサーチの代わりにベイズ最適化でdt空間を効率的に探索する。

### 動作

各問題設定 `(alpha, ic_type)` ごとに独立してPHYSBOを実行する:

- **Feature（入力変数）**: `log10(dt)` — 1次元、80個の離散候補点（`physbo.search.discrete_multi.Policy`）
- **目的変数（2つ）**: `-L2_error` と `-wall_time`（PHYSBOは最大化するため符号を反転）
- **探索方式**: 離散多目的最適化（80候補から逐次選択）。ガウス過程で候補間の類似性を学習

1. **候補点生成**: dt ∈ [1e-5, 1e-2] に対数等間隔で80候補を生成
2. **ランダム探索フェーズ**: 5点をランダムにサンプリングして初期データを取得
3. **ベイズ最適化フェーズ**: HVPIスコアで15点を逐次的に選択・評価
4. **結果選択**: 評価済みキャッシュから安定かつ最小L2誤差の点を選択
5. **キャッシュ**: 同一設定の再評価を防止

```python
# デフォルト設定
use_physbo = True          # 自動検出（利用不可時はグリッドにフォールバック）
physbo_n_candidates = 80   # 対数等間隔dt候補数
physbo_n_random = 5        # ランダム探索回数
physbo_n_bayes = 15        # ベイズ最適化回数
physbo_score = "HVPI"      # Hypervolume Pareto Indicator
```

### Grid vs PHYSBO

| 項目 | グリッドサーチ | PHYSBO |
|------|-------------|--------|
| Feature | — | log10(dt) — 1次元 |
| 目的変数 | L2誤差, wall_time | -L2誤差, -wall_time（2目的最大化） |
| 評価回数 | dt候補数 × 全問題 | 20回/問題（5random + 15bayes） |
| 探索空間 | 固定グリッド | 対数空間80候補から選択 |
| 探索戦略 | 網羅的 | HVPIで有望領域に集中 |
| 適用条件 | 常に利用可 | physboパッケージ必要 |

## 5. 改善サイクルの全体フロー（7フェーズ・逐次実行）

各フェーズは前段の出力に依存するため、**すべて逐次実行**される。

### Phase 1: パレート解析

- ParetoAnalysisAgentがパラメータスイープを実行
- デフォルト: alpha = [0.0, 0.5, 1.0], dt = [0.001, 0.0005], nr = 61
- 各ソルバーについてParetoFrontを計算（安定点、パレート最適点）
- ソルバー横断解析: 各問題設定でのランキングと勝利回数
- 結果を `data/pareto_fronts/` にJSON保存

### Phase 2: ボトルネック検出

- BottleneckAnalysisAgentがパレートフロントを解析
- 8カテゴリのボトルネックを自動検出（安定性、精度ギャップ、速度ギャップなど）
- 深刻度（high/medium/low）と影響ソルバーを特定
- `improvement_history.json` に記録

### Phase 3: 提案生成

- ProposalGenerationAgentがボトルネックごとに1-3件の改善提案を生成
- 提案タイプ: parameter_tuning, algorithm_tweak, new_solver
- 実装スケッチ（擬似コード）を含む具体的な提案

### Phase 4: 多観点評価

- EvaluationAgentが4視点の重み付きスコアリングで各提案を評価
- 加重平均スコアに基づく推薦（approve / consider / reject）
- スコア順にソート

### Phase 5: レビュー

- **自動モード**: スコア ≥ 4.0 の提案を自動承認
- **対話モード**: ユーザーが各提案をY/n/qで判断
- ステータスを approved / rejected に更新

### Phase 6: 実装

- parameter_tuning: 実装スケッチのプレビュー表示
- algorithm_tweak / new_solver: 手動実装のガイダンス提示
- 承認された提案を implemented に更新

### Phase 7: レポート・アーカイブ

- ReportAgentが包括的なマークダウンレポートを生成
- `data/cycle_reports/cycle_NNN_YYYYMMDD_HHMMSS.md` に保存
- `improvement_history.json` のサイクル状態を更新

## 6. 設計上の特徴

### 階層的設計

- **CycleCoordinator**: 最上位の制御層。サイクル全体のオーケストレーションと状態管理
- **Agent層**: 各エージェントは独立した責務を持ち、明確な入出力インターフェースで連携
- **データ層**: JSON/CSVによる永続化。サイクルの中断・再開に対応

### モジュール性

- 各エージェントは独立にテスト・差し替え可能
- BottleneckAnalysisAgentの検出カテゴリは新規追加が容易
- EvaluationAgentの視点は重みの調整や新規視点の追加が可能
- PHYSBOの利用はオプショナル（自動フォールバック付き）

### 拡張性

- 新しいソルバーの追加: SolverBaseを継承し、SOLVERSリストに登録するだけ
- 新しいボトルネックカテゴリ: BottleneckAnalysisAgentにチェック関数を追加
- 新しい評価視点: EvaluationAgentにPerspectiveクラスを追加
- 仮説の追加: HypothesisTrackerに登録し、ExperimentAnalyzerにテスト関数を実装

### 逐次パイプラインと並列化の可能性

現在のシステムは完全に逐次実行であり、これはフェーズ間のデータ依存関係による
必然的な設計である。並列化が構造的に可能な箇所と適切な手法は以下の通り:

| 並列化ポイント | 現状 | 適切な手法 | MPIの適否 |
|-------------|------|-----------|----------|
| Phase 4: 4 Perspective間 | forループ | `concurrent.futures` | 不適（オーバーヘッド過大） |
| Phase 1: ソルバー間のPareto解析 | forループ | `ProcessPoolExecutor` | 適（大規模時） |
| Phase 1: 問題設定間のPHYSBO | forループ | `ProcessPoolExecutor` | 適（大規模時） |
| Phase間（1→2→3...） | **不可** | — | — |

**MPIが適切なケース**: Phase 1で数十～数百のソルバー×パラメータ設定の組み合わせを
分散クラスタ上で並列実行する場合。各ソルバー実行は独立しているため、
`mpi4py`で各ランクに(solver, config)ペアを割り当て可能。

**単一マシンでの並列化**: `concurrent.futures`がPythonの標準的な手法。
EvaluationAgentのPerspectiveが将来LLM呼び出し等の重い処理を伴う場合は
`ThreadPoolExecutor`（I/Oバウンド）が最適。数値計算を伴う場合は
`ProcessPoolExecutor`（CPUバウンド、GIL回避）が適切。

### 再現性

```bash
# 改善サイクル実行
python docs/analysis/method_improvement_cycle.py --cycles 3

# 仮説駆動実験
python docs/analysis/experiment_framework.py --cycles 3

# 対話モード
python docs/analysis/experiment_framework.py --interactive
```

## 7. 実行結果サマリー

### Cross-Solver Rankings（ソルバー横断ランキング）

| Rank | Solver | Avg Rank | 安定性 |
|------|--------|----------|--------|
| #1 | implicit_fdm | 1.3 | 100% |
| #2 | cell_centered_fvm | 2.0 | 100% |
| #3 | compact4_fdm | 2.7 | 100% |
| #4 | p2_fem | 3.5 | 100% |
| #5 | imex_fdm | 4.8 | 100% |
| #6 | cosine_spectral | 5.2 | 67% |
| #7 | pinn_stub | 6.5 | — |
| #8 | chebyshev_spectral | 7.0 | — |

### 検出されたボトルネック例

- **Speed gap**: fastest=2.98ms (cell_centered_fvm), slowest=2361.36ms (p2_fem) — 792倍の速度差
- **Stability**: cosine_spectralがalpha ≥ 0.5で不安定化

### 仮説検証例

- **H1** "Smaller dt improves spectral solver stability" → **Confirmed** (confidence 1.0)
  - dt=0.0001~0.0005: 安定性100%, dt=0.001: 50%, dt=0.002: 0%

## 8. スコアリング機構の詳細

### 判定方法

EvaluationAgentの4つのPerspectiveは、**キーワード文字列マッチング**でスコアを決定する。
各Perspectiveの`evaluate()`メソッドは、デフォルトスコア3.0から出発し、
`proposal.title`や`proposal.proposal_type`に特定キーワードが含まれるかを`if/else`で判定する。

### 各Perspectiveのルール

**AccuracyPerspective** (weight=1.5):

| 条件 | スコア |
|------|--------|
| title に "resolution" or "accuracy" | 4.5 |
| title に "adaptive" | 4.0 |
| type == `parameter_tuning` | 3.5 |
| （マッチなし） | 3.0 |

**SpeedPerspective** (weight=1.0):

| 条件 | スコア |
|------|--------|
| title に "optimize" or "fast" | 4.5 |
| title に "adaptive" | 2.5 |
| title に "resolution" + "increase" | 2.0 |
| （マッチなし） | 3.0 |

**StabilityPerspective** (weight=1.2):

| 条件 | スコア |
|------|--------|
| title に "stability" or "adaptive" | 5.0 |
| title に "constrain" | 4.5 |
| type == `algorithm_tweak` | 3.5 |
| （マッチなし） | 3.0 |

**ComplexityPerspective** (weight=0.8):

| 条件 | スコア |
|------|--------|
| type == `parameter_tuning` | 4.5 |
| type == `algorithm_tweak` | 2.5 |
| type == `new_solver` | 1.5 |
| implementation_sketch > 20行 | score - 1.0 |

### 総合スコアの計算

```
overall = (acc×1.5 + spd×1.0 + stb×1.2 + cpx×0.8) / (1.5 + 1.0 + 1.2 + 0.8)
```

### 制約と課題

- 提案の**内容を理解して**評価しているのではなく、タイトルのキーワード有無で機械的に決定
- キーワードが一つもマッチしなければ全視点3.0、overall=3.0で固定
- スコア定数（4.5, 2.5等）は経験的に設定されたヒューリスティック値で、根拠の説明はない
- LLMベースの評価に置き換えれば、提案内容の意味的理解に基づくスコアリングが可能になる

## 9. 発展: Advanced Multi-Agent System

### 概要

`docs/analysis/advanced_multi_agent.py` に、より「マルチエージェントらしい」プロトタイプが実装されている。
改善サイクル（improvement_agents.py）との最大の違いは、**CriticAgentによる反論機構**を持つ点である。

### アーキテクチャ

```
AdvancedCoordinator
  │
  ├── Phase 1: Initial Analysis（並列可能な独立解析）
  │     ├── StatisticsAgent  — ソルバー分布・支配率の検出
  │     └── FeatureAgent     — 決定木ベースの特徴量重要度分析
  │
  ├── Phase 2: Hypothesis Testing
  │     └── HypothesisAgent  — 3つの仮説を自動テスト
  │           ├── alpha閾値仮説（alpha > X → 常にFDM勝利？）
  │           ├── problem_stiffness仮説（剛性がソルバー選択を決定？）
  │           └── grid vs physics仮説（格子パラメータ vs 物理パラメータの重要度）
  │
  ├── Phase 3: Critical Review（★改善サイクルにない機能）
  │     └── CriticAgent — 他エージェントの主張を検証・反論
  │           ├── 支配率 > 99% → "訓練データの多様性不足では？"
  │           └── 特徴量のユニーク値 < 5 → "重要度が膨張している可能性"
  │
  └── Phase 4: Synthesis
        └── SynthesisAgent — 知見+仮説+批判を統合レポートに合成
```

### 改善サイクルとの比較

| 観点 | 改善サイクル | Advanced System |
|------|------------|-----------------|
| エージェント間の関係 | 一方向パイプライン | 批判的レビューあり |
| CriticAgent | なし | 他エージェントの主張を検証 |
| 入力データ | ソルバー実行結果 | 訓練データ (X, y) |
| 仮説テスト | ExperimentRunner（外部） | HypothesisAgent（内蔵） |
| LLM使用 | なし | なし（ルールベース模擬） |
| 目的 | ソルバー改善提案 | ソルバー選択パターンの分析 |

### 真のマルチエージェント化への展望

現在の両システムはルールベースであるが、CriticAgentのような**反論・検証パターン**は
LLMを導入した際に最も効果を発揮する。例えば:

- ProposalGenerationAgentがLLMで提案を生成
- CriticAgentがLLMで「その提案の弱点」を指摘
- 複数ラウンドの議論を経て提案を洗練

このDebateパターンは、LLMの hallucination を相互チェックで抑制する手法として
近年のマルチエージェント研究で注目されている。

### 実行結果（1512サンプル）

```bash
python docs/analysis/advanced_multi_agent.py
```

**StatisticsAgent**: implicit_fdm が 85.5% で支配的。T_center の分散がゼロ（非情報的特徴量）。

**FeatureAgent**: 最重要特徴量は `t_end` (importance=0.28)。13組の高相関ペアを検出
（例: `energy_content` と `half_max_radius` で相関0.99）。

**HypothesisAgent（3件 confirmed）**:
1. alpha > 0.5 → 常にFDM勝利 (confidence 100%, FDM wins 99.9%)
2. Spectral は problem_stiffness < 1.90 の時のみ勝利 (confidence 95%)
3. Grid params が physics params より重要 (confidence 80%, Grid:10 vs Physics:3)

**CriticAgent（1件）**: `t_end` のユニーク値が3つしかない → 重要度が膨張している可能性 (minor)

## 10. 考察: What vs Why — 欠けている物理的洞察

### 現状の問題

現在のエージェント群は **What（何が起きたか）** を自動的に検出するが、
**Why（なぜそうなるのか）** を物理的・数学的に説明する機能を持たない。

| エージェント | What（自動化済み） | Why（欠けている） |
|---|---|---|
| ParetoAnalysis | implicit_fdm が rank #1 | なぜ? → Crank-Nicolson は A-stable |
| Bottleneck | 速度差 792倍 | なぜ? → P2 FEM の行列組立コスト |
| Hypothesis | α>0.5 → FDM勝利 (99.9%) | なぜ? → 非線形 χ が stiff ODE を生成 |
| Hypothesis | Spectral が stiffness>1.9 で失敗 | なぜ? → 陽的時間進行 + CFL制限 |
| Evaluation | P001 score = 3.38 | なぜ3.38? → キーワードマッチ（内容理解ではない） |

### 必要な物理的説明

本来、以下のような因果関係の説明が自動生成されるべきである:

1. **Implicit FDM が #1 の理由**: Crank-Nicolson 法は放物型PDEに対してA-stable。
   非線形拡散率 χ を暗黙的に扱うためCFL制約がない。

2. **Spectral の不安定性の理由**: 陽的時間進行を使用しており、
   Δt ≤ C·(Δr)²/χ_max のCFL条件に従う。高α → 大きな χ → 許容 Δt が縮小。

3. **Compact4 が高α で有利な理由**: 空間4次精度 O(Δr⁴) vs 標準FDM の 2次精度 O(Δr²)。
   急峻な勾配（高α）で打ち切り誤差の差が顕在化。

4. **P2 FEM が遅い理由**: 2次要素で自由度が 2N+1。各タイムステップで
   スパース行列の組立＋求解が必要。バンド構造の利用が不十分。

### 欠けているエージェント: PhysicalInsightAgent

**ソルバーの数学的性質** × **PDEの物理的特性** → **なぜ適しているかの説明** を生成する
エージェントが存在しない。このエージェントは以下を入力として:

- ソルバーの性質: 陰的/陽的、空間精度次数、基底関数の種類
- PDEの特性: χ の剛性、勾配の急峻さ、CFL条件
- ベンチマーク結果: L2誤差、安定性率、実行時間

以下を出力すべきである:

- **因果関係の説明**: 「このソルバーがこの問題設定で優れている理由は...」
- **失敗の診断**: 「このソルバーがこの設定で不安定な原因は...」
- **改善の方向性**: 数学的根拠に基づく具体的な改善策

現在このような説明は `latex_report_agent.py` にテンプレート文として
ハードコードされているが、データから自動導出されたものではない。
LLMの統合により、ベンチマーク結果と数値解析の知識を組み合わせた
物理的洞察の自動生成が可能になると考えられる。
