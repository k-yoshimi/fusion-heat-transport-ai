# 実行マニュアル

## 1. セットアップ

```bash
# 依存パッケージのインストール
pip install -e ".[dev]"

# (オプション) PINNソルバーを使う場合
pip install -e ".[torch]"
```

必須依存: `numpy>=1.24`, `pytest>=7.0`

---

## 2. ベンチマークの実行

### 基本実行

```bash
python -m app.run_benchmark
```

デフォルトパラメータ: `--alpha 0.0 0.5 1.0 --nr 51 --dt 0.001 --t_end 0.1 --init gaussian`

### パラメータのカスタマイズ

```bash
# 非線形性パラメータを変える
python -m app.run_benchmark --alpha 0.0 0.5 1.0 2.0

# グリッド解像度を上げる（計算時間増加）
python -m app.run_benchmark --nr 101 --dt 0.0005

# 急峻な初期条件を使う
python -m app.run_benchmark --init sharp

# 長時間シミュレーション
python -m app.run_benchmark --t_end 0.5 --dt 0.001
```

### Makefileショートカット

```bash
make benchmark   # デフォルト実行
make test        # テスト実行
make clean       # outputs/ のクリーンアップ
```

---

## 3. リファレンス解の生成方法

### 概要

リファレンス解は **「同じImplicit FDMソルバーを、4倍の解像度で実行したもの」** です。
これにより、各ソルバーの「真の解にどれだけ近いか」を定量評価できます。

### 生成プロセスの詳細

`app/run_benchmark.py` の `compute_reference()` 関数が担当:

```
ベンチマーク用グリッド      リファレンス用グリッド
nr = 51点                    nr_fine = 4×51 - 3 = 201点
dt = 0.001                   dt_fine = 0.001 / 4 = 0.00025
dr = 1/50 = 0.02             dr_fine = 1/200 = 0.005
```

**手順:**

1. **グリッド細分化**: `nr_fine = 4 * nr - 3` で4倍の空間解像度
2. **初期条件の補間**: `np.interp` でfineグリッドに補間
3. **時間刻みの細分化**: `dt_fine = dt / 4` で4倍の時間解像度
4. **ImplicitFDM実行**: Crank-Nicolson法で高精度に解く
5. **ダウンサンプル**: fineグリッドの結果をもとのグリッド点数に戻す

### なぜこの方法か

- Crank-Nicolson法は**空間2次精度 × 時間2次精度**
- グリッドを4倍にすると誤差は約 (1/4)² = **1/16 に減少**
- つまりリファレンスは通常解の約16倍の精度
- 解析解が存在しないPDEでも使える実用的な手法（Richardson外挿の考え方）

### リファレンスを単独で生成するには

Pythonから直接呼び出せます:

```python
import numpy as np
from app.run_benchmark import compute_reference, make_initial

nr = 51
r = np.linspace(0, 1, nr)
T0 = make_initial(r, "gaussian")  # or "sharp"

# alpha=0.5 のリファレンス解を生成
T_ref = compute_reference(T0, r, dt=0.001, t_end=0.1, alpha=0.5)

print(T_ref.shape)   # (401, 51) — (時間ステップ数+1, 空間点数)
print(T_ref[-1])      # 最終時刻の温度プロファイル
```

---

## 4. 誤差評価の定義

### L2誤差（相対, 円筒座標重み付き）

```
L2 = sqrt( ∫(T - T_ref)² r dr / ∫T_ref² r dr )
```

`r` による重み付けは円筒座標の体積要素を反映。中心(r=0)付近より外側(r→1)の誤差が重視されます。

### L∞誤差（最大絶対誤差）

```
L∞ = max |T - T_ref|
```

全点での最悪ケースの誤差。

---

## 5. ソルバー選択ポリシー

```
score = L2_error + λ × wall_time
```

- `λ = 0.1`（デフォルト）: 精度重視
- `λ = 1.0`: 計算速度をより重視
- `λ = 0.0`: 純粋に精度のみで選択

`policy/select.py` の `select_best()` で変更可能。

---

## 6. 出力ファイル

ベンチマーク実行後、`outputs/` に生成:

| ファイル | 内容 |
|---------|------|
| `outputs/benchmark.csv` | 全ソルバー×全αの結果テーブル |
| `outputs/benchmark.md` | マークダウン形式のサマリー |

### CSVの列

| 列名 | 説明 |
|------|------|
| `name` | ソルバー名 |
| `alpha` | 非線形性パラメータ |
| `l2_error` | L2誤差 (vs リファレンス) |
| `linf_error` | L∞誤差 |
| `wall_time` | 実行時間 [秒] |
| `max_abs_gradient` | max\|dT/dr\| |
| `zero_crossings` | dT/dr のゼロ交差数 |
| `energy_content` | ∫T·r·dr (熱エネルギー) |
| `max_chi` / `min_chi` | 熱拡散率の最大/最小 |
| `max_laplacian` | max\|d²T/dr²\| |
| `T_center` / `T_edge` | 中心/端の温度 |

---

## 7. テスト

```bash
# 全テスト実行
python -m pytest tests/ -v

# 個別テスト
python -m pytest tests/test_features.py -v   # 特徴量抽出
python -m pytest tests/test_solvers.py -v    # ソルバー
python -m pytest tests/test_policy.py -v     # 選択ポリシー
```

17テストが全てパスすることを確認:
- `test_features.py` (8テスト): 解析的プロファイル(T=1-r²)での勾配・ラプラシアン・エネルギーの検証
- `test_solvers.py` (5テスト): 各ソルバーの基本動作・境界条件の検証
- `test_policy.py` (4テスト): 選択ロジックの正当性

---

## 8. 対象PDE

### 方程式

```
∂T/∂t = (1/r) ∂/∂r (r χ ∂T/∂r)
```

### 非線形拡散係数

```
χ(|∂T/∂r|) = 1 + α|∂T/∂r|
```

- `α = 0`: 線形拡散（解析解が存在）
- `α > 0`: 勾配が大きい領域で拡散が強まる（プラズマの異常輸送モデル）

### 境界条件

- `r = 0`: Neumann条件 `∂T/∂r = 0`（対称性）
- `r = 1`: Dirichlet条件 `T = 0`（壁温度固定）

### r=0 の特異点処理

`(1/r)∂/∂r(r χ ∂T/∂r)` は r=0 で 0/0 型不定形。
L'Hôpitalの定理を適用:

```
lim_{r→0} (1/r)∂/∂r(r χ ∂T/∂r) = 2χ ∂²T/∂r²
```

これにより r=0 でも安定に計算できます。
