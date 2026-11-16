# Fibonacci-QECC research code

Fibonacci構造を用いた有限サイズの量子誤り訂正符号について、消失訂正性能と関連する特徴量を調べるPythonコードとJupyter Notebookです。

## 内容

| フォルダ | 解析内容 |
|---|---|
| [fibonacci_recoverability_kl](fibonacci_recoverability_kl/README.md) | 古典的一意復元性とKnill–Laflamme（KL）訂正可能性の比較 |
| [fibonacci_finite_size_scaling](fibonacci_finite_size_scaling/README.md) | 系の大きさを変えたときの、連続消失・任意消失に対する訂正率の比較 |
| [fibonacci_boundary_distance](fibonacci_boundary_distance/README.md) | 消失位置の境界からの距離と訂正性能の関係を調べる補助解析 |
| [fibonacci_logical_information_leakage](fibonacci_logical_information_leakage/README.md) | 消失領域への論理情報漏洩と厳密なKL条件の比較 |
| [fibonacci_collision_analysis](fibonacci_collision_analysis/README.md) | 軌道内・軌道間のcollisionと消失訂正性能の解析 |
| [fibonacci_sequence_features](fibonacci_sequence_features/README.md) | 配列の統計的特徴と訂正性能・情報漏洩・collisionの関係の解析 |
| [fibonacci_arbitrary_erasure_code_design](fibonacci_arbitrary_erasure_code_design/README.md) | Fibonacci置換規則を固定したseed pairの探索と任意消失性能の評価 |

各テーマは独立したフォルダにまとめています。定義、計算条件、結果の解釈と制約は各フォルダのREADMEを参照してください。

## 最初に見るNotebook

- 基本となる比較: [fibonacci_recoverability_kl.ipynb](fibonacci_recoverability_kl/fibonacci_recoverability_kl.ipynb)
- 符号設計の実行例: [fibonacci_arbitrary_erasure_code_design_demo.ipynb](fibonacci_arbitrary_erasure_code_design/fibonacci_arbitrary_erasure_code_design_demo.ipynb)

各テーマの `*_example.ipynb` と符号設計の `*_demo.ipynb` には既存の実行結果を残しています。符号設計の通常版Notebookには自分で設定する項目があるため、まずdemoをご覧ください。

## 再実行

リポジトリを取得して、実行したいテーマのフォルダへ移動します。例えば、基本となる比較の場合は次の順です。

```bash
cd fibonacci_recoverability_kl
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyterlab
```

そのフォルダ内のNotebookを開き、同じPython環境のカーネルで実行してください。Notebookだけを別の場所に移すと、同梱モジュールや参照データを読み込めなくなります。

## 収録範囲

コード、Notebook、説明書、実行に必要な参照データを収録した公開用の軽量版です。計算で再生成できる `results/`、`example_results/`、`figures/`、`demo_run/` 等の結果ファイルとキャッシュは省いています。各READMEで説明している結果ファイルは、必要に応じてNotebookから再生成してください。研究コードの計算ロジックと同梱の参照データは元の成果物から変更していません。

`source_sha256.json` は今回収録したファイルのSHA256です。`source_manifest.json` や `reference/` の来歴情報には作成時の参照元のパスが含まれますが、そのパスにあるファイルを実行時に要求するものではありません。

## 解釈に関する補足

- 有限サイズ・指定された条件での数値解析です。実験結果だけから一般の場合の定理を主張するものではありません。
- 標本抽出を行う解析では、未検出と不存在を区別します。
- 境界距離解析のopen候補集合と周期境界の解析は別の設定です。既定のopen候補集合の制約などは、そのテーマのREADMEに記載しています。

実装と整理には生成AIを使用しています。既存関数や参照データの来歴は各テーマ内の説明・来歴ファイルを参照してください。
