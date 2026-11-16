# fibonacci_boundary_distance

Fibonacci open chainの境界距離を調べる補助解析Notebookです。既存ringの定義を引き継ぎ、ring / openを別々に保存・描画します。

## 開き方

ZIPをフォルダごと展開し、同じフォルダ内の `fibonacci_boundary_distance.ipynb` を開きます。Notebook単体ではなく同梱Pythonファイルも必要です。

```bash
python -m pip install -r requirements.txt
python -m jupyterlab fibonacci_boundary_distance.ipynb
```

手動入力セルでseed、inflation回数、window長、境界距離、消失数を指定します。Nはinflation結果から自動計算します。最初は準備・ring sanity check・benchmarkまで実行されます。見積りを確認後 `RUN_FULL_ANALYSIS=True` にして先頭から再実行してください。

`fibonacci_boundary_distance_example.ipynb` は少数パターンで全工程を実行した例です。n=1,2,3,4、W=8、m=1,2,4、B=0,2,5,10、任意消失のexact閾値50、sampling80試行です。本番10,000試行とは区別しています。

## 重要な定義上の制約

既存recoverabilityは論理クラス内の候補列について、残存bitへの写像が単射である条件です。論理ラベル0/1を識別できるかとは別です。

依頼にある固定原点open列をそのまま使うと、候補集合は各クラス1本ずつです。この場合recoverabilityは全消失でもTrueです。既定モードの率1やB_star=0は、物理的な境界効果の有無を示しません。非自明な境界比較には、研究上意図する非周期的候補集合を `OPEN_CANDIDATE_FACTORY` に指定してください。定義欄、型・長さ検証、評価・保存機能は実装済みです。

open KLは既定で未計算（NA）です。明示的な等正振幅状態の宣言と根拠を入力した場合だけ計算可能です。ringのcyclic orbitをopenへ暗黙流用しません。任意の複素振幅は扱いません。

## 引き継いだ実装

`recoverability_kl.py` と `legacy_functions.py` は以前の成果物を変更せず同梱しています。inflationは1→10, 0→1、既定seedは000111 / 010101です。source_manifest.jsonに出典とSHA256を記録しています。

## 比較と保存

連続消失は全探索。任意消失は組合せ数によってexact / sampledを選び、同じW,mの相対パターンを全B・左右・中央・n・geometryで共有します。samplingは一様な復元抽出で、重複も試行として保持します。

左・右・中央、両端までの余白、固定Bと固定B/Nを記録します。B_starは最寄りの端が指定側である観測距離に限定し、最初の交差と、その後の観測点で持続する交差を分けます。未到達はNA。有限鎖の中央を無限系の極限とはみなしません。

results/ にdetail、summary、threshold、中央との差、左右比較、対応差、モデル情報を保存し、geometry_open/ と geometry_ring/ にも分離します。detailはCSVに全行を逐次保存し、画面用プレビューだけ件数制限します。SAVE_RESULTS=Falseなら保存しません。SAVE_FIGURESは独立です。

実行例は example_results/ と example_figures/ に分離しています。図はPNGで、geometryと固定site／比率を分けます。計算していないopen KL図は生成しません。rateのCIは点ごとのWilson区間、中心との差は対応試行からの標準誤差です。B_starの統計的信頼性を保証するものではありません。

## 検証

```bash
python -m unittest -v test_boundary_distance
```

検証記録は VALIDATION.md、実行例の範囲と結果は EXAMPLE_FINDINGS.md を参照してください。元のfinite ring QECCとopen補助モデルの結論は分けて解釈してください。
