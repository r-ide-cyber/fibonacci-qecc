# 検証記録

23件のunittestがすべて成功しました。

- 小さいringと明示的open候補の全消失部分集合について、独立したdense部分トレースと整数KL判定が一致。
- 異なるsupportサイズの正規化、自己衝突とクラス間衝突の区別、未定義open KLのNA保持。
- ringの周期境界をまたぐ平行移動、同じ相対パターンの共有、exact/sampleの再現性。
- windowの左右・中央座標、無効条件のskip、中央欠如、厳密不等号によるB_star、非単調な格子。
- detail全行の逐次保存と表示上限、SAVE_RESULTS=Falseの無書込み。
- geometry別CSVのseed先頭ゼロ保持、seedのk0/k1とinflation後のbit数の区別。
- 全条件がskipされた保存付き再実行は既存ファイルを変更せず例外で終了。

本番用・実行例の両Notebookは32セル（コード14セル）。IPythonで実際のセルを順番に実行し、rich outputを保存しました。nbformat検証と実行エラー0を確認済みです。実行はカーネル通信を使わないin-process IPythonです。

本番用は指定のn=7,8,9,10で準備・sanity check・benchmarkまで実行しています。本番の全pattern計算は開始していません。N=165,267,432,699をinflationから取得し、5,400件のring平行移動チェックが成功しました。

小規模実行例は9,792件のdetail評価、432行のlong summary、30枚のPNGを生成しました。geometry別CSVの合計と全体detail、summaryの分母が一致しています。open KLの未計算値は空欄を保ち、open KL図は作成していません。

ring KLの距離依存図とopenのしきい値図を目視確認しました。凡例・ラベルに欠けがなく、open単元候補の結果は境界効果の証拠でない旨を表示しています。

既存2ファイルのSHA256は元成果物と同一です。詳細はartifact_validation.jsonとsource_manifest.jsonを参照してください。
