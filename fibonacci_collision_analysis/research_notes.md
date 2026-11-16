# 実行例から分かること

本番設定 n=4–8、m=1–10 の大規模計算は未実行です。本番Notebookはseed確認・15テスト・計画・benchmarkまで実行済みです。以下は実行済みの小規模例に基づきます。

## 中心仮説の判定

固定されたcross pairについて、任意消失の最小消失数がHamming weight、連続消失の最小消失数がcyclic spanになるという機構は、定義から厳密に成立します。

ただし、**実際のKL訂正率差はcross collisionだけでは説明できません**。既存seed `000111 / 010101`、n=1、N=9、m=2では、任意消失36通りを全探索すると、訂正可能なのは9通り（25%）です。連続消失は9開始位置すべて訂正可能（100%）です。この設定のd_crossは4であり、2-site消失ではcross collisionは一切起きません。失敗27通りはすべて対角KL条件の不一致です。

この例は、中心仮説がcross collisionの機構として正しくても、観測された訂正率差の唯一の説明にはならないことを示します。Notebookは各設定で次の分解を検証します。

`KL failure rate = P(cross collision) + P(diagonal failure and no cross collision)`

## 既存seedのサイズ系列

| n | N | orbit 0 size | orbit 1 size | d_cross | ell_cross_min |
|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 6 | 2 | 2 | 4 |
| 1 | 9 | 9 | 3 | 4 | 5 |
| 2 | 15 | 15 | 5 | 4 | 7 |
| 4 | 39 | 39 | 13 | 4 | 15 |

n=0は実装検証用です。N=699（n=10）はgeometryのbenchmarkも実行し、d_cross=4、ell_cross_min=235でした。そこでの消失パターンの性能評価は行っていません。cross 162,867組、self合計270,979組のgeometry計算はこの環境で約0.74秒でした。これはKL・漏洩・CSV出力を含まない実測です。

## Q1–Q6への現時点の回答

1. **d_crossと任意消失訂正率**：同じ組成の7pair、n=1、N=12、m=2の全探索では、Pearson −0.575、Spearman −0.593でした。大きいd_crossが必ず良い完全KL訂正率を意味するわけではありません。7pairは共通seedを含み独立標本ではなく、一般的な相関の結論には使えません。
2. **最小spanと連続訂正率**：同じgroupでは連続訂正率が一定のため、相関係数は未定義としてNAです。「無相関」とは判定していません。
3. **任意消失に弱いpairのcross collision数**：上記N=9,m=2はcross collisionなしで訂正率が25%です。低い訂正率から多数のcross collisionがあるとは推論できません。
4. **小weight・大spanの支持**：support表・joint spectrum・閾値別割合を用意しました。固定pairのcollision条件を表す指標として有効ですが、対角KLによる失敗を含む性能差とは分けて評価が必要です。
5. **予測指標**：期待collision組数、unique support、multiplicity、実測cross発生率を比較できます。`1-P(cross)`はKL訂正率の上限です。汎化性能を検証した学習済み予測モデルは作っていません。対角KL情報を加えた候補指標を、別のseed群で評価する必要があります。
6. **新しいpairの設計**：d_cross増大、低weight support削減、期待cross組数低減、最小span維持はoff-diagonal条件を改善する候補です。同じsupportのmultiplicityだけを減らしてもcollision発生確率が減るとは限りません。最終選択には対角KLと完全KL訂正率を併用してください。

## 再現性

単一pair例はn=0,1,2,4、m=1,2,3,4,6。7pair例はn=0,1,2。全探索上限200、それを超える設定は一様復元抽出120回です。各CSVにexact/sampled、母集団数、試行数、unique pattern数、信頼区間を保存しています。windowの実行例も同梱しています。

既存コードとの照合はKL 776パターン、情報漏洩45パターンすべて一致しました。小さい系の明示的partial traceなどを含む15テストも通っています。検証記録は `artifact_validation.json`、旧研究との照合は `reference/` にあります。
