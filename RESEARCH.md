# pic2meas 実現可能性調査レポート

調査日: 2026-11-27
対象: 画像（実写・イラスト）から骨格・姿勢を推定し、身長または画像内の基準物をもとに身体寸法を算出する。あわせて服を解析し、スカート丈などを推定する。

---

## 1. 結論

**作れる。ただし「何を」「どの精度で」出すかを項目ごとに分ける必要がある。**

- **実写 × 骨格・姿勢**: 既製モデルで十分な精度が出る。課題はほぼない。
- **実写 × 長さ（身長、股下、腕の長さ、肩幅など）**: 身長の入力または基準物があれば実用レベルに届く見込み。単眼画像ではスケールが決まらないため、スケールの与え方が精度を左右する。
- **実写 × 周囲長（胸囲、胴囲、腰囲）**: 単眼1枚では誤差が大きい。数cm〜10cm超の誤差を前提にする。正面と側面の2枚撮影や体にフィットした服、LiDAR などで改善する。
- **イラスト**: 物理的な正解がない。「頭身」「部位の比率」「設定身長を与えたときの換算cm」を出す機能として設計すべき。周囲長は意味のある値にならない。
- **服の属性（種類、丈区分、袖、襟、色など）**: VLM とセグメンテーションでかなり取れる見込み。PoC での検証が必要。
- **服の寸法（スカート丈のcm値など）**: 確立した既製手法はない。「ウエスト位置から裾までの長さ」を骨格キーポイント、服のマスク、スケールから計算する自前のパイプラインになる。膝上、膝丈、ミディなどの相対区分はスケールなしでも出せる。

---

## 2. 実現可能性マトリクス

凡例: ◎ 既製で実用 / ○ 組み合わせで実用 / △ 条件付き・誤差大 / × 実質不可

| 項目 | 実写 | イラスト | 主な失敗要因 |
|---|---|---|---|
| 2D骨格・姿勢 | ◎ RTMPose/RTMW、MediaPipe | ○ 専用モデル（Bizarre Pose）か fine-tune が必要 | イラストの誇張デフォルメ、極端なアングル |
| 3D体型（メッシュ） | ○ SAM 3D Body など | × 学習分布の外。リアル体型モデルに当てはまらない | ゆったりした服、遮蔽 |
| 長さ（身長比、股下、腕） | ○ スケールがあれば実用 | ○ 比率と頭身なら可。cmは設定身長からの換算のみ | パース（遠近）、前かがみ、座り姿勢 |
| 周囲長（胸囲、胴囲、腰囲） | △ 単眼は誤差大。正面＋側面で改善 | × | 奥行きの欠落、服の膨らみ |
| 服カテゴリ・属性 | ○ VLM、DeepFashion2 系 | ○ VLM（要検証） | 重ね着、小物、画風 |
| スカート丈（相対区分） | ○ 裾と膝キーポイントの位置関係 | ○ 同左 | 座り姿勢、裾の広がり、遮蔽 |
| スカート丈（cm） | △ スケール依存。自前パイプライン | △ 設定身長からの換算 | 上記に加えてパースによる短縮 |

### 精度の参考値（条件がそれぞれ違うので横並び比較はしないこと）

- **SHAPY（CVPR 2022、単眼）**: 胸囲の平均絶対誤差は約 65 mm。
- **SMPL-X を写真1枚にフィットした修士論文プロジェクト**: 平均誤差は約 6〜12%。作業中のプロジェクトで暫定値。誤差の主因は全体スケール、奥行き、ゆったりした服。
- **Sekar 2026（弱キャリブレーション、被験者4人、Webカメラ）**
  - 胴囲誤差は、ランドマーク方式と回帰方式で約 0.3〜24.8 cm（0.13〜9.77 inch）。基準物と側面シルエットを使う方式で約 0.2〜20.5 cm（0.09〜8.08 inch）。20.5 cm は、ゆったりした服による外れ値と論文中で説明されている。
  - 脚長誤差は、基準物方式で約 2.0〜13.6 cm（0.80〜5.35 inch）。他の2方式では約 7.8〜26.0 cm（3.08〜10.24 inch）。
  - 結論として、キャリブレーションの手間を増やすほど周囲長が安定するというトレードオフがある。
- **Prism Labs のスマホ3Dスキャンアプリ**: 胴囲の MAE は 3.4 cm。多視点の3Dスキャンであり、単眼1枚の値ではない。
- **スマホ機械学習アプリの検証研究**: 胴囲と腰囲で 0.5〜1.9 cm。これは**再現性（precision）誤差**で、正確度ではない。
- **Bodygram（商用API）**: 公式ドキュメント自身が、正面と側面の2枚撮影が最も正確で、1枚（selfie）は軽量版としている。
- **衣類単体の寸法計測（MDPI 2022）**: スカートの総丈で平均誤差 1.65 cm。人が着ていない衣類画像に対する値。
- **VLM（GPT/Gemini/Claude）での体寸法推定**: 厳密な比較研究は見つからなかった。逸話レベルの報告と、「同じ画像でも結果がぶれる」という報告がある。**cm値には使わず、カテゴリ判定に使う**のが妥当。
- **SAM 3D Body（推奨スタックの中核）**: 寸法精度を評価した数値は**今回の調査では見つからなかった**。上の数値はすべて SMPL 系など別手法のもの。マトリクスの「○」は見立てであり、PoC の構成 B で検証する。

---

## 3. スケールの決め方（最重要の設計判断）

単眼画像は相似変換に対して不定なので、何らかの絶対値が必要になる。

| 方式 | 精度 | ユーザー負担 | 備考 |
|---|---|---|---|
| **身長の入力** | ○ | 低 | 最もシンプル。先行研究の多くも身長でスケールを与えている。足元から頭頂までが写っていることが前提 |
| **基準物（A4用紙、カード、ArUco マーカー）** | ○ | 中 | **体と同じ奥行き面に置くこと**が必須。床置きは不可。4隅が取れればホモグラフィで傾きを補正できる。Sekar 2026 では基準物方式で周囲長が安定した。（設計上の見立て: 検出の安定性は ArUco > A4 の輪郭検出と考えられる） |
| **メトリック深度推定（Depth Pro など）** | △ | なし | 補助用。（設計上の見立て: 人体計測に使える絶対精度は期待しにくい） |
| **LiDAR / ARKit（iPhone Pro 系）** | ○〜◎（見立て） | 低 | 実測の奥行きが得られる。対応端末が限られ、ネイティブアプリが前提 |
| **イラスト: 設定身長の入力** | 定義上の値 | 低 | 「キャラ設定 158cm」などから換算する。正解は存在しない |

**推奨**: MVP は「身長入力」を必須にし、「基準物（ArUco/A4）」はオプションにする。深度推定は姿勢の補正にだけ使う。

---

## 4. 推奨技術スタック

### 前提

- 商用利用の有無は未確定。そのため**商用可ルート**を主案にし、研究専用の選択肢は別記する。
- ライセンスは 2026-11-27 時点で GitHub の LICENSE を確認した範囲で書いている（第7章を参照）。

### アーキテクチャ（MVP）

```
[Web/モバイル UI] ──画像＋身長──▶ [API: Python FastAPI (GPU)]
                                    ├─ 人物検出 + 2D骨格: RTMPose/RTMW (rtmlib, ONNX)
                                    ├─ 人物・服のセグメンテーション: SAM 3（テキストプロンプト "skirt" 等）
                                    ├─ 3D体型: SAM 3D Body → MHR メッシュ → スライスして周囲長を算出
                                    ├─ スケール: 身長 or ArUco/A4 検出 (OpenCV)
                                    ├─ 服の属性: VLM（構造化出力でカテゴリ・丈区分）
                                    └─ 寸法計算: 自前ロジック (numpy / trimesh)
```

| レイヤ | 推奨 | 代替 / 補足 |
|---|---|---|
| フロント | Next.js（Web）。撮影ガイド（A-pose、全身が入るように）を出す | LiDAR を使うなら Swift (ARKit) |
| オンデバイスの簡易版 | MediaPipe Pose Landmarker（Web/モバイル、コードは Apache-2.0） | モデルカードの条件は別途確認する |
| API | Python + FastAPI + PyTorch / ONNX Runtime、GPU インスタンス | 重いモデルは非同期ジョブ化する |
| 2D骨格（実写） | RTMPose / RTMW（MMPose、Apache-2.0、COCO-WholeBody で 70+ mAP） | ViTPose（コードは Apache-2.0、重みのライセンスは明記なし） |
| 2D骨格（イラスト） | MMPose の Human-Art（絵画・漫画・アニメ等を含むデータセット）で学習したモデルか、RTMPose をアニメデータで fine-tune | Bizarre Pose Estimator（重みは **AGPL-3.0**）。VLM にキーポイントを出させる案は精度を PoC で検証する |
| 3D体型（実写） | **SAM 3D Body**（SAM License: 商用利用可、輸出規制などの条件付き）＋ **MHR**（Apache-2.0） | **Anny**（Apache-2.0、身長・体重・年齢から体型を生成。数値からのフィットや補完に使う） |
| セグメンテーション | **SAM 3**（SAM License）。テキストで服の部位を指定できる | 人体パーツ分割: 自前 fine-tune。イラスト: SkyTNT anime-segmentation（Apache-2.0）、See-through（ライセンス未確認） |
| 深度 | Depth Anything V2 **Small**（Apache-2.0） | Depth Pro（Apple のサンプルコードライセンス。利用・改変・再配布は可だが、特許権は明示的に許諾対象外。HF のモデルカードには研究用途限定とする表記もあり食い違うため、要法務確認）。DA-V2 の Base 以上は CC-BY-NC |
| 服の属性 | VLM（例: Claude）に JSON スキーマで「種類 / 丈区分 / 袖 / 襟 / 柄 / 素材感」を出させる | DeepFashion2 で学習した検出器（データセットの利用条件を確認する） |
| 基準物 | OpenCV の ArUco 検出 ＋ `findHomography` | A4 は輪郭と4隅の検出 |

### 避けるべきもの（商用の場合）

- **SMPL / SMPL-X**: 学術用途限定。商用には Meshcapade の有償ライセンスが必要（最小で €1,500/年〜）。SMPL のパラメータを出力する HMR モデルの多くも、メッシュ化に SMPL ファイルが必要になる。Anny でも smplx トポロジを使うと非商用になる。
- **Sapiens（初代）**: CC-BY-NC-4.0。
- **Sapiens2**: 独自ライセンスで、禁止用途に **"for biometric processing"** と監視用途が含まれる。身体寸法の推定がこれに当たると解釈される余地があるため、避けるのが無難。
- **Depth Anything V2 の Base/Large**: CC-BY-NC-4.0。

### 研究・個人利用ルート（非商用なら選択肢が広がる）

- **3D体型**: SMPL-X 系の HMR（SMPLer-X、SHAPY など）＋ SMPLify-X によるフィット。寸法抽出の先行研究が最も多い。
- **人体の統合モデル**: Sapiens（初代、CC-BY-NC）。骨格、パーツ分割、深度、法線を1系統で扱える。
- **深度**: Depth Anything V2 Large（CC-BY-NC）。
- **イラストの骨格**: Bizarre Pose の重み（AGPL-3.0）をそのまま使う。

---

## 5. 服解析パイプライン（スカート丈）

既製の「スカート丈推定モデル」は見つからなかった。以下の組み合わせで実装する。

1. **スカートのマスク**: SAM 3 にテキスト "skirt" を与えて取得する。イラストでは精度を PoC で確認する。
2. **裾のライン**: マスクの下端輪郭を取る。下端の中央値を使うか、左右端を取る。DeepFashion2 系の hemline ランドマークを使ってもよい。
3. **基準点**: 骨格から腰（hip）、膝（knee）、足首（ankle）を取る。ウエスト位置は、マスクの上端か腰キーポイントからのオフセットで決める。
4. **相対区分（スケール不要）**: 裾のY座標を膝と足首の間で正規化し、「マイクロ / ミニ / 膝上 / 膝丈 / ミディ / ロング / マキシ」に分ける。人が着用している場合は「腰から裾まで ÷ 脚長」で正規化する方式が先行研究にある。
5. **cm値**: 「ウエストから裾までのピクセル長 × スケール（cm/px）」で出す。可能なら 3D メッシュ上で再投影し、パースによる短縮を補正する。
6. **VLM でクロスチェック**: VLM に丈区分を答えさせ、幾何計算の結果と食い違ったら信頼度を下げる。

**失敗パターン**: 座り姿勢（脚が手前に向く）、強い俯瞰やあおり（短縮）、フレアやプリーツによる裾の広がり、手やバッグによる遮蔽、ロングコートとの重ね着。こうした入力は信頼度スコアを出して弾く設計にする。

**衣類単体の画像**（人が着ていない状態）の場合は別問題で、セグメンテーションとキーポイントで実寸を出す先行研究（MDPI 2022、スカート丈の誤差 1.65 cm）がそのまま参考になる。

---

## 6. イラスト特有の注意点

- 頭身やデフォルメの差が大きく、リアル体型モデル（MHR、SMPL、Anny）へのフィットは学習分布の外になる。**3D体型推定は使わず、2D の比率で計測する**。
- 出せる値は、頭身、股下比、肩幅比などの**比率**と、**設定身長を与えたときの換算cm**。
- 周囲長は画風に依存して意味をなさないので、出さない。
- 骨格推定の候補は2つ。1つは Human-Art（CVPR 2023。絵画や漫画などの人工シーンを含み、COCO だけで学習したモデルは人工シーンで精度が落ちると報告）で学習した MMPose 系モデル。もう1つは Bizarre Pose（4,000枚の COCO 互換アニメキーポイントデータセット。重みは AGPL）。Human-Art データセットの CC ライセンスの種類は未確認。商用なら自前データでの fine-tune も視野に入れる。
- パーツ分割は See-through（2026、単一画像からアニメキャラをレイヤー分解。19〜23パーツ）が使える可能性がある。ライセンスは未確認。
- 学習データとしてイラストを集める場合、著作権と利用規約の確認が必要。

---

## 7. ライセンス確認状況

| モデル | ライセンス | 商用 | 確認方法 |
|---|---|---|---|
| SAM 3D Body | SAM License | 可（ロイヤリティ無料。再配布時はライセンスの同梱が必要。リバースエンジニアリング禁止。輸出規制、ITAR、軍事用途は禁止。研究発表時は謝辞が必要） | GitHub LICENSE の 1.b 条を全文確認 ✅ |
| MHR | Apache-2.0 | 可 | GitHub API ✅ |
| SAM 3 | SAM License（SAM 3D Body と同じ条文） | 可（条件は同上） | GitHub LICENSE の 1.b 条を全文確認 ✅ |
| Sapiens2 | Sapiens2 License | 条件付き。**biometric processing と監視用途は禁止**。学習データに含まれる個人の特定も禁止 | GitHub LICENSE.md 本文 ✅ |
| Sapiens（初代） | CC-BY-NC-4.0 | 不可 | HF モデルカード |
| Depth Pro | Apple のサンプルコードライセンス（README ではコードと重みの両方に適用） | 利用・改変・再配布は可。特許権は明示的に許諾外。HF カードの表記（apple-amlr、研究用途限定）と食い違う | GitHub LICENSE 全文 ✅ / HF との不整合 ⚠️ |
| Human-Art | リポジトリは Apache-2.0、データセットは CC（種類は未確認） | データセットは要確認 | GitHub API ⚠️ |
| Depth Anything V2 | Small: Apache-2.0 / Base 以上: CC-BY-NC-4.0 | Small のみ可 | GitHub README ✅（メトリック版の重みは未明記 ⚠️） |
| RTMPose/RTMW/DWPose | Apache-2.0（MMPose） | 可。ただし学習データ（COCO-WholeBody など）の条件は別途確認 | HF ミラーと MMPose ⚠️ |
| ViTPose | コードは Apache-2.0。重みは明記なし | 重みは不明 | GitHub ⚠️ |
| MediaPipe | コードは Apache-2.0 | モデルカードは要確認 | GitHub API ✅ / モデルカード ⚠️ |
| Anny | Apache-2.0（アセットは CC0。smplx トポロジは非商用） | 可（smplx を使わない場合） | GitHub README ✅ |
| SMPL / SMPL-X | 学術用途限定 | Meshcapade の有償ライセンスが必要 | 公式サイト |
| Bizarre Pose の重み | AGPL-3.0 | 条件付き（派生物も AGPL） | HF ミラー ⚠️ |

⚠️ の行は二次情報か一部未確認なので、採用前に一次ソースを確認すること。Meshcapade が Epic Games に買収されたという情報は個人ブログのみが出典で、未確認。

---

## 8. リスク・法務

- **個人情報**: 体型データや身体画像が要配慮個人情報または個人識別符号に当たるかは、検索では結論が出なかった。個人情報保護委員会は 2026 年改正で「顔特徴データ等」や子どもの情報の規律を検討中。**未確定事項として法務確認が必要**。
- **第三者や未成年の画像**: 他人の写真をアップロードして寸法を推定できてしまう。特に未成年の体型やスカート丈を推定する用途は悪用リスクが高い。以下の対策を設計段階で入れることを推奨する。
  - 本人撮影フロー（撮影ガイドやライブ撮影）を基本にする。
  - 画像を保存しない。処理後に即時削除する。
  - オンデバイス処理の選択肢を用意する。
  - 利用規約で第三者の画像を禁止する。
  - 未成年と推定された画像の扱いを決めておく。
- **モデルのライセンス条項**: Sapiens2 の biometric 禁止条項のように、用途の制限がサービスの性質と衝突しうるものがある。

---

## 9. 次のステップ（PoC 提案）

精度は手元のデータで測るしかない。まず 1〜2 週間規模の PoC を推奨する。

1. **評価データを作る**: 実写 10〜20 人分。巻尺による実測値を付け、正面と側面の写真を撮る。服の条件（フィット / ゆったり）を分ける。イラストは 30〜50 枚で、設定身長が分かるキャラを使う。
2. **比較する構成**
   - A: RTMPose ＋ 身長スケール ＋ 2D 計算（最小構成）
   - B: SAM 3D Body ＋ MHR メッシュのスライス
   - C: B ＋ ArUco 基準物
   - 服の属性: VLM と SAM 3 ＋ 幾何計算
3. **指標**: 部位ごとの MAE（cm）、丈区分の正解率、失敗率（弾いた割合）。
4. **判断**: 周囲長で実用的な誤差（目安 ±3 cm）が出るかどうかで、「単眼1枚で行く」「正面と側面の2枚を必須にする」「長さのみ提供する」のどれにするかを決める。

---

## Sources

- [A Focused Human Body Model for Accurate Anthropometric Measurements Extraction (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_A_Focused_Human_Body_Model_for_Accurate_Anthropometric_Measurements_Extraction_CVPR_2025_paper.pdf)
- [Leveraging Anthropometric Measurements to Improve Human Mesh Estimation (CVPRW 2025)](https://arxiv.org/pdf/2409.17671)
- [SHAPY: Accurate 3D Body Shape Regression (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Choutas_Accurate_3D_Body_Shape_Regression_Using_Metric_and_Semantic_Attributes_CVPR_2022_paper.pdf)
- [smplx-llm-diagnosis (GitHub)](https://github.com/aneehashabbir/smplx-llm-diagnosis)
- [An Empirical Study of Monocular Human Body Measurement Under Weak Calibration (arXiv 2601.01639)](https://arxiv.org/html/2601.01639v1)
- [Inferring Body Measurements from 2D Images: A Comprehensive Review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12193998/)
- [Beyond BMI: 3D Smartphone Application (Prism Labs)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11606355/)
- [Evaluation of automated anthropometrics produced by smartphone-based ML](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10442791/)
- [Bodygram Platform docs](https://docs.bodygram.com/platform/first-scan) / [Pose guidelines](https://docs.bodygram.com/platform/pose-guidelines)
- [SAM 3D Body (Meta blog)](https://ai.meta.com/blog/sam-3d/) / [HF: facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3)
- [MHR (GitHub)](https://github.com/facebookresearch/MHR)
- [Anny (NAVER LABS Europe blog)](https://europe.naverlabs.com/blog/anny-a-free-to-use-3d-human-parametric-model-for-all-ages/) / [arXiv 2511.03589](https://arxiv.org/abs/2511.03589) / [GitHub naver/anny](https://github.com/naver/anny)
- [SMPL-X license](https://smpl-x.is.tue.mpg.de/modellicense.html) / [Meshcapade Body Models](https://meshcapade.com/assets/body-models)
- [A 3D body scan for nine cents without SMPL (dev.to)](https://dev.to/arkadiuss/a-3d-body-scan-for-nine-cents-without-smpl-4me1)
- [Sapiens (Meta)](https://www.meta.com/emerging-tech/codec-avatars/sapiens/) / [Sapiens2 arXiv 2604.21681](https://arxiv.org/abs/2604.21681)
- [SAM 3: Segment Anything with Concepts (arXiv 2511.16719)](https://arxiv.org/abs/2511.16719)
- [RTMW (arXiv 2407.08634)](https://arxiv.org/abs/2407.08634) / [rtmlib](https://github.com/Tau-J/rtmlib) / [MMPose releases](https://github.com/open-mmlab/mmpose/releases)
- [ViTPose (GitHub)](https://github.com/ViTAE-Transformer/ViTPose)
- [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
- [Depth Pro (GitHub)](https://github.com/apple/ml-depth-pro) / [Depth Anything V2 (GitHub)](https://github.com/DepthAnything/Depth-Anything-V2)
- [Transfer Learning for Pose Estimation of Illustrated Characters (WACV 2022)](https://arxiv.org/abs/2108.01819) / [ComfyUI-AnimePose weights (AGPL)](https://huggingface.co/mrpm/ComfyUI-AnimePose-weights)
- [Human-Art (CVPR 2023)](https://arxiv.org/abs/2303.02760) / [GitHub IDEA-Research/HumanArt](https://github.com/IDEA-Research/HumanArt)
- [See-through: Layer Decomposition for Anime Characters (arXiv 2602.03749)](https://arxiv.org/html/2602.03749)
- [SkyTNT anime-segmentation dataset](https://huggingface.co/datasets/skytnt/anime-segmentation)
- [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) / [Fashion Landmark Detection (arXiv 1608.03049)](https://arxiv.org/abs/1608.03049)
- [Detailed Garment Recovery from a Single-View Image (arXiv 1608.01250)](https://arxiv.org/pdf/1608.01250)
- [An Intelligent Solution for Automatic Garment Measurement (MDPI 2022)](https://www.mdpi.com/2076-3417/12/9/4470)
- [Federated Fine-Grained Classification of Fashion Apparel (arXiv 2008.12350)](https://arxiv.org/pdf/2008.12350)
- [We Tested Top AI Image Analysis Models for Body Measurement](https://wittysparks.com/ai-image-analysis-models-body-measurement)
- [個人情報保護委員会 資料 (2026-09)](https://www.ppc.go.jp/files/pdf/260916_shiryou-1.pdf) / [ガイドライン通則編](https://www.ppc.go.jp/personalinfo/legal/guidelines_tsusoku/)

---

## 10. 追加条件: 完全非商用・ローカル実行のみ（2026-11-27 追記）

### 前提

- 商用利用はしない。処理はすべて手元のマシンで行い、サーバーへは送らない。
- 実行環境は次の3つから1つを選ぶ。
  - PC: GPU あり
  - ブラウザ: WebGPU か WASM
  - Mac: ANE が使える

### 非商用にすると広がる選択肢

- SMPL-X 系の手法（SMPLer-X、SHAPY、SMPLify-X）。
- Sapiens 初代（CC-BY-NC）。骨格、パーツ分割、深度、法線をまとめて扱える。
- Depth Anything V2 Large。
- Bizarre Pose（AGPL）。

**例外: Sapiens2 は非商用でも使えない。** biometric processing の禁止は、商用かどうかに関係なく適用される。

### 「利用者の自己責任」で解決しないこと

ライセンスの義務は、作者がモデルを**配布する行為**そのものに発生する。免責文を書いても消えない。対策は次のとおり。

- **重みは同梱しない。** 初回起動時に、利用者のマシンが公式の配布元から直接ダウンロードする。
- **SMPL-X は再配布が禁止されている。** 利用者自身に登録してもらい、ファイルを置いてもらう。
- **AGPL のコード・重みを組み込むなら、アプリも AGPL で公開する。**
- **服の属性に使う VLM もローカルで動かす。** Claude などの API はサーバー送信にあたるので使えない。ローカルで動く VLM（Qwen 系、Gemma 系など）を使い、各モデルのライセンスを確認する。

### プラットフォーム比較

| | PC（GPU, Python） | Mac（MLX + Core ML/Core AI） | ブラウザ（WebGPU/WASM） |
|---|---|---|---|
| 3D体型（SAM 3D Body / SMPL-X 系） | ◎ 公式実装がそのまま動く（CUDA） | ○ MLX 移植版（mlx-community/sam-3d-body-dinov3-bf16）がある。品質と速度は未検証。動くのは ANE ではなく GPU（Metal） | ○ **InstantHMR**（SAM 3D Body を蒸留した 32M パラメータの ONNX。MHR メッシュを出力。第12章）なら現実的。SAM 3D Body 本体は重すぎる |
| 2D骨格 | ◎ | ○ RTMPose などを Core ML/Core AI に変換すれば ANE で動く | ○ ONNX Runtime Web、MediaPipe |
| セグメンテーション（SAM 3 / Sapiens seg） | ◎ | ○ MLX 版 SAM 3（mlx-community/sam3-image）あり | △ SAM 系の WebGPU デモはあるが、SAM 3 の移植は未確認 |
| 深度 | ◎ | ◎ Apple 公式の Core ML 版 Depth Anything V2 Small（ANE） | ○ transformers.js のデモあり（Small） |
| ローカル VLM（服の属性） | ◎ 7B 級も可 | ○ mlx-vlm | △ 2B 級が限界で、精度は低い |
| 周囲長（3D メッシュが必要） | 可 | 可（MLX 版が動けば） | 可の見込み（InstantHMR。体型の精度は未検証） |
| 配布のしやすさ | △ Python 環境と CUDA が必要 | ○ 署名付きアプリ | ◎ URL を開くだけ |

### 推奨

1. **まずは PC（GPU, Python）で作る。** 全機能が公式実装で動く。精度を検証する PoC の基準実装にもなる。
2. **Mac は2番手で、実現性は高め。** MLX 移植版がある。PoC の結果を見て、移植版が公式実装と同等の出力を出すか比べてから決める。
3. **ブラウザも有力。** 2D 版なら確実に作れる。3D も InstantHMR で狙える（第12章）。※初版でブラウザの3Dを「×」としていたのは調査漏れだったので訂正した。

---

## 11. スカート丈だけに絞った場合のブラウザ構成（2026-11-27 追記）

**ブラウザだけで作れる。** 3D体型推定はいらず、2D の骨格、スカートのマスク、スケールの3つで足りる。

| 役割 | 候補 | 実行環境 |
|---|---|---|
| 骨格（腰・膝・足首） | MediaPipe Pose Landmarker、RTMPose（onnxruntime-web） | WebGPU / WASM |
| スカートのマスク（実写） | `Xenova/segformer_b2_clothes`（18クラス。"Skirt" クラスの IoU は約 0.65） | transformers.js |
| スカートのマスク（イラスト・保険） | SlimSAM（14MB、WASM でも動く）か SAM 2.1（WebGPU）。腰と膝の間をクリックして点プロンプトで指定 | transformers.js / onnxruntime-web |
| スケール | 身長の入力 or ArUco マーカー（OpenCV.js）。イラストは設定身長 | WASM |
| 補正 UI | 骨格の点と裾のラインをドラッグで直せるようにする | 素の Canvas |

### 計算

- **丈（cm）** = (ウエスト Y − 裾 Y) [px] × (身長 cm ÷ 頭頂から足元までの px)
- **区分** = 裾 Y を膝と足首の位置で正規化（スケール不要）

### 限界

- 正面から撮った立ち姿勢が前提。座り姿勢、強いあおりや俯瞰、裾の広がりには弱い。
- イラストは骨格の自動推定が外れやすい。手動で直せる UI で補う。
- SegFormer の clothes モデルは実写で学習している。イラストでは SAM の点プロンプトのほうが安定する見込み（未検証）。

---

## 12. ブラウザでの3D体型推定（2026-11-27 追記）

**可能。** 軽量な HMR モデルを onnxruntime-web（WebGPU）で動かす。

| 候補 | 規模 | 出力 | ライセンス | 備考 |
|---|---|---|---|---|
| **InstantHMR**（mohamdev/InstantHMR） | 20〜32M パラメータ。ONNX が公式配布 | MHR メッシュ（18,439頂点）と体型パラメータ 45 次元。メッシュ化までグラフ内で行う | コードは Apache-2.0。重みは SAM License で、medium 以下は非商用 | SAM 3D Body のアノテーションで学習。速度は RTX 4070 で約 1 ms、スマホ NPU で約 5 ms。**第一候補** |
| Multi-HMR ViT-S / B | ViT-S / ViT-B（672px） | SMPL-X（`multiHMR_672_L_anny` は Anny） | CC BY-NC-SA 4.0 | SMPL-X のファイルは利用者に用意してもらう必要がある。ONNX 化は自前 |
| SAM 3D Body 本体 | DINOv3 系の大型 ViT | MHR | SAM License | GPU でも1枚に数秒かかる。ブラウザでは非現実的 |

### 構成

```
人物検出（MediaPipe など） → InstantHMR（WebGPU） → MHR メッシュ
  → 身長でスケールを合わせる → 胸・胴・腰の高さで水平に切って周囲長を出す（JS で数十行）
```

### 注意

- 公開されている精度は姿勢（PA-MPJPE）だけで、**体型や寸法の精度は未評価**。単眼1枚なので、周囲長は平均体型に寄りやすいと見込む。
- PoC では、巻尺の実測値と比べること。比較対象として PC 版の SAM 3D Body も並べる。
- イラストは学習分布の外なので、3D は実写のみを対象にする。

---

## 13. 詳細比較（完全非商用・ローカル限定、2026-11-27 追記）

ファイルサイズは 2026-11-27 時点の Hugging Face API で確認した。速度は各公式の公表値で、測定環境はそれぞれ違う。

### 13.1 3D体型モデルの比較

| | SAM 3D Body（公式） | SAM 3D Body（MLX） | **InstantHMR** | Multi-HMR |
|---|---|---|---|---|
| 規模 | ckpt 2.1GB ＋ MHR 0.7GB | 2.8GB（bf16） | 20〜32M パラメータ。ONNX 59〜129MB（fit 付き medium は 83MB）＋ `mhr-body.onnx` 12.7MB | ViT-S / B / L（672〜896px） |
| 速度 | GPU で1枚あたり数秒（Fast SAM 3D Body 論文） | 未計測 | RTX 4070（TensorRT）で約 1〜1.5 ms、Galaxy S23 の NPU で 5〜10 ms | 29 ms（S）〜126 ms（L-896）、GPU |
| 姿勢の精度 | 最高水準（基準） | 公式と同等の見込み（未検証） | 3DPW の PA-MPJPE で 39.3（medium, fit2）〜46.6 mm（nano） | 3DPW の PVE で 89.9〜102.4 mm（指標が違うので直接は比べられない） |
| 体型・寸法の精度 | 未公表 | 未公表 | 未公表（SAM 3D Body の蒸留版なので、上限は本体程度と見込む） | 未公表 |
| 出力 | MHR | MHR | MHR メッシュ（18,439頂点）、形状 45 次元、**2D/3D キーポイント 70点** | SMPL-X（Anny 版もある） |
| 実行環境 | PC（CUDA） | Mac（MLX/GPU） | **PC（CUDA/TensorRT）、Mac（`--device coreml`）、ブラウザ（ONNX なので onnxruntime-web で動く見込み）** | PC（ONNX 化は自前） |
| ライセンス | SAM License | SAM License | コード Apache-2.0、重みは SAM License の非商用 | CC BY-NC-SA 4.0。SMPL-X のファイルが別途必要 |
| 実績 | 公式実装 | なし | なし | なし |

**判断**: 3つの実行環境すべてで動く可能性があるのは InstantHMR だけ。SAM 3D Body（公式）は、PC 上で精度の基準として使う。

### 13.2 スカート丈に必要な部品の比較（ブラウザ向け）

| 役割 | 候補 | サイズ | 長所 | 短所 |
|---|---|---|---|---|
| 腰・膝・足首 | **InstantHMR の 70 キーポイント** | 3D と共用（追加なし） | 3D と同じ骨格になる。追加のダウンロードがいらない | 実写のみ。人物検出が別途必要 |
| | MediaPipe Pose（heavy） | 30.7MB | 実績が多く、人物検出も内蔵 | 33点のみ。イラストに弱い |
| スカートのマスク | **segformer_b2_clothes** | fp32 110MB / fp16 55.5MB / 量子化 28.8MB | "Skirt" クラスが最初からある。クリック不要 | 実写で学習。Skirt の IoU は 0.65 |
| | **SlimSAM-77** | fp16 で約 21MB | 軽い。WASM でも動く。イラストにも点プロンプトで使える | クリックか、骨格から自動でプロンプトを作る必要がある |
| | SAM 2.1（tiny〜large） | 145〜878MB | 境界が正確 | 重い。WebGPU が必須 |
| スケール | 身長入力 | 0 | 確実 | 全身が写っている必要がある |
| | ArUco（OpenCV.js） | 約 8MB 前後（ビルドによる） | 身長が分からなくても使える | マーカーを用意する手間 |

### 13.3 プラットフォームの総合比較

| 観点 | ブラウザ（WebGPU/WASM） | Mac（MLX + Core ML/ANE） | PC（GPU, Python） |
|---|---|---|---|
| スカート丈（2D） | ◎ | ◎ | ◎ |
| 3D体型と周囲長 | ○ InstantHMR | ○ InstantHMR（CoreML EP）／ SAM 3D Body（MLX） | ◎ SAM 3D Body（公式）＋ InstantHMR |
| イラスト | ○ 2D のみ（手直し UI） | ○ 同左 | ○ 同左＋Bizarre Pose（AGPL）が使える |
| 初回ダウンロード | 約 150〜250MB（ブラウザにキャッシュ） | 同程度。SAM 3D Body を使うなら ＋2.8GB | 数 GB |
| 対象ユーザー | 誰でも（Chrome/Edge。Safari は版によって要確認） | Mac 利用者のみ | GPU がある PC の利用者のみ |
| 配布 | URL を開くだけ。静的ホスティングでよい | 署名付きアプリ | Python 環境と CUDA |
| 開発のしやすさ | ○ JS だけで済む | △ Swift と MLX/Core ML への変換 | ◎ 公式実装がそのまま動く |
| ANE の活用 | なし（WebGPU → Metal） | ◎ | なし |
| 最大のリスク | InstantHMR の fit グラフが WebGPU EP で全演算動くか（動かない演算は WASM に落ちて遅くなる） | 移植と変換の手間。MLX 版の品質 | 一般の人には配れない |
| 「サーバーに送らない」の示しやすさ | ◎ 通信は DevTools で見える。オフラインでも動く | ◎ | ◎ |

### 13.4 推奨（更新）

1. **本体はブラウザ。** InstantHMR、segformer_b2_clothes（または SlimSAM）、身長入力、手直し UI で組む。Mac でも WebGPU（Metal）で動くので、Mac 専用版はいらない。ANE を使うメリットは、この用途では速度に効くほどではない。
2. **検証用に PC。** SAM 3D Body の公式実装を使い、同じ画像で InstantHMR と SAM 3D Body の出力（メッシュと周囲長）を比べる。
3. **最初に確かめること**: InstantHMR の fit グラフが onnxruntime-web の WebGPU で動くかどうか。これが駄目なら、fit なしのグラフ（81〜129MB）にするか、fit を JS 側で実装する。
