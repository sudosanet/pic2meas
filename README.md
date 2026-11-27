# pic2meas

画像1枚から、スカート丈・周囲長（バスト／アンダーバスト／ウエスト／ヒップ）・JIS のブラサイズ・頭身を推定し、手で補正できる研究・評価用ツール。推論はブラウザ内だけで行い、画像は送信しない。調査の経緯は `RESEARCH.md`。

## 仕組み

- **切り抜き**: SegFormer B2 clothes。全体で人物を見つけたあと、人物まわりを拡大してもう一度かける。透過 PNG ならアルファ値を人物の範囲に使う。
- **骨格と3D体型**: InstantHMR（medium, fit2）と MHR。背骨が 10° 以上傾いていたら、画像をまっすぐに回してから体型をもう一度推定する。
- **イラスト**: 骨格の点は、絵で学習した RTMPose（Human-Art）が自信を持つところだけ差し替える。3D体型は MHR の骨格スケール（頭・胴・肩・上腕・腰幅・太もも・すね）を絵の比率に合わせ直してから測る（点を動かすと再計算）。`?noart` / `?nofit` でそれぞれ無効にできる。
- **姿勢に依存しない計測**: 周囲長と身長は、推定した体を「まっすぐ立った姿勢」に戻してから測る（MHR の姿勢パラメータを 0 にする）。胸・胴・脚の周囲長は、腕が触れないように肩を上げた姿勢で測る。スカート丈・丈の区分は、画像の縦方向ではなく背骨と脚の骨に沿った位置で測る。
- **周囲長**: 体を 1cm ごとに水平に切り、腕を除いた断面の凸包の周長を取る。画像に描く線は、その断面を姿勢付きの体（同じメッシュ）で投影したもので、幅は胴だけ。高さはバスト > アンダー > ウエスト > ヒップの順に、3cm 以上離して決める。
- **胸**: 体型モデルには胸の形がないので、DAViD の表面法線を胸のまわりで積分し、体型モデルより前に出ている量を出す。そこに左右の球冠（＋傾きの平面）を当てはめ、バストは「アンダーの断面（胸郭の広がり分を足す）＋球冠」で測る。球冠は画像上の円と「胸の出っぱり」で手直しできる。`?nobreast` で無効。
- **出力**: スカート丈と丈の区分、裾の位置（膝上・股下から・床から）、身長・座高・股下・脚や腕の長さ・肩幅、周囲長（バスト・アンダー・ウエスト・ヒップ・首・太もも・ふくらはぎ）、比率、JIS ブラサイズ、頭身。股下は、まっすぐ立たせた体の正中線の一番低い点から推定する。
- **JIS ブラサイズ**: トップとアンダーの差で、AA 7.5cm から 2.5cm 刻み。
- **推論の実行**: Web Worker で動かす。WebGPU が使えなければ WASM に切り替わる（`?ep=wasm` で WASM を強制できる）。

## 開発

```sh
pnpm install
pnpm models     # モデルを Hugging Face から取得してローカルの R2 に入れる
pnpm dev
pnpm test
```

RTMPose は mmpose から ONNX に変換して使う（`tools/convert_rtmpose.py`。手順はファイル冒頭）。DAViD は `pnpm models` が取得し、`tools/convert_david.py`（uv）で fp16 にする。

### 評価

`pnpm dev` を起動したまま実行する。`eval/data/` と `eval/vrm/` は画像の権利のため git に入れない。

```sh
pnpm eval render   # eval/vrm/*.vrm を4つの姿勢で描画し、正解寸法（Tポーズのメッシュから）と一緒に eval/data/vrm/ へ
pnpm eval          # eval/data/{vrm,profile}/<id>/{image.png,truth.json} をアプリで測り、eval/results/<日付>.md に誤差表
```

VRoid Studio で書き出すときは「服で隠れた体のメッシュを削除」を切っておく（残っていないと胴は服の表面で測る）。公式プロフィールの絵は `truth.json` に `heightCm` と分かる値（`girthCm.bust` など）だけ書く。

イラスト用に3D体型推定そのものを追加学習する案は `docs/future-anime-hmr.md`。

## デプロイ

`main` に push すると Cloudflare Workers Builds がデプロイする（ビルド `pnpm build`、デプロイ `pnpm exec wrangler deploy`）。手動なら `pnpm run deploy`（`pnpm deploy` は pnpm の別コマンド）。

モデルと onnxruntime の wasm は 25MiB を超えるので R2 バケット `pic2meas-models` に置き、Worker が `/models/*` として返す。モデルを差し替えたら `pnpm models --remote`。

## ライセンス

モデルのライセンスと帰属表示は `public/licenses.html` にある。
