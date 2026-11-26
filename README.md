# pic2meas

画像1枚から、スカート丈・周囲長（バスト／アンダーバスト／ウエスト／ヒップ）・JIS のブラサイズ・頭身を推定し、手で補正できる研究・評価用ツール。推論はブラウザ内だけで行い、画像は送信しない。調査の経緯は `RESEARCH.md`。

## 仕組み

- **切り抜き**: SegFormer B2 clothes。透過 PNG ならアルファ値を人物の範囲に使う。
- **骨格と3D体型**: InstantHMR（medium, fit2）と MHR。
- **周囲長**: メッシュを 1cm ごとに水平に切り、腕を除いた断面の凸包の周長を取る。イラストは、断面の横幅をシルエットの幅に合わせる。
- **JIS ブラサイズ**: トップとアンダーの差で、AA 7.5cm から 2.5cm 刻み。
- **推論の実行**: Web Worker で動かす。WebGPU が使えなければ WASM に切り替わる（`?ep=wasm` で WASM を強制できる）。

## 開発

```sh
pnpm install
pnpm models     # モデルを Hugging Face から取得してローカルの R2 に入れる
pnpm dev
pnpm test
```

## デプロイ

```sh
pnpm exec wrangler login
pnpm exec wrangler r2 bucket create pic2meas-models
pnpm models --remote
pnpm run deploy   # `pnpm deploy` は pnpm の別コマンドなので run を付ける
```

モデルと onnxruntime の wasm は、静的アセットの上限 25MiB を超えるので R2 に置き、Worker が `/models/*` として返す。

## ライセンス

モデルのライセンスと帰属表示は `public/licenses.html` にある。
