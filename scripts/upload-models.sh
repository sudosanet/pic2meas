#!/bin/sh
# Downloads the models from their official Hugging Face repos and puts them (with their licences) into R2.
# Usage: pnpm models [--local|--remote]   (default --local: the bucket that `pnpm dev` uses)
set -eu
TARGET=${1:---local}
BUCKET=pic2meas-models
ORT=$(node -p "require('./node_modules/onnxruntime-web/package.json').version")
CACHE=${PIC2MEAS_MODEL_CACHE:-$HOME/.cache/pic2meas-models}
mkdir -p "$CACHE"

fetch() { # url file
  [ -s "$CACHE/$2" ] || curl -fL --retry 3 -o "$CACHE/$2" "$1"
}
put() { # local-file r2-key
  echo "put $2"
  pnpm exec wrangler r2 object put "$BUCKET/$2" --file "$1" "$TARGET" >/dev/null
}

HF=https://huggingface.co
fetch "$HF/Xenova/segformer_b2_clothes/resolve/main/onnx/model.onnx" segformer_b2_clothes.onnx
fetch "$HF/momolesang/InstantHMR/resolve/main/instanthmr-medium-fit2.onnx" instanthmr-medium-fit2.onnx
fetch "$HF/momolesang/InstantHMR/resolve/main/mhr-body.onnx" mhr-body.onnx
fetch https://raw.githubusercontent.com/NVlabs/SegFormer/master/LICENSE LICENSE-SegFormer.txt
fetch https://raw.githubusercontent.com/Naturalpad-studio/InstantHMR/main/LICENSE-SAM LICENSE-InstantHMR-SAM.txt
fetch https://raw.githubusercontent.com/facebookresearch/MHR/main/LICENSE LICENSE-MHR.txt
(cd "$CACHE" && shasum -a 256 *.onnx)

put "$CACHE/segformer_b2_clothes.onnx" models/segformer_b2_clothes-fp32-v1.onnx
put "$CACHE/instanthmr-medium-fit2.onnx" models/instanthmr-medium-fit2-v1.1.onnx
put "$CACHE/mhr-body.onnx" models/mhr-body-v1.1.onnx
for f in LICENSE-SegFormer.txt LICENSE-InstantHMR-SAM.txt LICENSE-MHR.txt; do put "$CACHE/$f" "models/$f"; done
for f in ort-wasm-simd-threaded.asyncify.wasm ort-wasm-simd-threaded.asyncify.mjs; do
  put "node_modules/onnxruntime-web/dist/$f" "models/ort-$ORT/$f"
done
