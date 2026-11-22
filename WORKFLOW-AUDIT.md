# 記事の配布ワークフローとの照合

対象記事：https://note.com/synth_brain/n/naecc37369fde
対象配布JSON：DriveファイルID 1NdsgcYZOpglZtV0skt5rtxyeOAMRzYTv（WF-1 FL2VA）。
記事閲覧・JSON照合：2026-11-22。
配布ファイルの保存状態と、記事の速度測定時の状態は同一とは断定できません。

| 項目 | 配布JSON保存状態 | v6までの構成 | article-v0.7 |
| --- | --- | --- | --- |
| 出力 | node226、1段階目 | アプリ初期値は2段階 | 比較時はupscale=false |
| 2段階目 | nodes193/194/198/199/200/201/202/213/223はmode4（bypass） | 有効 | 任意で維持、比較時は無効 |
| SLA | ノードなし | v6は有効 | 既定オフ |
| Spectrum | node192、有効、blend0.3 | v6は無効 | 保存JSONの全widget値を照合して一致 |
| MODEL配線 | UNET→Spectrum→SigmaShift→Guider | v3はSLA→SigmaShift→Spectrum | 配布JSONと同じ順序 |
| 映像出力 | node208 VAEDecode、有効 | VAEDecodeTiled、v3のみFast版 | 標準VAEDecode |
| Fast VAE | node201、256/64、出力GPU、bypass | v3で512/128、出力CPUを有効化 | 既定オフ |
| Encoder loader | ClipProjDeviceLoader、cuda:0、dynamic | 標準CLIPLoader | 配布と同一のloader設定 |
| 投影モデル | mmh3-4b-ClipProj-v3.1-mlp | v3-mlp | v3.1、配布ファイルSHA256固定 |
| 映像VAE | Kijai版3,171,670,912 bytes、SHA256 9bb2d96f… | Comfy-Org版2,811,065,184 bytes、SHA256 52a2c8c7… | 記事リンク先Kijai版へ変更、revision/hash固定 |
| 本体・音声VAE・4B TE | 同名モデル | 同名モデル | 維持 |
| サンプラー | res_multistep/simple/4、shift12/3 | 同じ | 維持 |
| 保存JSONの例 | 0.2MP/8秒、first_frame未接続、T2V | 384²→768²/2秒、I2V等 | 比較用例800²/8秒、T2V |

## 速度の読み方

記事本文にはL4、800×800・8秒、7分13秒→3分37秒とあるが、その数値に使ったAPIグラフ・入力画像・プロンプト・seed・ComfyUIログ・計測範囲は本文から確定できない。
配布JSONでは2段階目が無効なので、3分37秒が2段階生成の時間だと解釈した過去の案内には根拠が不足していた。
保存例の0.2MPは800×800でもない。保存例を速度計測時の完全な再現データとして扱うこともできない。

現在のServerless実行時間にはモデル取得・検証・起動も含まれる。
比較にはモデル準備後のworkflow_secondsを用い、GPU/画像有無/解像度/尺を揃える。
記事のColabとRunPodではCPU・RAM・PCIe・ライブラリ・キャッシュ・GPU共有状態が違う可能性がある。
同じL4でも同じ秒数を保証するものではないが、GPU変更の前に処理経路を合わせる必要がある。

## 再現できていない点

- 配布JSONのSpectrum revisionは5161f0457bc8c52535212d6783eee73f439e1537、こちらは現在のComfyUI互換性を確認した7df883bb2361b7884ad789e97effd1dc37c8963aを維持。
- ComfyUI本体も記事のランタイム全体・起動フラグ・Python依存関係とは一致を確認できていない。
- SelectCLIPDeviceのdefaultパスは省略。出力はMP4/H264/CRF23で、記事のautoエンコード設定と一致を保証しない。
- 任意の2段階I2Vではv5の先頭画像再エンコード対策を維持。記事にはない修正。
- article-v0.7のDockerビルド・GPU生成・画質・速度は未検証。CPUテスト14件と元JSONの全Spectrum widget・Encoder loader・projection・有効decoderの照合は成功。

## アプリで比較する設定

Branchをarticle-v0.7へ変更、BuildsとReleasesを確認。
2段階高解像度化オフ、正方形、0.6MP（800×800）、8秒、ランダムシードオフ。
GPUはL4のまま。画像入力の有無とプロンプトは比較目的に合わせて同じものを使う。
article-compare-request.jsonは画像なしT2Vの短いプロンプト例であり、記事の動画の厳密な再現プロンプトではない。
旧版へ戻す場合はsla-v0.6またはkeyframe-v0.5の既存ブランチを使う。
