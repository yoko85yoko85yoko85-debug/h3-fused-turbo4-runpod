# H3 Fused Turbo — RunPod Serverless trial

## v8.1：PEP 668によるビルド失敗の修正

Branch=`runtime-v0.8.1`。生成コードとworker_versionはv8のままです。
ビルドログで依存関係の解決より前に`externally-managed-environment`で失敗していました。
Dockerイメージのメタデータを確認すると、新ベースはOSのPython/pipと/usr/localのPyTorchを使用していました。
システムPythonを上書きせず、python3-venvで/opt/h3-venvを作り、--system-site-packagesで同梱CUDA PyTorchを参照します。
アプリの依存関係はそのvenvへpython -m pipでインストール。PATHとsys.executableもこのvenvに揃えます。
ビルド中にvenv使用・Torch継承・CUDA13・SageAttentionのimportを確認します。GPUテストではありません。
モデル・Sage/CUDAの指定・生成設定はv8から変更していません。

RunPodのBranchをruntime-v0.8.1へ変更して保存。Allowed CUDA versionsは13.0以降、GPUはL4のまま。
ローカルでは仮想環境の動作確認と設定の差分確認を実施。新ベースでの完全DockerビルドはRunPodでの確認が必要です。

## v8：CUDA実行版とAttention起動設定の修正

Branch=`runtime-v0.8`、worker_version=`h3-fused-turbo4-v8-cu130-sage`。
v7の実ログに「最適化CUDA演算にはPyTorch cu130以上が必要」とあり、CUDA12.8で専用バックエンドが無効でした。
さらにColabノートブックを取得して起動コードを照合し、SageAttention・固定メモリ・コンパイラ設定の違いが確認できました。
この不一致が全速度差の原因かどうかは、修正後の実測が必要です。

- PyTorch2.11.0 / CUDA13.0 / cuDNN9の公開runtimeイメージへ変更。
- SageAttention1.0.6（Triton版）を追加し、--use-sage-attentionで起動。
- 配布Colabと同じ--disable-pinned-memoryと--disable-comfy-compilerを追加。
- --fastの一括指定やfp16_accumulationなどの追加最適化は行わない。
- Spectrum debug=trueにしてactual/forecastをログ出力。
- 出力JSONのruntime_environmentにPyTorch/CUDA/GPU/起動フラグを返す。
- CUDA13未満・CUDA GPU未認識ならモデル取得前にエラーとする。
- アプリ、モデル、標準VAE、画像の比率維持、任意2段階時の冒頭画像対策は維持。

RunPodのBranchをruntime-v0.8に変更し、AdvancedのAllowed CUDA versionsでCUDA13.0以降に対応したホストへ制限する。
GPUはL4のまま。CUDA13が選択肢にない場合、適当な古いCUDA版で再生成せず、設定画面を確認する。
Container disk80GB・Max1・Active0等は維持。
ビルド中はTorch/SageのimportとCUDAビルド版をGPUなしで検証する。GPUカーネル・VRAM・速度・画質の検証とは別。
比較は800×800・8秒・1段階・4ステップで、結果はworker_total_secondsとworkflow_secondsで分ける。

CPUテスト15件成功。Dockerビルド・GPU実生成・CUDA13ホスト上のカーネル実行・短縮率は未検証。
Colabの実際のTorch版とSageAttention版は不明なので、完全一致や3分37秒の再現とは主張しない。

## v7：記事の保存ワークフローを基準にした比較版

**配布JSONの2段階目は保存時点でbypassでした。** 記事の800×800・8秒・3分37秒を、こちらの2段階生成とそのまま比較したのは不適切でした。
記事の速度測定時に2段階目が有効だったかは不明です。詳しい差分と限界はWORKFLOW-AUDIT.md参照。

Branch=`article-v0.7`、worker_version=`h3-fused-turbo4-v7-article`。
SLAオフ、Spectrumオン（保存JSONの全widget値を一致）、標準VAEDecode。
モデル配線もUNET→Spectrum→SigmaShiftへ合わせます。
ClipProjDeviceLoader cuda:0/dynamicと投影v3.1を使用。
映像VAEも記事のリンク先Kijai版へ変更（同名でも従来のComfy-Org版とはサイズ・SHA256が違いました）。投影モデルのサイズ・SHA256・revisionも更新済み。
SpectrumパッケージとComfyUI本体の実行版まで記事と一致したわけではありません。完全再現や同じ速度を保証しません。

既存アプリでは**2段階高解像度化をオフ**、正方形・0.6MP（800×800）・8秒が記事の記載条件に対応します。
新しいデプロイを生成済みと誤認しないよう、Builds/Releaseとworker_versionを確認してください。
比較にはworker_total_secondsだけでなくworkflow_secondsを使います。モデルダウンロード時間を生成速度と混ぜないでください。
追加の有料生成はこちらからは実行していません。

CPUテスト14件と配布JSONのSpectrum全設定・Encoder loader/projection・有効なdecoderの一致を確認しました。
GPU実行・画質・速度は未検証です。任意の2段階I2Vには冒頭画像条件の再エンコード修正を維持しています。

## v6：SLA単独比較版

ブランチ `sla-v0.6`、worker_version=`h3-fused-turbo4-v6-sla`。
v5でユーザーが冒頭モザイクの解消を報告しました（768×768、56フレーム、実行179秒）。
その先頭画像の再エンコード処理を保ち、既定設定の変更はSLA有効化だけです。
Spectrum=false、fast_vae=false、sla=true。SLAのパラメーターはv3と同じ0.85/64、音声保護、参照Light、初回dense、8192未満dense、Tritonです。
過去ログでは2段階目で疎アテンションが動作しましたが、今回の画質・速度は未検証です。
出力accelerationは要求設定です。実効はContainerログのSLA calls/sparse/dense/fallbackで確認します。

RunPodのBranchをsla-v0.6へ変更・保存。アプリの変更は不要です。
比較時は前回と同じGPU・画像・プロンプト・解像度・尺・シードを使用してください。
前回スクリーンショットの条件：384×384→768×768、要求2秒（56フレーム）、seed2001309655。
アプリの「シードと生成設定」でランダムをオフにして上記シードを入力します。
初回のモデル取得・JIT等が時間に影響します。worker_total_seconds、workflow_secondsも確認します。

CPUテスト14件成功。SLA以外の配線がv5標準構成と一致することを検証しました。
GPU実生成・画質・速度改善は更新後の比較が必要です。結果が悪ければkeyframe-v0.5へ戻せます。

## v5：I2Vの先頭画像条件を最終解像度で再エンコード

ブランチ `keyframe-v0.5`、worker_version=`h3-fused-turbo4-v5-keyframe`。
アップロード動画では56フレームの冒頭2フレーム（約0.083秒）に格子状の崩れがあり、3フレーム目以降は正常でした。
原因はまだ断定できません。原因候補のひとつである潜在空間の画像条件補間を置き換えます。

従来の2段階I2Vでは、低解像度でVAEエンコードしたfirst-frame条件をnearest-exactで拡大していました。
v5では画像を最終サイズへピクセル空間で拡大し、映像VAEで再エンコードして2段階目の画像条件にします。
元のテキスト・画像意味埋め込みを共有するため、Qwenの再エンコードは追加しません。
新しいノードH3ReencodeFirstFrameは画像条件をコピーして更新し、1段階目の条件を変更しません。
ComfyUIの依存解決順により、生成前に条件を準備できます。追加VAEエンコード分の負荷はあります。

対象は画像あり・2段階生成のみ。画像なしと1段階生成は変更なし。
フレームを削除する処理や、元画像への差し替えは行いません。fps・長さ・音声も変更しません。
Spectrum・SLA・Fast VAEはv4と同じく既定オフ。明示指定による比較機能だけを残します。

CPUテスト13件成功：画像条件の配線、VAE再エンコード呼び出し、原条件の非破壊性を含みます。
DockerビルドとGPUでの冒頭フレーム改善・速度は未検証です。修正済みと断定しないでください。
RunPodのRepository configurationのBranchを`keyframe-v0.5`へ変更して保存します。
ビルドとRelease反映後、同じ入力画像・プロンプト・seedで2秒だけ比較します。
既存アプリからそのまま利用でき、結果にkeyframe_conditioning=reencoded_at_output_resolutionを返します。

## v4：画質不良の切り分け

ユーザーからv3で約5.17秒・768×768・480秒、画像がモザイク状に崩れるとの報告がありました。
原因をSpectrum/SLA/Fast VAEのどれかに特定できるログ・比較結果は未取得です。
`quality-v0.4`は3種類を**既定でオフ**に戻します。アプリはそのまま使えます。
明示的なspectrum/sla/fast_vae指定がない限り、v2と同じ標準生成・タイルVAEの配線になります。
高速化ノードは切り分け用途だけに残しています。修正版でのGPU画質・速度は未確認です。
この変更で速度が上がるとは主張していません。長い生成を再実行する前に、RequestsのJSONとContainerログで原因を調べます。
worker_version=`h3-fused-turbo4-v4-quality`。モデル準備込みworker_total_secondsと、準備後workflow_secondsは引き続き返します。

以下のv3記述は以前の構成です。v4では省略時の3フラグはfalseです。

既存のH3・FLUX・Moody Krea 2とは別の試験用Endpointです。
モデルを同梱せず、初回生成時に必要なファイルだけを取得します。

## 採用モデルと実行方式

- MATLOWAI/minimax-h3-fused-turbo-int8-convrot
- RefDeltaを融合した本体にTurbo 8-step @1.0とMystic v2 @0.7を融合したもの。
- 作者の推奨に合わせ、4ステップ、res_multistep / simple、BasicGuider（CFGなし）。
- 映像shift12 / 音声shift3。
- Qwen3-VL-4B FP8 + ClipProj v3 MLPでテキストエンコーダーを軽量化。
- 映像VAEはINT8 ConvRot、音声VAEはFP32。
- 映像の出力はVAEDecodeTiled、音声付きMP4（H.264）。
- 任意で2段階高解像度化：学習済みH3潜在2×アップスケーラー + 4ステップ再描画（denoise 0.35）。
- v3でSpectrum・SLA・H3専用Fast VAE Decodeを追加。追加LoRAはありません。

これは専用Turbo4 LoRAへの交換ではありません。Turbo8を内蔵したモデルを4ステップで実行する構成です。
内蔵Turbo/Mysticの強度は変更できません。Turbo LoRAを二重に読み込まないようにしています。

## 保存ファイルと容量

| ファイル | 概数 |
| --- | ---: |
| Fused INT8 ConvRot本体 | 20.98GB |
| Qwen3-VL-4B FP8 | 5.24GB |
| ClipProj v3 MLP | 0.50GB |
| 映像VAE INT8（v7 Kijai版） | 3.17GB |
| 音声VAE FP32 | 0.61GB |
| H3 learned latent upscaler | 0.059GB |

v7合計30,562,019,312 bytes（約30.56GB）。サイズと必要なVRAM量は別です。
models.jsonに配布元・固定コミット・サイズ・SHA256を記録しています。

ビルド中にモデルを取得しません。初回取得前に必要容量＋3GiBを確認します。
Hugging Faceのlocal_dirへ1組だけ保存し、ComfyUIへはコピーせずシンボリックリンクを使用。
入力画像と生成動画は返却後に削除。履歴削除の空HTTP200は正常として処理します。
初期設定ではGlobal Volumeを使いません。ファイルロック等のないGlobal Volumeへの直接ダウンロードは、この初期版では未検証です。

## 検証状況

- CPUテスト12件成功：入力検証、フレーム刻み、4ステップ固定、2段階配線・条件更新・音声維持、出力解像度、画像の黒余白、動画返却・後片付け、容量不足時の取得停止、秘密情報のマスク。
- 作者の固定コミットのCombinedノード入力スキーマと今回の配線を照合済み。
- v1の512×512・短いT2VはユーザーのRunPodでCompletedを確認（初回実行4分6秒）。
- v2の512→1024・短い2段階生成はユーザーのRunPodでCompletedを確認（実行7分43秒）。v3高速化のDockerビルドとGPU生成・速度・画質・ピークメモリは未確認です。
- 記事の800×800・8秒・3分37秒の再現は確認していません。設定・GPU・尺・キャッシュ状態が異なり、同じ速度とは限りません。

## RunPodへの初回デプロイ

1. Serverless → New Endpoint → Deploy from a GitHub repository。
2. このリポジトリ、Branch main、Dockerfile path /Dockerfile、Build context .、Queue方式。
3. 別名h3-fused-turbo4-testで作成。既存Endpointは変更しません。
4. 最初の候補設定：RTX A5000またはRTX4090・24GB、GPU1、Max workers1、Active workers0、Idle timeout5秒。
5. Container disk80GB、Initialization timeout900秒、Execution timeout1800秒（有効）。開始コマンド・Model欄・追加環境変数は空欄。
6. BuildsでCompleted、Releasesで最新コミットの反映を確認。

GPU・初期化・モデル取得を伴うワーカーの稼働は課金対象です。容量や性能は上記設定での実テストが必要です。

### 最初の診断（GPUモデルはロードしないが、ワーカー稼働は課金され得ます）

```json
{"input":{"diagnose":true}}
```

ok=trueは設定を読めたという意味で、GPU動画生成の成功確認ではありません。

### 最初の短い生成テスト

smoke-request.jsonをRunPodのRequestsへ入力し、1回だけ送信します（v2ではupscale=true）。
512×512、要求2秒（17k+5のフレーム刻みに切り上げるため実際は56フレーム、約2.33秒）、seed42。
v7初回はモデル約30.56GBの取得・SHA256確認・ロードが入ります。失敗しても自動再送しません。
エラー時はdetail、phase、engine_log_tailに原因を返します。

### 画像から動画にする場合

同じinputへimage_base64（PNG/JPEG/WebP、最大4,000,000文字、2000万画素まで）を追加します。
EXIF回転を反映し、出力サイズへ縦横比を保って収め、余白を黒で補います。
image_base64を省略するとT2Vです。入力した場合は必ず画像を使い、無視しません。
最終フレーム指定・参照画像モードは対象外です。
既存H3アプリはworker_versionと追加LoRA入力の契約が異なるため、まだそのまま接続しないでください。
基本生成を確認後にアプリ接続を調整します。

入力：prompt、width、height、duration_seconds、seed、image_base64（任意）、upscale（任意、既定false）、diagnose。
各辺384〜1344px・32px刻み、総画素数最大1344×768、要求長さ1〜15秒。
15秒や最大解像度での生成可能性は未確認です。まず512×512の短い生成から確認します。
動画は6MiBを上限としてbase64返却。超える場合は短くするか解像度を下げてください。

## 2段階高解像度化（v2）

`upscale: true`で有効になります。width/heightは**1段階目のサイズ**です。
1段階目は従来どおり4ステップ。完全にノイズを除いた`denoised_output`を学習済みモデルで縦横2倍にし、
H3の画像・キーフレーム条件も拡大し、denoise 0.35・4ステップで再描画します。
Combinedが再ノイズ化するため、2段階目のSamplerにはDisableNoiseを使います。
音声は再ノイズ化・再生成せず、1段階目からデコードします。フレーム数・fpsは同じです。

```json
{"input":{"prompt":"A small red toy ball rolls slowly across a wooden table in a quiet sunlit room. Soft room ambience, no dialogue, no music.","width":512,"height":512,"duration_seconds":2,"seed":42,"upscale":true}}
```

- 上記は512×512 → 1024×1024。384×384 → 768×768、480×480 → 960×960も指定できます。
- 作者の学習モデルは2倍固定です。512→800の任意倍率はこの版では行いません。
- 比率・尺は維持します。I2Vの入力画像は1段階目のサイズへ比率を保ち黒余白で収めます。
- 試験上限：ベースは32px刻み・各辺384〜768、面積512×512以下（出力面積1024×1024以下）。
- 結果width/heightは最終サイズ、base_width/base_heightは1段階目。upscale、pass2_steps、pass2_denoiseも返します。
- `upscale: false`または省略で従来の1段階生成。追加モデル約59MBも起動準備時に取得・検証します。
- 2段階目は高解像度で再描画するためVRAMと時間が増えます。15秒が必ず通る方式ではありません。
  まず512×512・2秒で確認し、長さを伸ばしてください。出力6MiB上限も引き続き適用します。
- 2段階目失敗時はエラーを返します。低解像度動画へ黙って置き換えたり、自動再送したりしません。

既存EndpointのManage → Edit endpoint → Repository configuration → Branchを`upscale-v0.2`へ変更し、
Save Endpoint。BuildsがCompletedになり、Releasesがこの構成を指すことを確認してから送信します。
worker_versionは`h3-fused-turbo4-v2-upscale`です。GPU・ディスク・タイムアウト設定は既存のまま使えます。

## 高速化（v3 / speed-v0.3）

既存アプリの入力契約を保ち、Endpoint更新後は既定で以下を有効にします。

- Spectrum：blend_weight 0.30、degree 1、warmup 1、最終actual 1、bootstrap有効。音声への直接混合は0、音声保護を目的とするoffline_smoothing_replayは記事と同じく有効。
- 各生成段階に独立したSpectrumモデルを用い、1段階目と2段階目で履歴を共有しません。
- SLA：Tritonエンジン、sparsity 0.85、block64、音声保護、参照保護Light、最初のステップはdense。追加の実験的INT8 QK/PVはオフ。
- H3専用Fast VAE Decode：タイル512px・重なり128px、出力CPU。従来の小タイルに比べタイル数を減らします。VRAM使用量は増えるためGPU検証が必要です。
- 追加のモデルファイルはありません。既存の6ファイルを使います。

Spectrumは近似であり、特にTurboの少ないステップでは構図・動き・細部が変わることがあります。
SLAは8192トークン未満や保護対象でdense処理になります。カーネル非対応時にも作者実装はdenseへフォールバックします。
ノードが存在するだけでは高速化が実際に効いた証拠にはなりません。
ContainerログのSpectrum actual/forecast数、SLA sparse/dense数・fallback警告と所要時間を確認してください。
結果のaccelerationは**要求設定**であり、実効の保証ではありません（runtime_effect_verified=false）。

比較・切り分けはRunPod Requestsのinputへ `spectrum:false`、`sla:false`、`fast_vae:false` を指定できます。
v3では省略時はすべてtrue。v4ではすべてfalse。upscaleは従来どおり任意です。1段階/2段階、画像あり/なしに適用します。
worker_total_secondsはモデル準備を含むワーカー処理時間、workflow_secondsはモデル準備後の生成・デコード・保存・応答準備時間です。

更新：Manage → Edit endpoint → Repository configuration → Branchを **speed-v0.3** に変更して保存。
Builds Completed・Releasesで反映を確認してから使用します。worker_versionは `h3-fused-turbo4-v3-speed`。
アプリ自体の更新やAPIキーの再登録は不要です。

CPU契約テスト12件と固定ソースのノード入力・VAEインターフェース照合を実施。
Dockerビルド・GPU実生成、SpectrumとSLA併用の実効・品質・短縮率はまだ未検証です。
モデル取得・GPUオフロードの時間をこれらのノードが解消するわけではありません。
まず同条件の短い動画で比較し、実効が確認できてから長い生成へ進めてください。

## 更新

GitHubのコミットだけではRunPodの更新が始まらないことがあります。
GitHub Releaseの公開が公式手順ですが、これまでのユーザー環境で反応しない例がありました。
その場合は修正版ブランチを作り、Manage → Edit endpoint → Repository configuration → Branch変更・保存で更新します。

## GPUなしの確認

```sh
python runtime.py --check
python -m unittest -q test_runtime.py
python -m py_compile runtime.py handler.py
```

## 参照元

- 元記事：https://note.com/synth_brain/n/naecc37369fde
- モデルと作者の4step設定：https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot
- ClipProj：https://github.com/nicolab28/ComfyUI-ClipProj
- 投影モデル：https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3
- VAEs：https://huggingface.co/Comfy-Org/MiniMax-H3
- Encoder：https://huggingface.co/Comfy-Org/Qwen3-VL
- 記事のCombinedノード：https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler
- アップスケーラー構造：https://github.com/mamad8c/ComfyUI-H3-Latent-Upscaler-Mamad8
- 追加モデル：https://huggingface.co/Tridae/H3LatentUpscaler

モデルの利用条件は各配布元を確認してください。モデルをGitHubへ保存したり再配布したりする構成ではありません。
