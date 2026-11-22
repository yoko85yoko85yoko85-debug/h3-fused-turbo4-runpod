# H3 Fused Turbo — RunPod Serverless trial

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
- Spectrum・SLA・2段階アップスケール・追加LoRAはこの初期試験版には入れていません。
  本体と4ステップ生成を先に確認してから、速度や画質を個別に比較します。

これは専用Turbo4 LoRAへの交換ではありません。Turbo8を内蔵したモデルを4ステップで実行する構成です。
内蔵Turbo/Mysticの強度は変更できません。Turbo LoRAを二重に読み込まないようにしています。

## 保存ファイルと容量

| ファイル | 概数 |
| --- | ---: |
| Fused INT8 ConvRot本体 | 20.98GB |
| Qwen3-VL-4B FP8 | 5.24GB |
| ClipProj v3 MLP | 0.50GB |
| 映像VAE INT8 | 2.81GB |
| 音声VAE FP32 | 0.61GB |

合計30,142,401,304 bytes（約30.1GB）。サイズと必要なVRAM量は別です。
models.jsonに配布元・固定コミット・サイズ・SHA256を記録しています。

ビルド中にモデルを取得しません。初回取得前に必要容量＋3GiBを確認します。
Hugging Faceのlocal_dirへ1組だけ保存し、ComfyUIへはコピーせずシンボリックリンクを使用。
入力画像と生成動画は返却後に削除。履歴削除の空HTTP200は正常として処理します。
初期設定ではGlobal Volumeを使いません。ファイルロック等のないGlobal Volumeへの直接ダウンロードは、この初期版では未検証です。

## 検証状況

- CPUテスト9件成功：入力検証、フレーム刻み、4ステップ固定、ワークフロー配線、画像の黒余白、動画返却・後片付け、容量不足時の取得停止、秘密情報のマスク。
- GPUモデルのロード、実際の動画生成・音質・画質・速度・ピークメモリ、Dockerビルドは未確認です。
- 記事の800×800・8秒・3分37秒の再現は確認していません。本構成はSpectrumを外しているため同じ速度とは限りません。

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

smoke-request.jsonをRunPodのRequestsへ入力し、1回だけ送信します。
512×512、要求2秒（17k+5のフレーム刻みに切り上げるため実際は56フレーム、約2.33秒）、seed42。
初回はモデル約30.1GBの取得・SHA256確認・ロードが入ります。失敗しても自動再送しません。
エラー時はdetail、phase、engine_log_tailに原因を返します。

### 画像から動画にする場合

同じinputへimage_base64（PNG/JPEG/WebP、最大4,000,000文字、2000万画素まで）を追加します。
EXIF回転を反映し、出力サイズへ縦横比を保って収め、余白を黒で補います。
image_base64を省略するとT2Vです。入力した場合は必ず画像を使い、無視しません。
最終フレーム指定・参照画像モード・アップスケールは初期版の対象外です。
既存H3アプリはworker_versionと追加LoRA入力の契約が異なるため、まだそのまま接続しないでください。
基本生成を確認後にアプリ接続を調整します。

入力：prompt、width、height、duration_seconds、seed、image_base64（任意）、diagnose。
各辺384〜1344px・32px刻み、総画素数最大1344×768、要求長さ1〜15秒。
15秒や最大解像度での生成可能性は未確認です。まず512×512の短い生成から確認します。
動画は6MiBを上限としてbase64返却。超える場合は短くするか解像度を下げてください。

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

モデルの利用条件は各配布元を確認してください。モデルをGitHubへ保存したり再配布したりする構成ではありません。
