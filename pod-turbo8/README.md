# H3通常版＋Turbo8：Pod一括導入

以前の専用Dockerイメージを使わず、配布元からComfyUI・モデルを取得します。
Fused Turbo4ではありません。通常版H3 INT8 + Qwen3-VL32B NVFP4/AWQ + Turbo8 LoRAです。
既存のServerlessエンドポイントやアプリは変更しません。

## 前提

- Ubuntu/Debian系、Linux x86_64のGPU Pod。Jupyter Terminalでrootとして実行。
- Python3.11〜3.13、24GB以上のNVIDIA GPU、ホストドライバがCUDA13に対応すること。
- コンテナディスク150GB推奨（初回実行時80GiB以上の空き）。モデル合計44.43GB。
- ホストRAM64GB以上推奨。GPU/ドライバは変更しません。
- ComfyUIを公開するHTTPポート8188をPod設定で用意。

## 実行

`install.sh`を保存して`bash install.sh`で実行します。JupyterのTerminalで使うコマンドです。
ノートブックのセルで実行する場合は、先頭に`!`または`%%bash`が必要です。

標準導入先は**`/opt/h3-turbo8`**（コンテナディスク）です。
`/workspace`のボリュームにモデルを置く構成ではありません。
既存ComfyUIのファイル、Python環境、Jupyter、FileBrowser設定を上書きしません。
専用venvと新規ComfyUIを作り、モデル配布元から5ファイルを取得・SHA256確認します。
ダウンロード途中の再実行はHFの途中ファイルを利用し、検証済みモデルを再利用します。

最後に、8188番で動くComfyUIがあれば**キューが空であることを確認して**停止し、
今回導入したComfyUIを同じ8188番で起動します。Supervisorの独立したComfyUIサービスは
停止できますが、Jupyter/FileBrowserと共有するサービス、不明なサービス、PID1の
ComfyUI、複数の候補がある場合は停止せず説明を表示します。
既存ComfyUIの処理が進行中なら停止しません。テンプレート固有の自動再起動で
ポートが再占有された場合も、他プロセスを強制終了しません。

Pod起動時の自動起動設定は変更しません。Pod再起動後、今回のComfyUIを再び使うには
以下の再起動コマンドを実行します。Pod自体をTerminateするとコンテナ内データは失われます。

## 導入後

RunPodのConnectから8188番HTTPサービスを開き、ComfyUIのワークフロー一覧で
**H3-Turbo8-I2V.json**を開きます。リロード後も見つからなければ、Jupyterから
`/opt/h3-turbo8/ComfyUI/user/default/workflows/H3-Turbo8-I2V.json`を保存し、
ComfyUI画面へドラッグしてください。

画像を選択し、プロンプトを入力してQueueします。
初期値：640×384、56フレーム（24fpsで約2.33秒）、8steps、固定seed42。
映像と音声を生成し、ComfyUI/output/H3-Turbo8へ保存します。
例示画像・プロンプトは中立的なマグカップのサンプルです。
サンプルは配置するだけで、自動でGPU生成を実行しません。

```bash
# ComfyUIだけを再起動
bash /opt/h3-turbo8/restart-comfyui.sh

# 起動ログ
 tail -n 100 /opt/h3-turbo8/comfyui.log

# 自動起動を避け、取得と配置だけ実行
H3_NO_START=1 bash install.sh

# 既存ComfyUIを停止せず別ポートで起動（Pod側にもそのHTTPポートを公開）
H3_PORT=8189 bash install.sh
```

## 固定構成

- ComfyUI公式ソース: `da8bd08e48deee74d9f6c3bc00aed365b00ec209`
- PyTorch2.11.0 + torchvision0.26.0、CUDA13.0 wheel、専用venv。
- Comfy-Org/MiniMax-H3: `4b53a35191437de05370da548e59aeabe2a5a09a`
- H3 INT8、Qwen3-VL32B NVFP4/AWQ、映像VAE FP16、音声VAE FP32、Turbo8 BF16 LoRA。
- すべてComfyUI標準ノード。カスタムノード不要、起動時も無効化。
- 元イメージと同じモデル・処理を使いますが、元イメージの全ライブラリを完全複製するものではありません。

## 検証範囲

シェル構文、埋め込みPython構文、16ノードの接続と設定、モデル対応、
ComfyUIのみのプロセス識別、実行中キュー/PID1/共有サービスを停止しない確認など8件通過。
実際のPodでの44GB一式ダウンロード・ComfyUI UI読み込み・GPU生成は未検証。
自動再起動はテンプレートによって対応できない場合があります。

配布元:
- https://github.com/Comfy-Org/ComfyUI
- https://huggingface.co/Comfy-Org/MiniMax-H3
