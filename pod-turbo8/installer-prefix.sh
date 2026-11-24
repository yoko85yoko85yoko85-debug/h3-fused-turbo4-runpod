#!/usr/bin/env bash
# Original H3 INT8 + Qwen3-VL32B + Turbo8, downloaded from upstream; no custom image.
set -euo pipefail
export PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
H3_POD_DIR="${H3_POD_DIR:-/opt/h3-turbo8}"
H3_PORT="${H3_PORT:-8188}"
H3_NO_START="${H3_NO_START:-0}"
if [[ "${1:-}" == "--help" ]]; then
  echo 'Usage: bash install.sh; optional H3_POD_DIR=/opt/h3-turbo8 H3_PORT=8188 H3_NO_START=1'
  echo 'Requires Linux x86_64, root, Python3.11-3.13, NVIDIA24GB+, CUDA13-compatible driver, at least80GB free on container disk.'
  exit 0
fi
[[ $(id -u) == 0 ]] || { echo 'Run in the Pod Jupyter Terminal as root (or sudo bash this script).'; exit 1; }
command -v python3 >/dev/null || { echo 'Select a PyTorch/ComfyUI Pod template with Python3.11-3.13.'; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'NVIDIA GPU was not detected; do not use a CPU Pod.'; exit 1; }
export H3_POD_DIR H3_PORT H3_NO_START
python3 - <<'PREFLIGHT'
import os,pathlib,re,shutil,subprocess,sys,platform
if not (3,11)<=sys.version_info[:2]<=(3,13):raise SystemExit('Python3.11-3.13 is required. Existing Jupyter Python is not modified.')
if platform.system()!='Linux' or platform.machine() not in ('x86_64','amd64'):raise SystemExit('Linux x86_64 Pod is required.')
if not shutil.which('apt-get'):raise SystemExit('Use a Debian/Ubuntu based PyTorch Pod template.')
if not os.environ['H3_PORT'].isdigit() or not 1024<=int(os.environ['H3_PORT'])<=65535:raise SystemExit('H3_PORT must be1024-65535.')
text=subprocess.check_output(['nvidia-smi'],text=True)
m=re.search(r'CUDA Version:\s*(\d+)\.(\d+)',text)
if not m or int(m[1])<13:raise SystemExit('This host driver does not report CUDA13 support. Choose a GPU host/template supporting CUDA13; no host driver will be changed by this script.')
root=pathlib.Path(os.environ['H3_POD_DIR']).resolve()
if root in (pathlib.Path('/'),pathlib.Path('/opt'),pathlib.Path('/workspace'),pathlib.Path('/root')):raise SystemExit('Use a dedicated installation directory, not a system root.')
if root.exists() and any(root.iterdir()) and not (root/'.h3-turbo8-managed').exists():raise SystemExit('Installation directory already contains unmanaged files; choose a different H3_POD_DIR.')
root.mkdir(parents=True,exist_ok=True)
if shutil.disk_usage(root).free<80*1024**3 and not (root/'.h3-turbo8-managed').exists():raise SystemExit('At least80GiB free is required for first installation (150GB container disk recommended).')
(root/'.h3-turbo8-managed').touch()
print('Install location:',root,'; free GiB:',round(shutil.disk_usage(root).free/1024**3,1))
PREFLIGHT
mkdir -p "$H3_POD_DIR/tmp" "$H3_POD_DIR/cache/hf"
export H3_POD_DIR H3_PORT H3_NO_START HF_HOME="$H3_POD_DIR/cache/hf" TMPDIR="$H3_POD_DIR/tmp"
exec 9>"$H3_POD_DIR/install.lock"
flock -n 9 || { echo 'Another H3 install is already running.'; exit 1; }
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ffmpeg python3-venv build-essential libgomp1
if [[ ! -x "$H3_POD_DIR/venv/bin/python" ]]; then python3 -m venv "$H3_POD_DIR/venv"; fi
PYTHON="$H3_POD_DIR/venv/bin/python"
if [[ ! -d "$H3_POD_DIR/ComfyUI/.git" ]]; then
  git init "$H3_POD_DIR/ComfyUI"
  git -C "$H3_POD_DIR/ComfyUI" remote add origin https://github.com/Comfy-Org/ComfyUI.git
fi
COMFY_REV=da8bd08e48deee74d9f6c3bc00aed365b00ec209
if [[ $(git -C "$H3_POD_DIR/ComfyUI" rev-parse HEAD 2>/dev/null || true) != "$COMFY_REV" ]]; then
  if [[ -n $(git -C "$H3_POD_DIR/ComfyUI" status --porcelain --untracked-files=no) ]]; then echo 'ComfyUI has tracked local edits; not overwriting them.'; exit 1; fi
  git -C "$H3_POD_DIR/ComfyUI" fetch --depth 1 origin "$COMFY_REV"
  git -C "$H3_POD_DIR/ComfyUI" checkout --detach "$COMFY_REV"
fi
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0 torchvision==0.26.0
printf '%s\n' 'torch==2.11.0' 'torchvision==0.26.0' > "$H3_POD_DIR/torch-constraints.txt"
"$PYTHON" -m pip install -c "$H3_POD_DIR/torch-constraints.txt" -r "$H3_POD_DIR/ComfyUI/requirements.txt" huggingface-hub==0.36.0
"$PYTHON" - <<'CHECKGPU'
import torch
if not torch.cuda.is_available():raise SystemExit('CUDA initialization failed; models have NOT been downloaded.')
p=torch.cuda.get_device_properties(0)
if p.total_memory<22*1024**3:raise SystemExit('Use a24GB-class GPU or larger for this original H3 setup.')
print('GPU:',p.name,'VRAM GiB:',round(p.total_memory/1024**3,2),'torch:',torch.__version__)
CHECKGPU
