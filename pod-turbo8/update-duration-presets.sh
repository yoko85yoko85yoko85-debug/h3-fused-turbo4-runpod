#!/usr/bin/env bash
set -euo pipefail
ROOT="${H3_POD_DIR:-/opt/h3-turbo8}"
[[ -x "$ROOT/venv/bin/python" && -f "$ROOT/restart-comfyui.sh" ]] || { echo 'Existing H3 installation was not found.'; exit 1; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BASE='https://raw.githubusercontent.com/yoko85yoko85yoko85-debug/h3-fused-turbo4-runpod/9fe9effd82c069f3d59f66a855de3a95f1db1ed7/pod-turbo8'
for file in duration_presets.py manage.py H3-Turbo8-I2V-Aspect.json; do
  curl -fL --retry 3 "$BASE/$file" -o "$TMP/$file"
done
"$ROOT/venv/bin/python" -m py_compile "$TMP/duration_presets.py" "$TMP/manage.py"
"$ROOT/venv/bin/python" -c "import json,sys; d=json.load(open(sys.argv[1])); assert any(n['type']=='H3DurationPreset' for n in d['nodes'])" "$TMP/H3-Turbo8-I2V-Aspect.json"
BACKUP="$ROOT/backups/duration-presets-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP" "$ROOT/ComfyUI/custom_nodes/h3_duration_presets" "$ROOT/ComfyUI/user/default/workflows"
cp "$ROOT/manage.py" "$BACKUP/manage.py"
if [[ -f "$ROOT/ComfyUI/user/default/workflows/H3-Turbo8-I2V-Aspect.json" ]]; then cp "$ROOT/ComfyUI/user/default/workflows/H3-Turbo8-I2V-Aspect.json" "$BACKUP/"; fi
install -m 644 "$TMP/duration_presets.py" "$ROOT/ComfyUI/custom_nodes/h3_duration_presets/__init__.py"
install -m 644 "$TMP/manage.py" "$ROOT/manage.py"
install -m 644 "$TMP/H3-Turbo8-I2V-Aspect.json" "$ROOT/ComfyUI/user/default/workflows/H3-Turbo8-I2V-Aspect.json"
echo 'Installed duration choices: 2 / 5 / 7 / 10 / 12 / 15 seconds. Models were not downloaded.'
echo "Previous files backed up to $BACKUP"
bash "$ROOT/restart-comfyui.sh"
echo 'Reload ComfyUI, then open H3-Turbo8-I2V-Aspect.json. Save open LoRA/prompt edits before replacing your canvas.'
