from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'installer-prefix.sh').read_text()
for name in ('models.json','workflow.py','manage.py','setup.py'):
    marker='H3_EMBED_'+name.replace('.','_').upper()
    s+='\ncat > "$H3_POD_DIR/'+name+'" <<\''+marker+"'\n"+(root/name).read_text()+'\n'+marker+'\n'
s+='''
"$PYTHON" "$H3_POD_DIR/setup.py" "$H3_POD_DIR"
cat > "$H3_POD_DIR/restart-comfyui.sh" <<'RESTART'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/venv/bin/python" "$HERE/manage.py" --root "$HERE" --port "${H3_PORT:-8188}"
RESTART
chmod +x "$H3_POD_DIR/restart-comfyui.sh"
if [[ "$H3_NO_START" != 1 ]]; then
  "$H3_POD_DIR/restart-comfyui.sh"
else
  echo "Install complete. To start/restart only ComfyUI: bash $H3_POD_DIR/restart-comfyui.sh"
fi
printf '\\nOpen the Pod HTTP service on port %s. In ComfyUI Workflows, open H3-Turbo8-I2V.json.\\n' "$H3_PORT"
echo 'Models: 44.43GB; output videos: '"$H3_POD_DIR/ComfyUI/output"
echo 'ComfyUI log: '"$H3_POD_DIR/comfyui.log"
echo 'Container disk is deleted when the Pod is terminated. Jupyter/FileBrowser were not restarted.'
'''
(root/'install.sh').write_text(s)
