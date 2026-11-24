#!/usr/bin/env bash
# Original H3 INT8 + Qwen3-VL32B + Turbo8, downloaded from upstream; no custom image.
set -euo pipefail
# Template CUDA13.2 pip constraints apply to its own venv, not this isolated one.
# Only this script's environment is changed; Jupyter and the template are untouched.
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_EXTRA_INDEX_URL PIP_INDEX_URL PYTHONPATH
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

cat > "$H3_POD_DIR/models.json" <<'H3_EMBED_MODELS_JSON'
{
  "repository": "Comfy-Org/MiniMax-H3",
  "revision": "4b53a35191437de05370da548e59aeabe2a5a09a",
  "files": [
    {
      "path": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
      "size": 20970379616,
      "sha256": "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
    },
    {
      "path": "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
      "size": 1956193000,
      "sha256": "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e"
    },
    {
      "path": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
      "size": 15687142551,
      "sha256": "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"
    },
    {
      "path": "vae/minimax_h3_audio_vae_fp32.safetensors",
      "size": 605254808,
      "sha256": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"
    },
    {
      "path": "vae/minimax_h3_video_vae_fp16.safetensors",
      "size": 5207808496,
      "sha256": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"
    }
  ]
}

H3_EMBED_MODELS_JSON

cat > "$H3_POD_DIR/workflow.py" <<'H3_EMBED_WORKFLOW_PY'
"""Build an editable UI graph and matching API graph using only core nodes."""
import json
from pathlib import Path

MODEL='minimax_h3_fl2va_pruned_int8_convrot.safetensors'
LORA='minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors'
CLIP='qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
VAE='minimax_h3_video_vae_fp16.safetensors'
AUDIO='minimax_h3_audio_vae_fp32.safetensors'

def build():
    prompt='A ceramic mug on a wooden table. The camera moves slowly closer. Soft room ambience, no dialogue, no music.'
    specs=[
     (1,'UNETLoader',{'unet_name':MODEL,'weight_dtype':'default'},['MODEL'],[MODEL,'default'],[0,0]),
     (2,'LoraLoaderModelOnly',{'model':[1,0],'lora_name':LORA,'strength_model':1.0},['MODEL'],[LORA,1.0],[360,0]),
     (3,'CLIPLoader',{'clip_name':CLIP,'type':'minimax','device':'default'},['CLIP'],[CLIP,'minimax','default'],[0,220]),
     (4,'VAELoader',{'vae_name':VAE},['VAE'],[VAE],[0,430]),
     (5,'VAELoader',{'vae_name':AUDIO},['VAE'],[AUDIO],[0,560]),
     (6,'LoadImage',{'image':'h3-example.png'},['IMAGE','MASK'],['h3-example.png','image'],[0,720]),
     (7,'MiniMaxH3ImageToVideo',{'clip':[3,0],'vae':[4,0],'prompt':prompt,'width':640,'height':384,'length':56,'first_frame':[6,0]},['CONDITIONING','LATENT'],[prompt,640,384,56],[740,240]),
     (8,'BasicGuider',{'model':[2,0],'conditioning':[7,0]},['GUIDER'],[],[1150,0]),
     (9,'RandomNoise',{'noise_seed':42},['NOISE'],[42,'fixed'],[740,0]),
     (10,'KSamplerSelect',{'sampler_name':'res_multistep'},['SAMPLER'],['res_multistep'],[1150,160]),
     (11,'BasicScheduler',{'model':[2,0],'scheduler':'simple','steps':8,'denoise':1.0},['SIGMAS'],['simple',8,1.0],[1150,290]),
     (12,'SamplerCustomAdvanced',{'noise':[9,0],'guider':[8,0],'sampler':[10,0],'sigmas':[11,0],'latent_image':[7,1]},['LATENT','LATENT'],[],[1500,180]),
     (13,'VAEDecode',{'samples':[12,0],'vae':[4,0]},['IMAGE'],[],[1840,180]),
     (14,'VAEDecodeAudio',{'samples':[12,0],'vae':[5,0]},['AUDIO'],[],[1840,360]),
     (15,'CreateVideo',{'images':[13,0],'audio':[14,0],'fps':24.0},['VIDEO'],[24.0,'auto','sRGB','auto','none'],[2190,180]),
     (16,'SaveVideo',{'video':[15,0],'filename_prefix':'H3-Turbo8/video','format':'auto','codec':'auto'},['VIDEO'],['H3-Turbo8/video','auto','auto','auto'],[2540,180]),
    ]
    api={str(i):{'class_type':t,'inputs':inputs} for i,t,inputs,_,_,_ in specs}
    nodes=[];byid={};links=[]
    for order,(i,t,inputs,outputs,widgets,pos) in enumerate(specs):
        n={'id':i,'type':t,'pos':pos,'size':[330,350 if i in (6,7,16) else 160],
          'flags':{},'order':order,'mode':0,'inputs':[],
          'outputs':[{'name':x,'type':x,'links':[],'slot_index':j} for j,x in enumerate(outputs)],
          'properties':{'Node name for S&R':t},'widgets_values':widgets}
        byid[i]=n;nodes.append(n)
    for i,_,inputs,_,_,_ in specs:
        for name,v in inputs.items():
            if not isinstance(v,list):continue
            upstream,slot=v;typ=byid[upstream]['outputs'][slot]['type'];lid=len(links)+1
            targetslot=len(byid[i]['inputs']);links.append([lid,upstream,slot,i,targetslot,typ])
            byid[i]['inputs'].append({'name':name,'type':typ,'link':lid})
            byid[upstream]['outputs'][slot]['links'].append(lid)
    return {'last_node_id':16,'last_link_id':len(links),'nodes':nodes,'links':links,'groups':[],
       'config':{},'extra':{'ds':{'scale':0.45,'offset':[30,100]}},'version':0.4},api

def write(root):
    root=Path(root);ui,api=build()
    dest=root/'ComfyUI/user/default/workflows';dest.mkdir(parents=True,exist_ok=True)
    (dest/'H3-Turbo8-I2V.json').write_text(json.dumps(ui,ensure_ascii=False,indent=2))
    (root/'H3-Turbo8-I2V-API.json').write_text(json.dumps(api,ensure_ascii=False,indent=2))

if __name__=='__main__':
    import sys
    write(sys.argv[1])

H3_EMBED_WORKFLOW_PY

cat > "$H3_POD_DIR/manage.py" <<'H3_EMBED_MANAGE_PY'
"""Restart only an idle ComfyUI on the requested port, preserving other services."""
import argparse,json,os,signal,socket,subprocess,time,urllib.request,urllib.error
from pathlib import Path


def process_table(proc=Path('/proc')):
    result={}
    for p in proc.iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0');args=[a.decode(errors='replace') for a in args if a]
            cwd=os.readlink(p/'cwd');ppid=int(next(x for x in (p/'status').read_text().splitlines() if x.startswith('PPid:')).split()[1])
            result[int(p.name)]=(args,cwd,ppid)
        except (OSError,ValueError,StopIteration):pass
    return result


def is_comfy(args,cwd):
    if not args:return False
    return any(Path(a).name=='main.py' for a in args[1:]) and ('comfy' in cwd.lower() or any('comfy' in a.lower() for a in args))


def port_of(args):
    try:
        if '--port' in args:return int(args[args.index('--port')+1])
        return int(next(a.split('=',1)[1] for a in args if a.startswith('--port='))) if any(a.startswith('--port=') for a in args) else 8188
    except (ValueError,IndexError):return None


def ancestors(pid,table):
    result=set()
    while pid in table and pid not in result:
        result.add(pid);pid=table[pid][2]
    return result


def check_idle(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/queue',timeout=5) as r:d=json.load(r)
    except Exception:raise RuntimeError('Cannot confirm existing ComfyUI is idle; it was NOT stopped. Check its queue and retry.')
    if not isinstance(d.get('queue_running'),list) or not isinstance(d.get('queue_pending'),list):raise RuntimeError('Unrecognized ComfyUI queue response; no service was stopped.')
    if d.get('queue_running') or d.get('queue_pending'):
        raise RuntimeError('ComfyUI has running/queued work; it was NOT stopped. Finish it before retrying.')


def is_runpod_template():
    start=Path('/start.sh')
    return Path('/opt/comfyui-auth-proxy/auth_proxy.py').is_file() and start.is_file() and 'COMFYUI_INTERNAL_PORT=18188' in start.read_text()


def connection_config(public_port):
    if public_port==8188 and is_runpod_template():return 18188,'127.0.0.1',True
    return public_port,'0.0.0.0',False


def check_proxy(public_port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{public_port}/',timeout=5) as r:pass
    except urllib.error.HTTPError as e:
        if e.code==401:return
        raise RuntimeError('Template password proxy is not ready; no service was stopped.')
    except Exception:
        raise RuntimeError('Template password proxy is unavailable. Set COMFYUI_PASSWORD in Pod settings and let template startup complete. No service was stopped.')
    raise RuntimeError('Expected the template password proxy on public8188. Refusing to bypass it.')


def wait_for_port(listen,port,seconds=15):
    # Match aiohttp's listener semantics. Without SO_REUSEADDR, TIME_WAIT
    # connections from the old server can look like a live port owner.
    deadline=time.monotonic()+seconds
    while True:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:
                probe.bind((listen,port))
                return
            except OSError:
                if time.monotonic()>=deadline:
                    raise RuntimeError('A live listener still occupies port '+str(port)+'. No unrelated service was stopped. Run: ss -ltnp | grep '+str(port)) from None
        time.sleep(1)


def restart(root,port):
    public_port=port;port,listen,preserve_proxy=connection_config(public_port)
    if preserve_proxy:check_proxy(public_port)
    table=process_table();pids=[pid for pid,(args,cwd,_) in table.items() if is_comfy(args,cwd) and port_of(args)==port]
    if len(pids)>1:raise RuntimeError('Multiple ComfyUI processes on the selected port; no service was stopped.')
    if pids:
        pid=pids[0]
        if pid==1:raise RuntimeError('ComfyUI is container PID1; it was not stopped to avoid stopping the Pod. Use H3_PORT=8189 and expose that port instead.')
        check_idle(port);owned=ancestors(pid,table);supervised=False
        try:
            r=subprocess.run(['supervisorctl','status'],capture_output=True,text=True,timeout=15)
            for line in r.stdout.splitlines():
                words=line.split()
                if len(words)>3 and words[1]=='RUNNING' and words[2]=='pid':
                    parent=int(words[3].rstrip(','))
                    if parent in owned:
                        if 'comfy' not in words[0].lower() or any(parent in ancestors(child,table) and any('jupyter' in arg.lower() or 'filebrowser' in arg.lower() for arg in entry[0]) for child,entry in table.items()):
                            raise RuntimeError('ComfyUI is controlled by a shared/unknown service; no service was stopped.')
                        subprocess.run(['supervisorctl','stop',words[0]],check=True,timeout=60);supervised=True;break
        except FileNotFoundError:pass
        if not supervised:os.kill(pid,signal.SIGTERM)
        for _ in range(30):
            try:os.kill(pid,0)
            except ProcessLookupError:break
            time.sleep(1)
    # Refuse to kill a respawned process or an unrelated listener.
    wait_for_port(listen,port)
    env=os.environ.copy();env.update(HF_HOME=str(root/'cache/hf'),PYTHONUNBUFFERED='1')
    command=[str(root/'venv/bin/python'),str(root/'ComfyUI/main.py'),'--listen',listen,'--port',str(port),
             '--disable-all-custom-nodes','--disable-auto-launch']
    log=root/'comfyui.log'
    with log.open('ab',buffering=0) as f:
        process=subprocess.Popen(command,cwd=root/'ComfyUI',env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (root/'comfyui.pid').write_text(str(process.pid))
    for _ in range(180):
        if process.poll() is not None:raise RuntimeError('ComfyUI exited. See '+str(log))
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/object_info',timeout=3) as r:info=json.load(r)
            graph=json.loads((root/'H3-Turbo8-I2V-API.json').read_text())
            missing={n['class_type'] for n in graph.values()}-set(info)
            if missing:raise RuntimeError('Missing core nodes: '+', '.join(sorted(missing)))
            for n in graph.values():
                schema=info[n['class_type']]['input']
                for key,value in n['inputs'].items():
                    if isinstance(value,list):continue
                    spec=schema.get('required',{}).get(key) or schema.get('optional',{}).get(key)
                    if spec and isinstance(spec[0],list) and value not in spec[0]:
                        raise RuntimeError('Unavailable model/option: '+str(value))
            print('READY: ComfyUI on public port '+str(public_port)+'; H3 core nodes and selected model filenames verified. GPU generation remains untested on this Pod.',flush=True)
            return
        except (urllib.error.URLError,TimeoutError,ConnectionError):time.sleep(1)
    raise RuntimeError('ComfyUI did not become ready within 180s; see '+str(log))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=8188);a=p.parse_args()
    if not 1024<=a.port<=65535:raise SystemExit('Invalid port')
    restart(a.root,a.port)

H3_EMBED_MANAGE_PY

cat > "$H3_POD_DIR/setup.py" <<'H3_EMBED_SETUP_PY'
"""Download verified original H3 Turbo8 assets; no image extraction or generation."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from huggingface_hub import hf_hub_download
from workflow import write
ROOT=Path(sys.argv[1]);COMFY=ROOT/'ComfyUI'
MANIFEST=json.loads((ROOT/'models.json').read_text())

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def matches(path,spec):
    if not path.is_file() or path.stat().st_size!=spec['size']:return False
    marker=ROOT/'verified'/ (spec['sha256']+'.json')
    fingerprint={'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
    if marker.exists() and json.loads(marker.read_text())==fingerprint:return True
    if digest(path)!=spec['sha256']:return False
    marker.parent.mkdir(parents=True,exist_ok=True);marker.write_text(json.dumps(fingerprint));return True


def main():
    missing=sum(spec['size'] for spec in MANIFEST['files'] if not (COMFY/'models'/spec['path']).is_file())
    if shutil.disk_usage(ROOT).free < missing+8*1024**3:raise RuntimeError('Insufficient container disk space for remaining model downloads plus8GiB headroom.')
    for spec in MANIFEST['files']:
        path=COMFY/'models'/spec['path']
        if matches(path,spec):print('Reuse verified: '+spec['path'],flush=True);continue
        print('Downloading '+spec['path']+' ('+str(round(spec['size']/1e9,2))+' GB)',flush=True)
        downloaded=Path(hf_hub_download(repo_id=MANIFEST['repository'],filename=spec['path'],revision=MANIFEST['revision'],local_dir=COMFY/'models',force_download=path.exists()))
        if not matches(downloaded,spec):raise RuntimeError('Model integrity mismatch: '+spec['path'])
        print('SHA256 OK: '+spec['path'],flush=True)
    write(ROOT)
    from PIL import Image,ImageDraw
    example=COMFY/'input/h3-example.png';example.parent.mkdir(parents=True,exist_ok=True)
    if not example.exists():
        im=Image.new('RGB',(640,384),(190,204,210));d=ImageDraw.Draw(im);d.rectangle((0,250,640,384),fill=(144,105,72));d.rounded_rectangle((255,120,370,275),radius=20,fill=(231,223,202));d.ellipse((348,150,408,235),outline=(231,223,202),width=13);im.save(example)
    print('DOWNLOAD_COMPLETE: models and UI/API workflows installed. No GPU generation has been submitted.',flush=True)

if __name__=='__main__':main()

H3_EMBED_SETUP_PY

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
printf '\nOpen the Pod HTTP service on port %s. In ComfyUI Workflows, open H3-Turbo8-I2V.json.\n' "$H3_PORT"
echo 'Models: 44.43GB; output videos: '"$H3_POD_DIR/ComfyUI/output"
echo 'ComfyUI log: '"$H3_POD_DIR/comfyui.log"
echo 'Container disk is deleted when the Pod is terminated. Jupyter/FileBrowser were not restarted.'
