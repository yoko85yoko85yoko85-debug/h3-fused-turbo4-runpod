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
