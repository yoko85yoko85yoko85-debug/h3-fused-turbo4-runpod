"""Lazy ComfyUI runtime. Importing this file never starts GPU work or downloads."""
import base64
import copy
import io
import uuid
import warnings
import math
from collections import deque
import re
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
VERSION = 'h3-fused-turbo4-v1'
COMFY_DIR = Path(os.environ.get('COMFY_DIR', '/opt/ComfyUI'))
MODEL_ROOT = Path(os.environ.get('MODEL_ROOT', '/models'))
API = 'http://127.0.0.1:8188'
LOCK = threading.Lock()
process = None
startup_error = None
phase = 'registered'
engine_log = deque(maxlen=100)


def safe_text(value):
    text = str(value)
    for key, secret in os.environ.items():
        if any(part in key.upper() for part in ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY', 'AUTHORIZATION')) and len(secret) >= 4:
            text = text.replace(secret, '[redacted]')
    text = re.sub(r'(?:hf_|rpa_|ghp_)[A-Za-z0-9_-]+', '[redacted-token]', text)
    text = re.sub(r'(?i)(Bearer\s+)[^\s]+', r'\1[redacted]', text)
    text = re.sub(r'https?://[^\s]+', '[url omitted]', text)
    return text[-12000:]


def capture_engine(stream):
    try:
        for line in stream:
            cleaned = safe_text(line.rstrip())
            engine_log.append(cleaned)
            print('[ComfyUI] ' + cleaned, flush=True)
    finally:
        stream.close()


def diagnostic_error(error):
    return {'error_type': type(error).__name__, 'phase': phase,
            'detail': safe_text(error), 'engine_log_tail': safe_text('\n'.join(list(engine_log)[-40:])),
            'worker_version': VERSION}


def load_manifest():
    return json.loads((HERE / 'models.json').read_text())


def validate(p):
    if not isinstance(p, dict):
        raise ValueError('input must be an object')
    allowed = {'prompt', 'image_base64', 'width', 'height', 'duration_seconds', 'seed', 'diagnose'}
    if set(p) - allowed:
        raise ValueError('Unsupported input field; no extra LoRA or upscaler in this test worker')
    if p.get('diagnose') is True:
        return {'diagnose': True}
    prompt = p.get('prompt', '')
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 3000:
        raise ValueError('prompt must contain 1–3000 characters')
    out = {'prompt': prompt.strip()}
    for key, default, low, high in [('width', 512, 384, 1344), ('height', 512, 384, 1344),
                                   ('seed', 42, 0, 2147483647)]:
        value = p.get(key, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(key + ' is outside the allowed integer range')
        out[key] = value
    if out['width'] % 32 or out['height'] % 32 or out['width'] * out['height'] > 1344 * 768:
        raise ValueError('Dimensions must be multiples of 32 and area <= 1344x768')
    seconds = p.get('duration_seconds', 2)
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 1 <= seconds <= 15:
        raise ValueError('duration_seconds must be between 1 and 15')
    frames = math.ceil(seconds * 24)
    frames += (5 - frames) % 17
    out.update(duration_seconds=frames/24, num_frames=frames)
    if 'image_base64' in p:
        out['_source_png'] = normalize_source(p['image_base64'], out['width'], out['height'])
    return out


def normalize_source(encoded, width, height):
    from PIL import Image, ImageOps
    if not isinstance(encoded, str) or not 1 <= len(encoded) <= 4_000_000:
        raise ValueError('image_base64 must contain an image of at most 3 MB')
    raw = base64.b64decode(encoded, validate=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as original:
            if original.format not in ('PNG', 'JPEG', 'WEBP') or original.width * original.height > 20_000_000:
                raise ValueError('Source must be PNG/JPEG/WebP and at most 20 million pixels')
            original = ImageOps.exif_transpose(original)
            rgba = original.convert('RGBA')
            background = Image.new('RGB', rgba.size, 'black')
            background.paste(rgba, mask=rgba.getchannel('A'))
            scaled = ImageOps.contain(background, (width, height), Image.Resampling.LANCZOS)
            canvas = Image.new('RGB', (width, height), 'black')
            canvas.paste(scaled, ((width-scaled.width)//2, (height-scaled.height)//2))
    buffer = io.BytesIO()
    canvas.save(buffer, format='PNG')
    return buffer.getvalue()


def graph(p, source_filename=None):
    g = copy.deepcopy(json.loads((HERE / 'workflow_api.json').read_text()))
    g['7']['inputs'].update(prompt=p['prompt'], width=p['width'], height=p['height'], length=p['num_frames'])
    g['9']['inputs']['noise_seed'] = p['seed']
    if source_filename:
        if not re.fullmatch(r'frame_h3_[a-f0-9]{32}\.png', source_filename):
            raise ValueError('Invalid internal image name')
        g['6'] = {'class_type': 'LoadImage', 'inputs': {'image': source_filename}}
        g['7']['inputs']['first_frame'] = ['6', 0]
        suffix = source_filename.removeprefix('frame_h3_').removesuffix('.png')
    else:
        suffix = uuid.uuid4().hex
    g['16']['inputs']['filename_prefix'] = 'jobs/' + suffix + '/video'
    return g


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_models():
    from huggingface_hub import hf_hub_download
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    entries = load_manifest()['models']
    # local_dir avoids a second full copy in a separate download cache.
    needed = sum(m['bytes'] for m in entries if not (MODEL_ROOT / m['folder'] / m['remote_path']).exists())
    if shutil.disk_usage(MODEL_ROOT).free < needed + 3 * 1024**3:
        raise RuntimeError('Insufficient model disk space; no download was started')
    for m in entries:
        folder = MODEL_ROOT / m['folder']
        folder.mkdir(parents=True, exist_ok=True)
        path = Path(hf_hub_download(repo_id=m['repo'], filename=m['remote_path'],
                    revision=m['revision'], local_dir=str(folder)))
        if path.stat().st_size != m['bytes']:
            raise RuntimeError('Downloaded model has incorrect size: ' + m['filename'])
        marker = path.with_suffix(path.suffix + '.verified.json')
        fingerprint = {'sha256': m['sha256'], 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
        try:
            verified = json.loads(marker.read_text()) == fingerprint
        except (OSError, ValueError):
            verified = False
        if not verified:
            if digest(path) != m['sha256']:
                raise RuntimeError('Downloaded model checksum mismatch: ' + m['filename'])
            marker.write_text(json.dumps(fingerprint))
        # Support-model repositories use subdirectories; symlink only, never copy weights.
        target = COMFY_DIR / 'models' / m['folder'] / m['filename']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        if target.exists():
            raise RuntimeError('Model destination already exists: ' + str(target))
        target.symlink_to(path.resolve())


def call(path, payload=None, timeout=15):
    req = urllib.request.Request(API + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw.strip() else {}


def ready():
    global process, startup_error, phase
    if startup_error:
        raise RuntimeError(startup_error)
    if process is not None:
        if process.poll() is None:
            return
        # A new user request may start a fresh engine; never retry a submitted job.
        process = None
    try:
        phase = 'model_download_or_verification'
        ensure_models()
        phase = 'comfy_startup'
        process = subprocess.Popen([sys.executable, str(COMFY_DIR / 'main.py'), '--listen', '127.0.0.1',
                    '--port', '8188', '--disable-auto-launch', '--disable-metadata'], cwd=COMFY_DIR,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
        threading.Thread(target=capture_engine, args=(process.stdout,), daemon=True).start()
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('ComfyUI failed to start; see container logs')
            try:
                info = call('/object_info')
                missing = {node['class_type'] for node in graph(validate({'prompt': 'test'})).values()} | {'LoadImage'}
                missing -= set(info)
                if missing:
                    raise RuntimeError('Required nodes missing: ' + ', '.join(sorted(missing)))
                phase = 'ready'
                return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        raise RuntimeError('ComfyUI startup timed out')
    except Exception as error:
        if process is not None and process.poll() is None:
            process.terminate()
        startup_error = 'Startup failed; restart after fixing configuration: ' + str(error)
        raise


def generate(p):
    source = None
    try:
        if '_source_png' in p:
            ready()
            folder = COMFY_DIR / 'input'
            folder.mkdir(parents=True, exist_ok=True)
            source = folder / ('frame_h3_' + uuid.uuid4().hex + '.png')
            source.write_bytes(p['_source_png'])
        return execute(p, source.name if source else None)
    finally:
        if source is not None:
            source.unlink(missing_ok=True)


def execute(p, source_filename=None):
    global phase
    ready()
    phase = 'workflow_submission'
    workflow = graph(p, source_filename)
    response = call('/prompt', {'prompt': workflow})
    if response.get('node_errors') or not response.get('prompt_id'):
        raise RuntimeError('Workflow rejected: ' + json.dumps(response.get('node_errors', {})))
    phase = 'inference'
    prompt_id = response['prompt_id']
    deadline = time.monotonic() + 900
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('ComfyUI exited during generation')
            item = call('/history/' + prompt_id).get(prompt_id)
            if item:
                if item.get('status', {}).get('status_str') == 'error':
                    messages = item.get('status', {}).get('messages', [])
                    detail = next((m[1].get('exception_message', '') for m in reversed(messages) if isinstance(m, list) and len(m) > 1 and m[0] == 'execution_error' and isinstance(m[1], dict)), '')
                    raise RuntimeError('ComfyUI execution failed: ' + safe_text(detail))
                images = item.get('outputs', {}).get('16', {}).get('images', [])
                if images:
                    phase = 'output_read'
                    image = images[0]
                    base = (COMFY_DIR / 'output').resolve()
                    file = (base / image.get('subfolder', '') / image['filename']).resolve()
                    expected = (base / Path(workflow['16']['inputs']['filename_prefix']).parent).resolve()
                    if not file.is_relative_to(expected) or file.suffix.lower() != '.mp4' or file.stat().st_size > 6 * 1024**2:
                        raise RuntimeError('Invalid video output or MP4 exceeds 6 MiB; shorten duration or reduce resolution')
                    try:
                        encoded = base64.b64encode(file.read_bytes()).decode()
                    finally:
                        file.unlink(missing_ok=True)
                        try:
                            file.parent.rmdir()
                        except OSError:
                            pass
                    phase = 'output_cleanup'
                    try:
                        call('/history', {'delete': [prompt_id]})
                    except Exception as cleanup_error:
                        # A housekeeping error must not discard an already generated image.
                        print('History cleanup warning: ' + safe_text(cleanup_error), flush=True)
                    phase = 'ready'
                    return {'video_base64': encoded, 'mime_type': 'video/mp4', 'mode': 'h3_fused_i2v' if source_filename else 'h3_fused_t2v',
                            'worker_version': VERSION, 'model': 'MATLOWAI/minimax-h3-fused-turbo-int8-convrot',
                            'steps': 4, 'fps': 24, 'turbo': True, 'audio': True, 'spectrum': False, 'upscale': False,
                            **{key: p[key] for key in ('width', 'height', 'seed', 'num_frames', 'duration_seconds')}}
            time.sleep(1)
        raise TimeoutError('Generation timed out')
    except Exception:
        # Avoid leaving an unobserved GPU task running after returning an error.
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        raise
    finally:
        output_dir = (COMFY_DIR / 'output' / Path(workflow['16']['inputs']['filename_prefix']).parent).resolve()
        if output_dir.is_relative_to((COMFY_DIR / 'output' / 'jobs').resolve()):
            shutil.rmtree(output_dir, ignore_errors=True)


def diagnose():
    manifest = load_manifest()
    return {'ok': True, 'worker_version': VERSION, 'engine_running': process is not None and process.poll() is None,
            'downloads_started_by_diagnostic': False, 'planned_model_bytes': sum(m['bytes'] for m in manifest['models']),
            'model_files': [m['filename'] for m in manifest['models']], 'gpu_inference_verified': False}


if __name__ == '__main__':
    # Build-time smoke check: no torch, GPU, network or model download.
    assert '--check' in sys.argv
    assert len(load_manifest()['models']) == 5
    g = graph(validate({'prompt': 'A ceramic mug on a wooden table.'}))
    assert g['11']['inputs']['steps'] == 4
    assert g['8']['class_type'] == 'BasicGuider'
    print('BUILD_SMOKE_OK: configuration and workflow; no weights downloaded')
