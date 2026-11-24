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
