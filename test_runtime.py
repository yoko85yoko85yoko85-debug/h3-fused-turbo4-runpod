"""CPU-only contracts; never start a GPU, download weights or launch ComfyUI."""
import base64
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
import handler
import runtime


class Contracts(unittest.TestCase):
    def test_frames_dimensions_and_fixed_sampling(self):
        for seconds in (1,2,3,5,8,15):
            p=runtime.validate({'prompt':'A red toy ball rolls slowly.','duration_seconds':seconds})
            self.assertGreaterEqual(p['num_frames']/24,seconds)
            self.assertEqual(p['num_frames']%17,5)
            g=runtime.graph(p)
            self.assertEqual(g['11']['inputs']['steps'],4)
            self.assertEqual(g['10']['inputs']['sampler_name'],'res_multistep')
            self.assertEqual(g['8']['class_type'],'BasicGuider')
            self.assertEqual(g['3']['class_type'],'ClipProjApply')
            self.assertEqual(g['2']['inputs']['type'],'krea2')
            self.assertNotIn('LoraLoaderModelOnly',[v['class_type'] for v in g.values()])
            self.assertEqual(g['13']['class_type'],'MiniMaxH3VAEDecodeFast')
            self.assertEqual(g['7']['inputs']['length'],p['num_frames'])
            self.assertNotIn('first_frame',g['7']['inputs'])
            self.assertEqual(g['16']['inputs']['format.codec'],'h264')
            for node in g.values():
                for value in node['inputs'].values():
                    if isinstance(value,list):self.assertIn(value[0],g)

    def test_i2v_full_source_and_graph(self):
        image=Image.new('RGB',(100,200),'red');buf=io.BytesIO();image.save(buf,format='PNG')
        p=runtime.validate({'prompt':'Slow natural motion.','image_base64':base64.b64encode(buf.getvalue()).decode()})
        normalized=Image.open(io.BytesIO(p['_source_png']))
        self.assertEqual(normalized.size,(512,512))
        self.assertEqual(normalized.getpixel((0,256)),(0,0,0))
        self.assertEqual(normalized.getpixel((256,256)),(255,0,0))
        g=runtime.graph(p,'frame_h3_'+'a'*32+'.png')
        self.assertEqual(g['7']['inputs']['first_frame'],['6',0])
        self.assertEqual(g['6']['class_type'],'LoadImage')
        with self.assertRaises(ValueError):runtime.graph(p,'../escape.png')

    def test_two_pass_latent_audio_and_conditioning(self):
        p=runtime.validate({'prompt':'A toy ball rolls.', 'upscale':True})
        self.assertEqual((p['output_width'],p['output_height']),(1024,1024))
        g=runtime.graph(p,'frame_h3_'+'b'*32+'.png')
        self.assertEqual(g['19']['inputs']['samples'],['12',1])
        self.assertEqual(g['19']['inputs']['audio_denoise'],0)
        self.assertEqual(g['19']['inputs']['method'],'learned model')
        self.assertEqual(g['19']['inputs']['positive'],['7',0])
        self.assertEqual(g['20']['inputs']['conditioning'],['19',1])
        self.assertEqual(g['22']['inputs']['noise'],['21',0])
        self.assertEqual(g['21']['class_type'],'DisableNoise')
        self.assertEqual(g['22']['inputs']['sigmas'],g['19']['inputs']['sigmas'])
        self.assertEqual(g['18']['inputs']['denoise'],0.35)
        self.assertEqual(g['18']['inputs']['steps'],4)
        self.assertEqual(g['13']['inputs']['samples'],['22',0])
        self.assertEqual(g['14']['inputs']['samples'],['12',1])
        self.assertEqual(g['7']['inputs']['length'],p['num_frames'])
        self.assertEqual(g['7']['inputs']['first_frame'],['6',0])
        self.assertNotIn('22',runtime.graph(runtime.validate({'prompt':'x','upscale':False})))
        for node in g.values():
            for value in node['inputs'].values():
                if isinstance(value,list):self.assertIn(value[0],g)

    def test_acceleration_chain_and_opt_out(self):
        p=runtime.validate({'prompt':'A toy ball.', 'upscale':True})
        g=runtime.graph(p)
        self.assertEqual(g['17']['inputs']['model'],['23',0])
        self.assertEqual(g['8']['inputs']['model'],['24',0])
        self.assertEqual(g['20']['inputs']['model'],['25',0])
        self.assertEqual(g['24']['inputs']['model'],['17',0])
        self.assertEqual(g['25']['inputs']['model'],['17',0])
        self.assertEqual(g['23']['inputs']['engine'],'triton')
        self.assertTrue(g['23']['inputs']['protect_audio'])
        self.assertEqual(g['24']['inputs']['blend_weight'],.3)
        self.assertTrue(g['24']['inputs']['offline_smoothing_replay'])
        self.assertEqual(g['13']['inputs']['tile_overlap'],128)
        native=runtime.graph(runtime.validate({'prompt':'x','spectrum':False,'sla':False,'fast_vae':False}))
        self.assertEqual(native['13']['class_type'],'VAEDecodeTiled')
        self.assertEqual(native['8']['inputs']['model'],['17',0])
        self.assertEqual(native['17']['inputs']['model'],['1',0])
        self.assertNotIn('24',native)
        for flag in ('spectrum','sla','fast_vae'):
            with self.assertRaises(ValueError):runtime.validate({'prompt':'x',flag:'true'})

    def test_upscale_geometry_limits(self):
        p=runtime.validate({'prompt':'x','upscale':True,'width':640,'height':384})
        self.assertEqual((p['output_width'],p['output_height']),(1280,768))
        for values in ({'upscale':'true'},{'upscale':1},{'upscale':True,'width':768,'height':512}):
            with self.assertRaises(ValueError):runtime.validate({'prompt':'x',**values})

    def test_reject_invalid_without_gpu(self):
        with patch.object(runtime,'generate') as generate:
            for p in [None,{}, {'prompt':'x','width':641},{'prompt':'x','duration_seconds':16},
                      {'prompt':'x','duration_seconds':float('nan')},{'prompt':'x','seed':True},
                      {'prompt':'x','image_base64':'bad!'},{'prompt':'x','steps':8},
                      {'prompt':'x','lora':{}},{'prompt':'x','upscale':'yes'}]:
                self.assertIn('error',handler.handler({'input':p}))
            generate.assert_not_called()

    def test_diagnostic_and_pinned_assets(self):
        with patch.object(runtime,'ensure_models',side_effect=AssertionError),patch.object(runtime,'ready',side_effect=AssertionError):
            result=handler.handler({'input':{'diagnose':True}})
        self.assertTrue(result['ok']);self.assertFalse(result['downloads_started_by_diagnostic'])
        self.assertEqual(result['planned_model_bytes'],30201424152)
        self.assertEqual(len(runtime.load_manifest()['models']),6)
        for m in runtime.load_manifest()['models']:
            self.assertEqual(len(m['revision']),40);self.assertEqual(len(m['sha256']),64)

    def test_empty_history_response_is_success(self):
        with patch.object(runtime.urllib.request,'urlopen',return_value=io.BytesIO(b'')):
            self.assertEqual(runtime.call('/history',{'delete':['id']}),{})

    def test_video_result_cleanup(self):
        p=runtime.validate({'prompt':'A toy ball rolls.','upscale':True})
        # Mocked output bytes exercise transport/cleanup only, not codec validity.
        data=b'\x00\x00\x00\x18ftypmp42'
        self.assertEqual(data[4:8],b'ftyp')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);captured={}
            def call(path,payload=None,**kw):
                if path=='/prompt':
                    captured['g']=payload['prompt'];prefix=captured['g']['16']['inputs']['filename_prefix']
                    file=root/'output'/(prefix+'_00001_.mp4');file.parent.mkdir(parents=True);file.write_bytes(data)
                    captured['file']=file
                    return {'prompt_id':'job-1'}
                if path=='/history/job-1':
                    f=captured['file'];subfolder=str(f.parent.relative_to(root/'output'))
                    return {'job-1':{'status':{'status_str':'success'},'outputs':{'16':{'images':[{'filename':f.name,'subfolder':subfolder}]}}}}
                # A cleanup failure must not discard the generated video.
                raise RuntimeError('simulated housekeeping error')
            with patch.object(runtime,'COMFY_DIR',root),patch.object(runtime,'ready'),patch.object(runtime,'call',side_effect=call),patch.object(runtime,'process',SimpleNamespace(poll=lambda:None)):
                result=runtime.generate(p)
            self.assertEqual(base64.b64decode(result['video_base64']),data)
            self.assertEqual(result['steps'],4);self.assertTrue(result['audio']);self.assertTrue(result['spectrum'])
            self.assertTrue(result['upscale']);self.assertEqual(result['pass2_steps'],4)
            self.assertEqual((result['width'],result['height']),(1024,1024))
            self.assertEqual((result['base_width'],result['base_height']),(512,512))
            self.assertFalse(captured['file'].exists());self.assertFalse(captured['file'].parent.exists())

    def test_private_source_removed_on_error(self):
        p=runtime.validate({'prompt':'x'});p['_source_png']=b'source-test'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def fail(params,filename):
                self.assertEqual((root/'input'/filename).read_bytes(),b'source-test')
                raise RuntimeError('simulated sampler failure')
            with patch.object(runtime,'COMFY_DIR',root),patch.object(runtime,'ready'),patch.object(runtime,'execute',side_effect=fail):
                with self.assertRaises(RuntimeError):runtime.generate(p)
            self.assertEqual(list((root/'input').iterdir()),[])

    def test_insufficient_disk_never_downloads(self):
        fake=SimpleNamespace(hf_hub_download=lambda **kw:self.fail('Unexpected model download'))
        with tempfile.TemporaryDirectory() as temp,patch.dict(sys.modules,{'huggingface_hub':fake}),patch.object(runtime,'MODEL_ROOT',Path(temp)),patch.object(runtime.shutil,'disk_usage',return_value=SimpleNamespace(free=100)):
            with self.assertRaisesRegex(RuntimeError,'Insufficient'):runtime.ensure_models()

    def test_secret_redaction(self):
        with patch.dict(os.environ,{'HF_TOKEN':'hf_privateSecret123','RUNPOD_API_KEY':'rpa_privateSecret456'}):
            result=runtime.diagnostic_error(RuntimeError('hf_privateSecret123 rpa_privateSecret456 https://example.com/?token=private'))
        self.assertNotIn('privateSecret',str(result));self.assertNotIn('?token=',str(result))


if __name__=='__main__':unittest.main()
