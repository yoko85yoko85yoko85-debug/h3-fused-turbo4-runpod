import importlib.util,json,sys,tempfile,unittest,urllib.error
from pathlib import Path
from unittest.mock import patch,MagicMock
sys.path.insert(0,str(Path(__file__).parent))
import manage,workflow

class Tests(unittest.TestCase):
    def test_graph(self):
        ui,api=workflow.build();nodes={n['id']:n for n in ui['nodes']}
        self.assertEqual(len(nodes),16);self.assertEqual(api['11']['inputs']['steps'],8)
        self.assertEqual(api['3']['inputs']['clip_name'],workflow.CLIP)
        for lid,source,out,target,ins,kind in ui['links']:
            self.assertEqual(nodes[source]['outputs'][out]['type'],kind)
            self.assertEqual(nodes[target]['inputs'][ins]['link'],lid)
            self.assertIn(lid,nodes[source]['outputs'][out]['links'])
        self.assertEqual(api['7']['inputs']['length'],56)
        self.assertEqual(api['15']['inputs']['audio'],[14,0])

    def test_aspect_selector(self):
        ui,api=workflow.build_aspect();old_ui,old_api=workflow.build()
        nodes={n['id']:n for n in ui['nodes']}
        self.assertEqual(api['7']['inputs']['width'],[17,0])
        self.assertEqual(api['7']['inputs']['height'],[17,1])
        self.assertEqual(api['17']['inputs']['multiple'],32)
        for key in old_api:
            if key!='7':self.assertEqual(api[key],old_api[key])
        self.assertEqual(api['7']['inputs']['length'],[20,1])
        for lid,source,out,target,ins,kind in ui['links']:
            self.assertEqual(nodes[source]['outputs'][out]['type'],kind)
            self.assertEqual(nodes[target]['inputs'][ins]['link'],lid)
        self.assertEqual(next(x for x in nodes[7]['inputs'] if x['name']=='width')['widget'],{'name':'width'})

    def test_seconds_frame_conversion(self):
        import math
        ui,api=workflow.build_aspect()
        expression=api['20']['inputs']['expression']
        for seconds,expected in [(2,56),(3,73),(5,124),(8,192),(10,243),(15,362),(2.33,56)]:
            frames=eval(expression,{'__builtins__':{}},{'a':seconds,'ceil':math.ceil,'max':max})
            self.assertEqual(frames,expected)
            self.assertGreaterEqual(frames/24,seconds)
            self.assertEqual((frames-5)%17,0)
        self.assertEqual(api['20']['inputs']['values.a'],[19,0])

    def test_duration_presets(self):
        import duration_presets,math
        node=duration_presets.H3DurationPreset()
        choices=node.INPUT_TYPES()['required']['duration'][0]
        self.assertEqual(choices,['2 秒','5 秒','7 秒','10 秒','12 秒','15 秒'])
        for label,frames in zip(choices,[56,124,175,243,294,362]):
            value=node.select(label)[0];n=math.ceil(value*24)
            self.assertEqual(n+(5-n)%17,frames)
        with self.assertRaises(ValueError):node.select('3 秒')
        _,api=workflow.build_aspect()
        self.assertEqual(api['19']['class_type'],'H3DurationPreset')

    def test_original_controls_and_connections(self):
        import math
        ui,api=workflow.build_original_controls();_,base=workflow.build()
        for key in base:
            if key!='7':self.assertEqual(api[key],base[key])
        self.assertEqual(api['17']['inputs'],{'image':[6,0],'upscale_method':'nearest-exact','megapixels':0.4,'resolution_steps':32})
        self.assertEqual(api['18']['inputs']['image'],[17,0])
        self.assertEqual(api['7']['inputs']['width'],[18,0])
        self.assertEqual(api['7']['inputs']['height'],[18,1])
        self.assertEqual(api['7']['inputs']['first_frame'],[6,0])
        self.assertEqual(api['19']['inputs']['value'],10.0)
        self.assertEqual(eval(api['20']['inputs']['expression'],{'__builtins__':{}},{'a':2.34,'round':round,'max':max}),56)
        self.assertNotIn('H3DurationPreset',{n['class_type'] for n in api.values()})
        self.assertNotIn('ResolutionSelector',{n['class_type'] for n in api.values()})
        nodes={n['id']:n for n in ui['nodes']}
        for lid,source,out,target,slot,kind in ui['links']:
            self.assertEqual(nodes[source]['outputs'][out]['type'],kind)
            self.assertEqual(nodes[target]['inputs'][slot]['link'],lid)

    def test_write_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            workflow.write(tmp)
            self.assertTrue((Path(tmp)/'ComfyUI/user/default/workflows/H3-Turbo8-I2V.json').exists())

    def test_no_jupyter_matching(self):
        self.assertFalse(manage.is_comfy(['python','-m','jupyterlab'],'/workspace'))
        self.assertFalse(manage.is_comfy(['/filebrowser'],'/opt/ComfyUI'))
        self.assertTrue(manage.is_comfy(['python','main.py','--port','8188'],'/workspace/ComfyUI'))
        self.assertEqual(manage.port_of(['python','main.py','--port=8190']),8190)
        self.assertEqual(manage.ancestors(20,{20:([],'.',10),10:([],'.',1)}),{10,20})

    def test_active_queue_no_stop(self):
        response=MagicMock();response.__enter__.return_value=__import__('io').BytesIO(b'{"queue_running":[1],"queue_pending":[]}')
        with patch.object(manage,'process_table',return_value={123:(['python','main.py'],'/opt/ComfyUI',1)}),patch.object(manage.urllib.request,'urlopen',return_value=response),patch.object(manage.os,'kill') as kill:
            with self.assertRaisesRegex(RuntimeError,'running/queued'):manage.restart(Path('/fake'),8188)
            kill.assert_not_called()

    def test_pid1_never_stop(self):
        with patch.object(manage,'process_table',return_value={1:(['python','main.py'],'/opt/ComfyUI',0)}),patch.object(manage.os,'kill') as kill:
            with self.assertRaisesRegex(RuntimeError,'PID1'):manage.restart(Path('/fake'),8188)
            kill.assert_not_called()

    def test_shared_service_never_stop(self):
        table={123:(['python','main.py'],'/opt/ComfyUI',100),100:(['bash','start-all.sh'],'/opt',1),124:(['jupyter','lab'],'/workspace',100)}
        status=MagicMock(stdout='comfyui RUNNING pid 100, uptime 1:00:00')
        with patch.object(manage,'process_table',return_value=table),patch.object(manage,'check_idle'),patch.object(manage.subprocess,'run',return_value=status) as run,patch.object(manage.os,'kill') as kill:
            with self.assertRaisesRegex(RuntimeError,'shared/unknown'):manage.restart(Path('/fake'),8188)
            self.assertEqual(run.call_count,1);kill.assert_not_called()

    def test_template_keeps_password_proxy(self):
        with patch.object(manage,'is_runpod_template',return_value=True):
            self.assertEqual(manage.connection_config(8188),(18188,'127.0.0.1',True))
            self.assertEqual(manage.connection_config(8189),(8189,'0.0.0.0',False))
        with patch.object(manage.urllib.request,'urlopen',side_effect=urllib.error.HTTPError('http://localhost',401,'Auth',None,None)):
            manage.check_proxy(8188)
        with patch.object(manage.urllib.request,'urlopen',return_value=MagicMock()):
            with self.assertRaisesRegex(RuntimeError,'bypass'):manage.check_proxy(8188)

    def test_port_check_allows_time_wait(self):
        import socket
        server=socket.socket();server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind(('127.0.0.1',0));port=server.getsockname()[1];server.listen()
        client=socket.create_connection(('127.0.0.1',port));peer,_=server.accept()
        peer.shutdown(socket.SHUT_WR);client.recv(1);client.close();peer.close();server.close()
        manage.wait_for_port('127.0.0.1',port,seconds=0)

    def test_port_check_refuses_live_listener(self):
        import socket
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',0));server.listen()
            with self.assertRaisesRegex(RuntimeError,'live listener'):
                manage.wait_for_port('127.0.0.1',server.getsockname()[1],seconds=0)

    def test_manifest(self):
        m=json.loads((Path(__file__).parent/'models.json').read_text())
        self.assertEqual(len(m['files']),5)
        self.assertTrue(all(len(x['sha256'])==64 and x['size']>0 for x in m['files']))
        files={f['path'].split('/')[-1] for f in m['files']}
        self.assertEqual(files,{workflow.MODEL,workflow.LORA,workflow.CLIP,workflow.VAE,workflow.AUDIO})

    def test_embedded_python_syntax(self):
        shell=(Path(__file__).parent/'install.sh').read_text()
        for marker in ['PREFLIGHT','CHECKGPU','H3_EMBED_WORKFLOW_PY','H3_EMBED_MANAGE_PY','H3_EMBED_SETUP_PY']:
            content=shell.split("<<'"+marker+"'\n",1)[1].split('\n'+marker+'\n',1)[0]
            compile(content,'embedded-'+marker,'exec')
        self.assertIn('export H3_POD_DIR H3_PORT H3_NO_START\npython3',shell)

if __name__=='__main__':unittest.main()
