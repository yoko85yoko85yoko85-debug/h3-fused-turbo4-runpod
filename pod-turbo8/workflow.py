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

def build_aspect():
    ui,api=build()
    byid={n['id']:n for n in ui['nodes']}
    selector={'id':17,'type':'ResolutionSelector','title':'サイズ設定：アスペクト比 / MP',
       'pos':[370,600],'size':[330,210],'flags':{},'order':0,'mode':0,'inputs':[],
       'outputs':[{'name':'width','type':'INT','links':[],'slot_index':0},
                  {'name':'height','type':'INT','links':[],'slot_index':1}],
       'properties':{'Node name for S&R':'ResolutionSelector'},
       'widgets_values':['16:9 (Widescreen)',0.25,32],
       'widgets_values_named':{'aspect_ratio':'16:9 (Widescreen)','megapixels':0.25,'multiple':32}}
    ui['nodes'].append(selector)
    # Preserve input slot indices of the existing image/CLIP/VAE connections.
    # Numeric widgets become connected inputs; their saved widget values remain
    # in the original positions so prompt and length still deserialize normally.
    for slot,name in enumerate(('width','height')):
        lid=len(ui['links'])+1;target_slot=len(byid[7]['inputs'])
        ui['links'].append([lid,17,slot,7,target_slot,'INT'])
        byid[7]['inputs'].append({'name':name,'type':'INT','widget':{'name':name},'link':lid})
        selector['outputs'][slot]['links'].append(lid)
        api['7']['inputs'][name]=[17,slot]
    byid[7]['widgets_values'][1:3]=[672,384]
    ui['nodes'].append({'id':18,'type':'MarkdownNote','title':'サイズ目安表',
        'pos':[370,850],'size':[400,570],'flags':{},'order':17,'mode':0,
        'inputs':[],'outputs':[],'properties':{},'widgets_values':['## メガピクセルの目安\n\n**16:9・multiple＝32 の場合**\n\n| megapixels | 出力サイズ |\n|---|---|\n| 0.2 | 608 × 352 |\n| 0.3 | 736 × 416 |\n| 0.4 | 864 × 480 |\n| 0.5 | 960 × 544 |\n| 0.6 | 1056 × 608 |\n| 0.7 | 1152 × 640 |\n| 0.8 | 1216 × 672 |\n| 0.9 | 1280 × 736 |\n| 0.98 | 1344 × 768 |\n\nアスペクト比を変えると、同じMPでも幅・高さは変わります。\n32px刻みで丸めるため、比率と画素数は概算です。\n\n**正方形の例：1:1・0.6MP → 800 × 800**\n\n長さはH3ノードの length で指定します。']})
    ui['last_node_id']=18;ui['last_link_id']=len(ui['links'])
    api['17']={'class_type':'ResolutionSelector','inputs':{'aspect_ratio':'16:9 (Widescreen)','megapixels':0.25,'multiple':32}}
    expression='max(5, ceil(a * 24)) + (5 - max(5, ceil(a * 24))) % 17'
    duration={'id':19,'type':'H3DurationPreset','title':'動画の長さ（秒・6択）',
       'pos':[740,650],'size':[330,110],'flags':{},'order':0,'mode':0,'inputs':[],
       'outputs':[{'name':'FLOAT','type':'FLOAT','links':[],'slot_index':0}],
       'properties':{'Node name for S&R':'H3DurationPreset'},'widgets_values':['2 秒']}
    convert={'id':20,'type':'ComfyMathExpression','title':'秒 → H3フレーム数（変更不要）',
       'pos':[1140,650],'size':[380,180],'flags':{},'order':1,'mode':0,
       'inputs':[{'name':'values.a','type':'FLOAT,INT,BOOLEAN','link':None},
                 {'name':'values.b','type':'FLOAT,INT,BOOLEAN','shape':7,'link':None}],
       'outputs':[{'name':'FLOAT','type':'FLOAT','links':[],'slot_index':0},
                  {'name':'INT','type':'INT','links':[],'slot_index':1},
                  {'name':'BOOL','type':'BOOLEAN','links':[],'slot_index':2}],
       'properties':{'Node name for S&R':'ComfyMathExpression'},'widgets_values':[expression]}
    ui['nodes'].extend([duration,convert])
    lid=len(ui['links'])+1
    ui['links'].append([lid,19,0,20,0,'FLOAT'])
    duration['outputs'][0]['links'].append(lid);convert['inputs'][0]['link']=lid
    lid=len(ui['links'])+1;target_slot=len(byid[7]['inputs'])
    ui['links'].append([lid,20,1,7,target_slot,'INT'])
    convert['outputs'][1]['links'].append(lid)
    byid[7]['inputs'].append({'name':'length','type':'INT','widget':{'name':'length'},'link':lid})
    api['19']={'class_type':'H3DurationPreset','inputs':{'duration':'2 秒'}}
    api['20']={'class_type':'ComfyMathExpression','inputs':{'expression':expression,'values.a':[19,0]}}
    api['7']['inputs']['length']=[20,1]
    ui['last_node_id']=20;ui['last_link_id']=len(ui['links'])
    note=next(n for n in ui['nodes'] if n['id']==18)
    note['widgets_values'][0]=note['widgets_values'][0].replace('長さはH3ノードの length で指定します。',
      '## 動画の長さ\n\n「動画の長さ（秒・6択）」で2・5・7・10・12・15秒から選択します。\nH3の17k+5フレーム刻みに切り上げるため、指定より少し長くなります。\n\n| 指定秒数 | 実際の長さ（24fps） |\n|---|---|\n| 2 | 約2.33秒 |\n| 7 | 約7.29秒 |\n| 5 | 約5.17秒 |\n| 8 | 8.00秒 |\n| 10 | 約10.13秒 |\n| 12 | 約12.25秒 |\n| 15 | 約15.08秒 |\n\n「秒 → H3フレーム数」の数式は変更不要です。')
    note['size']=[400,920]
    return ui,api


def build_original_controls():
    """Match the uploaded workflow's active geometry and duration controls."""
    ui,api=build();byid={n['id']:n for n in ui['nodes']}
    def node(i,kind,title,pos,size,outputs,widgets,inputs=None):
        n={'id':i,'type':kind,'title':title,'pos':pos,'size':size,'flags':{},'order':i-1,'mode':0,
           'inputs':inputs or [],'outputs':[{'name':name,'type':typ,'links':[],'slot_index':slot} for slot,(name,typ) in enumerate(outputs)],
           'properties':{'Node name for S&R':kind},'widgets_values':widgets}
        ui['nodes'].append(n);byid[i]=n;return n
    node(17,'ImageScaleToTotalPixels','生成サイズ（MP・画像の比率を維持）',[370,680],[340,190],
         [('IMAGE','IMAGE')],['nearest-exact',0.4,32])
    node(18,'GetImageSize','リサイズ後の幅・高さ',[740,680],[260,130],
         [('width','INT'),('height','INT'),('batch_size','INT')],[])
    node(19,'PrimitiveFloat','動画の長さ（秒）',[740,880],[330,110],[('FLOAT','FLOAT')],[10.0])
    expression='max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17'
    node(20,'ComfyMathExpression','秒 → フレーム（以前と同じ式）',[1140,680],[380,180],
         [('FLOAT','FLOAT'),('INT','INT'),('BOOL','BOOLEAN')],[expression])
    def connect(source,out,target,name,widget=False,typ=None):
        output=byid[source]['outputs'][out];kind=typ or output['type'];lid=len(ui['links'])+1
        slot=len(byid[target]['inputs'])
        entry={'name':name,'type':kind,'link':lid}
        if widget:entry['widget']={'name':name}
        byid[target]['inputs'].append(entry);output['links'].append(lid)
        ui['links'].append([lid,source,out,target,slot,output['type']])
    connect(6,0,17,'image');connect(17,0,18,'image')
    connect(18,0,7,'width',True);connect(18,1,7,'height',True)
    connect(19,0,20,'values.a',typ='FLOAT,INT,BOOLEAN')
    byid[20]['inputs'].append({'name':'values.b','type':'FLOAT,INT,BOOLEAN','shape':7,'link':None})
    connect(20,1,7,'length',True)
    api['17']={'class_type':'ImageScaleToTotalPixels','inputs':{'image':[6,0],'upscale_method':'nearest-exact','megapixels':0.4,'resolution_steps':32}}
    api['18']={'class_type':'GetImageSize','inputs':{'image':[17,0]}}
    api['19']={'class_type':'PrimitiveFloat','inputs':{'value':10.0}}
    api['20']={'class_type':'ComfyMathExpression','inputs':{'expression':expression,'values.a':[19,0]}}
    api['7']['inputs'].update(width=[18,0],height=[18,1],length=[20,1])
    byid[7]['widgets_values'][3]=243
    # Exactly as before: the original input, not the resized preview, anchors H3.
    assert api['7']['inputs']['first_frame']==[6,0]
    old_table=next(n for n in build_aspect()[0]['nodes'] if n['id']==18)['widgets_values'][0].split('## 動画の長さ')[0]
    note='## 前のワークフローと同じ設定方法\n\n生成サイズは「生成サイズ」のmegapixelsで指定します。初期値0.4MP、32px刻み、nearest-exact。入力画像の縦横比を維持して幅・高さを自動取得します。\n\n手動のアスペクト比選択は不要です。元のResolution Selectorは未接続だったため使いません。\n\n時間は「動画の長さ（秒）」に入力します。元の初期値は10秒。元と同じround＋17k+5変換で約10.13秒になります。6択ノードは使いません。\n\n'+old_table
    node(21,'MarkdownNote','使い方とサイズ目安',[370,1040],[500,760],[],[note])
    ui['last_node_id']=21;ui['last_link_id']=len(ui['links'])
    return ui,api


def write(root):
    root=Path(root);ui,api=build()
    dest=root/'ComfyUI/user/default/workflows';dest.mkdir(parents=True,exist_ok=True)
    (dest/'H3-Turbo8-I2V.json').write_text(json.dumps(ui,ensure_ascii=False,indent=2))
    (root/'H3-Turbo8-I2V-API.json').write_text(json.dumps(api,ensure_ascii=False,indent=2))
    original_ui,original_api=build_original_controls()
    (dest/'H3-Turbo8-I2V-OriginalControls.json').write_text(json.dumps(original_ui,ensure_ascii=False,indent=2))
    (root/'H3-Turbo8-I2V-OriginalControls-API.json').write_text(json.dumps(original_api,ensure_ascii=False,indent=2))
    aspect_ui,aspect_api=build_aspect()
    (dest/'H3-Turbo8-I2V-Aspect.json').write_text(json.dumps(aspect_ui,ensure_ascii=False,indent=2))
    (root/'H3-Turbo8-I2V-Aspect-API.json').write_text(json.dumps(aspect_api,ensure_ascii=False,indent=2))

if __name__=='__main__':
    import sys
    write(sys.argv[1])
