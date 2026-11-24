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
