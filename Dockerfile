FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
ARG COMFY_REV=f99de883d5e65317c7f4a4b617b5aa7936622029
ARG CLIPPROJ_REV=c01ba8fb8f41b4f2094dbd0b185cdc238fb6134c
ARG UPSCALER_REV=895e3c471164423f0ea0e8eaf45eb701efe641ae
ARG UPSCALER_ARCH_REV=e98237773011523528353a8beb4863e65b099a38
RUN git init ComfyUI && cd ComfyUI && git remote add origin https://github.com/Comfy-Org/ComfyUI.git && git fetch --depth 1 origin ${COMFY_REV} && git checkout --detach FETCH_HEAD && rm -rf .git
RUN git init /opt/ComfyUI/custom_nodes/ComfyUI-ClipProj && cd /opt/ComfyUI/custom_nodes/ComfyUI-ClipProj && git remote add origin https://github.com/nicolab28/ComfyUI-ClipProj.git && git fetch --depth 1 origin ${CLIPPROJ_REV} && git checkout --detach FETCH_HEAD && rm -rf .git
RUN git init /opt/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3_LatentUpscaler && cd /opt/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3_LatentUpscaler && git remote add origin https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler.git && git fetch --depth 1 origin ${UPSCALER_REV} && git checkout --detach FETCH_HEAD && rm -rf .git
RUN git init /opt/ComfyUI/custom_nodes/ComfyUI-H3-Latent-Upscaler-Mamad8 && cd /opt/ComfyUI/custom_nodes/ComfyUI-H3-Latent-Upscaler-Mamad8 && git remote add origin https://github.com/mamad8c/ComfyUI-H3-Latent-Upscaler-Mamad8.git && git fetch --depth 1 origin ${UPSCALER_ARCH_REV} && git checkout --detach FETCH_HEAD && rm -rf .git
ARG SPECTRUM_REV=7df883bb2361b7884ad789e97effd1dc37c8963a
ARG SLA_REV=cc944b5064d1b15fc77664c7138d9752bba66302
RUN git init /opt/ComfyUI/custom_nodes/ComfyUI-Spectrum-MiniMax-H3 && cd /opt/ComfyUI/custom_nodes/ComfyUI-Spectrum-MiniMax-H3 && git remote add origin https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git && git fetch --depth 1 origin ${SPECTRUM_REV} && git checkout --detach FETCH_HEAD && rm -rf .git
# Install only the H3 SLA subpackage, not unrelated nodes from the parent repository.
RUN git init /tmp/plaguekind && cd /tmp/plaguekind && git remote add origin https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes.git && git fetch --depth 1 origin ${SLA_REV} && git checkout --detach FETCH_HEAD && cp -a ComfyUI-H3-SLA-Attention /opt/ComfyUI/custom_nodes/ComfyUI-H3-SLA-Attention && rm -rf /tmp/plaguekind
RUN python -c "import triton; print('Triton available')"
RUN python -c "import torch; print('torch=='+torch.__version__.split('+')[0])" > /tmp/torch-constraint.txt && pip install -c /tmp/torch-constraint.txt -r /opt/ComfyUI/requirements.txt runpod==1.7.9 'huggingface-hub>=0.36,<2'
WORKDIR /app
COPY handler.py runtime.py models.json workflow_api.json ./
RUN python -m py_compile handler.py runtime.py && python runtime.py --check
ENV COMFY_DIR=/opt/ComfyUI MODEL_ROOT=/models HF_HOME=/models/huggingface
# Model weights are downloaded on the first generation, never during image build.
CMD ["python", "-u", "handler.py"]
