FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
ARG COMFY_REV=f99de883d5e65317c7f4a4b617b5aa7936622029
ARG CLIPPROJ_REV=c01ba8fb8f41b4f2094dbd0b185cdc238fb6134c
RUN git init ComfyUI && cd ComfyUI && git remote add origin https://github.com/Comfy-Org/ComfyUI.git && git fetch --depth 1 origin ${COMFY_REV} && git checkout --detach FETCH_HEAD && rm -rf .git
RUN git init /opt/ComfyUI/custom_nodes/ComfyUI-ClipProj && cd /opt/ComfyUI/custom_nodes/ComfyUI-ClipProj && git remote add origin https://github.com/nicolab28/ComfyUI-ClipProj.git && git fetch --depth 1 origin ${CLIPPROJ_REV} && git checkout --detach FETCH_HEAD && rm -rf .git
RUN python -c "import torch; print('torch=='+torch.__version__.split('+')[0])" > /tmp/torch-constraint.txt && pip install -c /tmp/torch-constraint.txt -r /opt/ComfyUI/requirements.txt runpod==1.7.9 'huggingface-hub>=0.36,<2'
WORKDIR /app
COPY handler.py runtime.py models.json workflow_api.json ./
RUN python -m py_compile handler.py runtime.py && python runtime.py --check
ENV COMFY_DIR=/opt/ComfyUI MODEL_ROOT=/models HF_HOME=/models/huggingface
# Model weights are downloaded on the first generation, never during image build.
CMD ["python", "-u", "handler.py"]
