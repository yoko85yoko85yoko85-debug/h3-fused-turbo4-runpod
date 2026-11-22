"""Re-encode an I2V anchor at the refinement resolution, without another text encode."""


class H3ReencodeFirstFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'positive': ('CONDITIONING',), 'vae': ('VAE',), 'image': ('IMAGE',),
            'width': ('INT', {'default': 768, 'min': 32, 'max': 4096, 'step': 32}),
            'height': ('INT', {'default': 768, 'min': 32, 'max': 4096, 'step': 32}),
        }}

    RETURN_TYPES = ('CONDITIONING',)
    FUNCTION = 'reencode'
    CATEGORY = 'conditioning/minimax_h3'

    def reencode(self, positive, vae, image, width, height):
        if any(type(v) is not int or not 32 <= v <= 4096 or v % 32 for v in (width, height)):
            raise ValueError('Anchor dimensions must be multiples of 32')
        if not positive or not all(isinstance(entry, (list, tuple)) and len(entry) == 2
                and isinstance(entry[1], dict) for entry in positive):
            raise ValueError('Invalid I2V conditioning')
        for _, meta in positive:
            anchors = meta.get('minimax_keyframes')
            if not anchors or len(anchors) != 1 or anchors[0].get('resolved_frame_index') != 0:
                raise ValueError('Expected exactly one first-frame anchor')
        import comfy.utils
        # The caller provides the already padded source; both stages share its aspect.
        pixels = comfy.utils.common_upscale(image[:1, :, :, :3].movedim(-1, 1),
                                            width, height, 'lanczos', 'disabled').movedim(1, -1)
        latent = vae.encode(pixels)
        output = []
        for embedding, meta in positive:
            updated = meta.copy()
            anchor = meta['minimax_keyframes'][0].copy()
            anchor['latent'] = latent
            updated['minimax_keyframes'] = [anchor]
            output.append([embedding, updated])
        print(f'[H3 anchor] Re-encoded first frame at {width}x{height}; text conditioning reused', flush=True)
        return (output,)


NODE_CLASS_MAPPINGS = {'H3ReencodeFirstFrame': H3ReencodeFirstFrame}
