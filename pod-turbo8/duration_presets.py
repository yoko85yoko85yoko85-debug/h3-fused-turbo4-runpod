"""Six explicit H3 duration choices; no third-party dependencies."""
DURATIONS={f'{n} 秒':float(n) for n in (2,5,7,10,12,15)}

class H3DurationPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'duration':(list(DURATIONS),{'default':'2 秒'})}}
    RETURN_TYPES=('FLOAT',)
    RETURN_NAMES=('seconds',)
    FUNCTION='select'
    CATEGORY='H3/settings'

    def select(self,duration):
        if duration not in DURATIONS:raise ValueError('Choose 2, 5, 7, 10, 12 or 15 seconds')
        return (DURATIONS[duration],)

NODE_CLASS_MAPPINGS={'H3DurationPreset':H3DurationPreset}
NODE_DISPLAY_NAME_MAPPINGS={'H3DurationPreset':'動画の長さ（秒・6択）'}
