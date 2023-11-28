from pathlib import Path

# Use only with wav files in the original WagnerRing dataset

wav_fns = list(Path('WagnerRing_Public/01_RawData/audio_wav').glob('*.wav'))
for fn in wav_fns:
    version = fn.stem.split('_')[2][:2]
    act = fn.stem.split('_')[1][6:]
    new_fn = fn.parent / f'{version}_{act}.wav'
    fn.rename(new_fn)