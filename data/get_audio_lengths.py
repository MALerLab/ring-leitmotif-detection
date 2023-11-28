from pathlib import Path
import pandas as pd
import wave

wav_fns = sorted(list(Path('WagnerRing_Public/01_RawData/audio_wav').glob('*.wav')))
lengths = []
for fn in wav_fns:
    with wave.open(str(fn), "rb") as f:
        num_frames = f.getnframes()
        sr = f.getframerate()
        duration = num_frames / float(sr)
    lengths.append([fn.stem, duration, num_frames])
df = pd.DataFrame(lengths, columns=['Filename', 'Length', 'NumSamples'])
df.to_csv('audio_lengths.csv', index=False)