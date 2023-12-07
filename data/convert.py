from pathlib import Path
import wave
import pickle
import librosa
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

Path("CQT").mkdir(exist_ok=True)

durations = {
    'FileName': [],
    'NumFrames': [],
    'Duration': []
}
wav_fns = sorted(list(Path("WagnerRing_Public/01_RawData/audio_wav").glob("*.wav")))

print(f"Converting {len(wav_fns)} files...")
for fn in tqdm(wav_fns):
    durations['FileName'].append(fn.stem)
    with wave.open(str(fn), "rb") as wf:
        durations['NumFrames'].append(wf.getnframes())
        durations['Duration'].append(wf.getnframes() / wf.getframerate())
    y, sr = librosa.load(fn, mono=True)
    cqt = librosa.cqt(y, sr=sr, hop_length=512, bins_per_octave=12, n_bins=84)
    cqt_mag = np.abs(cqt) # Take magnitude of CQT
    max_norm = librosa.util.normalize(cqt_mag, axis=0, norm=np.inf)
    with open(f"CQT/{fn.stem}.pkl", "wb") as f:
        pickle.dump(max_norm, f)

df = pd.DataFrame(durations)
df.to_csv("durations.csv")