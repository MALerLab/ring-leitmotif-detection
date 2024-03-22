from pathlib import Path
import torch
import torchaudio
from tqdm.auto import tqdm

wav_fns = list(Path('WagnerRing_Public/01_RawData/audio_wav').glob('*.wav'))

for fn in tqdm(wav_fns):
    y, sr = torchaudio.load(fn)
    y = y.mean(dim=0)
    y = torchaudio.functional.resample(y, sr, 22050)
    new_fn = Path('wav-22050') / (fn.stem + '.pt')
    new_fn.parent.mkdir(exist_ok=True, parents=True)
    torch.save(y, new_fn)