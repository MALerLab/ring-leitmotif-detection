from pathlib import Path
import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm

wav_path = Path('../data/wav-22050')
instances_path = Path('../data/LeitmotifOccurrencesInstances/Instances')
wav_fns = sorted(list(wav_path.glob("*.pt")))

for fn in tqdm(wav_fns, ascii=True):
    version = fn.stem.split("_")[0]
    act = fn.stem.split("_")[1]
    instances = list(pd.read_csv(
        instances_path / f"P-{version}/{act}.csv", sep=";").itertuples(index=False, name=None))
    instances.sort(key=lambda x: x[1])
    instances_headtail = instances[:3] + instances[-3:]
    
    y = torch.load(fn)
    y = y.unsqueeze(0).expand(2, -1)

    for instance in tqdm(instances_headtail, leave=False, ascii=True):
        motif = instance[0]
        start = instance[1]
        end = instance[2]
        start_samp = int(round(start * 22050))
        end_samp = int(round(end * 22050))
    
        extract_fn = Path(f"extracts/{version}/{act}_{start}_{motif}.wav")
        extract_fn.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(extract_fn), y[:, start_samp:end_samp], 22050)