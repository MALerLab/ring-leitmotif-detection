from pathlib import Path
import math
import random
import pandas as pd
import torch
from tqdm.auto import tqdm
from data_utils import (
    sample_instance_intervals,
    generate_non_overlapping_intervals, 
    sample_non_overlapping_interval,
    idx2motif,
    motif2idx,
)
import constants as C

class OTFDataset:
    '''
    On-the-fly dataset: Loads waveforms and converts audio to CQT on __getitem__.\n
    '''
    def __init__(
            self,
            wav_path:Path,
            instances_path: Path,
            include_none_class=False,
            max_none_samples=3000,
            duration_sec=15,
            duration_samples=646,
            split="version",
            mixup_prob=0,
            mixup_alpha=0,
            device = "cuda"
    ):
        assert split in ["version", "act"]
        self.max_none_samples = max_none_samples
        self.split = split
        self.wav_fns = sorted(list(wav_path.glob("*.pt")))
        self.stems = [x.stem for x in self.wav_fns]
        self.instances_path = instances_path
        self.duration_sec = duration_sec
        self.duration_samples = duration_samples
        self.mixup_prob = mixup_prob
        self.cur_mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.device = device

        self.wavs = {}
        self.instances_gts = {}
        self.samples = []
        self.none_samples = []
        self.num_classes = len(idx2motif) + 1 if include_none_class else len(idx2motif)

        print("Loading data...")
        for fn in tqdm(self.wav_fns, leave=False, ascii=True):
            version = fn.stem.split("_")[0]
            act = fn.stem.split("_")[1]

            # Load waveform
            with open(fn, "rb") as f:
                self.wavs[fn.stem] = torch.load(f)
            num_frames = math.ceil(self.wavs[fn.stem].shape[0] / 512)

            # Create ground truth instance tensors
            self.instances_gts[fn.stem] = torch.zeros((num_frames, self.num_classes))
            instances = list(pd.read_csv(
                instances_path / f"P-{version}/{act}.csv", sep=";").itertuples(index=False, name=None))
            for instance in instances:
                motif = instance[0]
                if motif not in idx2motif:
                    continue
                start = instance[1]
                end = instance[2]
                start_idx = int(round(start * 22050 / 512))
                end_idx = int(round(end * 22050 / 512))
                motif_idx = motif2idx[motif]
                self.instances_gts[fn.stem][start_idx:end_idx, motif_idx] = 1

            if include_none_class:
                # Add "none" class to ground truth
                self.instances_gts[fn.stem][:, -1] = 1 - self.instances_gts[fn.stem][:, :-1].max(dim=1).values

        self.sample_intervals()

    def sample_intervals(self):
        print("Sampling intervals...")
        self.samples = []
        self.none_samples = []
        for fn in tqdm(self.wav_fns, leave=False, ascii=True):
            instances = list(pd.read_csv(
                self.instances_path / f"P-{fn.stem.split('_')[0]}/{fn.stem.split('_')[1]}.csv", sep=";").itertuples(index=False, name=None))
            instances = [x for x in instances if x[0] in idx2motif]
            total_duration = self.wavs[fn.stem].shape[0] // 22050

            # Sample leitmotif instances
            version = fn.stem.split("_")[0]
            act = fn.stem.split("_")[1]
            samples_act = sample_instance_intervals(
                instances, self.duration_sec, total_duration)
            # (version, act, motif, start_sec, end_sec)
            samples_act = [(version, act, x[0], int(round(x[1] * 22050 / 512)), int(
                round(x[1] * 22050 / 512) + self.duration_samples)) for x in samples_act]
            self.samples.extend(samples_act)

            # Sample non-leitmotif instances
            occupied = instances.copy()
            none_intervals = generate_non_overlapping_intervals(
                instances, total_duration)
            none_samples_act = []
            depleted = False
            while not depleted:
                samp = sample_non_overlapping_interval(
                    none_intervals, self.duration_sec)
                if samp is None:
                    depleted = True
                else:
                    occupied.append((None, samp[0], samp[1]))
                    none_intervals = generate_non_overlapping_intervals(
                        occupied, total_duration)
                    none_samples_act.append(samp)
            none_samples_act.sort(key=lambda x: x[0])
            # (version, act, start_sec, end_sec)
            none_samples_act = [(version, act, int(round(x[0] * 22050 / 512)), int(
                round(x[0] * 22050 / 512) + self.duration_samples)) for x in none_samples_act]
            self.none_samples.extend(none_samples_act)

        random.shuffle(self.none_samples)
        self.none_samples = self.none_samples[:min(len(self.none_samples), self.max_none_samples)]
        # Create none sample index lookup table
        self.none_samples_by_version = {}
        for version in C.TRAIN_VERSIONS + C.VALID_VERSIONS:
            self.none_samples_by_version[version] = [idx for (idx, x) in enumerate(
                self.none_samples) if x[0] == version]
        self.none_samples_by_act = {}
        for act in C.TRAIN_ACTS + C.VALID_ACTS:
            self.none_samples_by_act[act] = [idx for (idx, x) in enumerate(
                self.none_samples) if x[1] == act]

    def get_subset_idxs(self, versions=None, acts=None):
        """
        Returns a list of subset indices for given versions and/or acts.\n
        """
        if versions is None and acts is None:
            return list(range(len(self.samples) + len(self.none_samples)))
        elif versions is None:
            samples = [idx for (idx, x) in enumerate(
                self.samples) if x[1] in acts]
            none_samples = [idx + len(self.samples) for (idx, x)
                            in enumerate(self.none_samples) if x[1] in acts]
            return samples + none_samples
        elif acts is None:
            samples = [idx for (idx, x) in enumerate(
                self.samples) if x[0] in versions]
            none_samples = [idx + len(self.samples) for (idx, x)
                            in enumerate(self.none_samples) if x[0] in versions]
            return samples + none_samples
        else:
            samples = [idx for (idx, x) in enumerate(
                self.samples) if x[0] in versions and x[1] in acts]
            none_samples = [idx + len(self.samples) for (idx, x) in enumerate(
                self.none_samples) if x[0] in versions and x[1] in acts]
            return samples + none_samples

    def query_motif(self, motif: str):
        """
        Query with motif name. (e.g. "Nibelungen")\n
        Returns list of (idx, version, act, start_sec, end_sec)
        """
        motif_samples = [(idx, x[0], x[1], x[3] * 512 // 22050, x[4] * 512 // 22050)
                         for (idx, x) in enumerate(self.samples) if x[2] == motif]
        if len(motif_samples) > 0:
            return motif_samples
        else:
            return None

    def preview_idx(self, idx):
        """
        Returns (version, act, motif, y, start_sec, instances_gt)
        """
        if idx < len(self.samples):
            version, act, motif, start, end = self.samples[idx]
            fn = f"{version}_{act}"
            gt = self.instances_gts[fn][start:end, :]
            start_samp = start * 512
            end_samp = start + (self.duration_sec * 22050)
            start_sec = start_samp * 512 // 22050
            y = self.wavs[fn][start_samp:end_samp]
            return version, act, motif, y, start_sec, gt
        else:
            idx -= len(self.samples)
            version, act, start, end = self.none_samples[idx]
            fn = f"{version}_{act}"
            gt = torch.zeros((end - start, self.num_classes))
            start_samp = start * 512
            end = start + (self.duration_sec * 22050)
            start_sec = start_samp * 512 // 22050
            y = self.wavs[fn][start_samp:end_samp]
            return version, act, "none", y, start_sec, gt
        
    def get_wav(self, idx):
        if idx < len(self.samples):
            version, act, _, start, end = self.samples[idx]
            fn = f"{version}_{act}"
            start_samp = start * 512
            end_samp = start_samp + (self.duration_sec * 22050)
            return self.wavs[fn][start_samp:end_samp]
        else:
            idx -= len(self.samples)
            version, act, start, end = self.none_samples[idx]
            fn = f"{version}_{act}"
            start_samp = start * 512
            end_samp = start_samp + (self.duration_sec * 22050)
            return self.wavs[fn][start_samp:end_samp]
        
    def enable_mixup(self):
        self.cur_mixup_prob = self.mixup_prob
    
    def disable_mixup(self):
        self.cur_mixup_prob = 0

    def __len__(self):
        return len(self.samples) + len(self.none_samples)

    def __getitem__(self, idx):
        if idx < len(self.samples):
            version, act, _, start, end = self.samples[idx]
            fn = f"{version}_{act}"
            start_samp = start * 512
            end_samp = start_samp + (self.duration_sec * 22050)
            wav = self.wavs[fn][start_samp:end_samp]
            if random.random() < self.cur_mixup_prob:
                mixup_idx = 0
                if self.split == "version":
                    mixup_idx = random.choice(self.none_samples_by_version[version])
                else:
                    mixup_idx = random.choice(self.none_samples_by_act[act])
                v, a, s, _ = self.none_samples[mixup_idx]
                mixup_wav = self.wavs[f"{v}_{a}"][s * 512:s * 512 + (self.duration_sec * 22050)]
                wav = (1 - self.mixup_alpha) * wav + self.mixup_alpha * mixup_wav
            return wav, self.instances_gts[fn][start:end, :]
        else:
            idx -= len(self.samples)
            version, act, start, end = self.none_samples[idx]
            fn = f"{version}_{act}"
            start_samp = start * 512
            end_samp = start_samp + (self.duration_sec * 22050)
            wav = self.wavs[fn][start_samp:end_samp]
            return wav, torch.zeros((end - start, self.num_classes))
        
class Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def collate_fn(batch):
    wav, leitmotif_gt = zip(*batch)
    wav = torch.stack(wav)
    leitmotif_gt = torch.stack(leitmotif_gt)
    return wav, leitmotif_gt
