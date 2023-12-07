from pathlib import Path
import pickle
import pandas as pd
import torch
from tqdm.auto import tqdm
from data_utils import motif2idx, sample_instance_intervals, generate_non_overlapping_intervals, sample_non_overlapping_interval

class LeitmotifDataset:
    def __init__(self, pkl_path:Path, instances_path:Path, duration_sec=15, duration_samples=646):
        self.cqt = {}
        self.instances_gt = {}
        self.samples = []
        self.none_samples = []

        print("Creating dataset...")
        pkl_fns = sorted(list(pkl_path.glob("*.pkl")))
        for fn in tqdm(pkl_fns):
            # Load CQT data
            with open(fn, "rb") as f:
                self.cqt[fn.stem] = torch.tensor(pickle.load(f)).T # (time, n_bins)

            # Create ground truth instance tensors
            self.instances_gt[fn.stem] = torch.zeros((self.cqt[fn.stem].shape[0], 20))
            instances = list(pd.read_csv(instances_path / f"P-{fn.stem.split('_')[0]}/{fn.stem.split('_')[1]}.csv", sep=";").itertuples(index=False, name=None))
            for instance in instances:
                motif = instance[0]
                start = instance[1]
                end = instance[2]
                start_idx = int(round(start * 22050 / 512))
                end_idx = int(round(end * 22050 / 512))
                motif_idx = motif2idx[motif]
                self.instances_gt[fn.stem][start_idx:end_idx, motif_idx] = 1
            
            # Sample leitmotif instances
            samples_act = sample_instance_intervals(instances, duration_sec)
            samples_act = [(fn.stem, x[0], int(round(x[1] * 22050 / 512)), int(round(x[1] * 22050 / 512) + duration_samples)) for x in samples_act]
            self.samples.extend(samples_act)

            # Sample non-leitmotif instances
            total_duration = self.cqt[fn.stem].shape[0] * 512 // 22050
            occupied = instances.copy()
            none_intervals = generate_non_overlapping_intervals(instances, total_duration)
            none_samples_act = []
            depleted = False
            while not depleted:
                samp = sample_non_overlapping_interval(none_intervals, duration_sec)
                if samp is None:
                    depleted = True
                else:
                    occupied.append((None, samp[0], samp[1]))
                    none_intervals = generate_non_overlapping_intervals(occupied, total_duration)
                    none_samples_act.append(samp)
            none_samples_act.sort(key=lambda x: x[0])
            none_samples_act = [(fn.stem, int(round(x[0] * 22050 / 512)), int(round(x[0] * 22050 / 512) + duration_samples)) for x in none_samples_act]
            self.none_samples.extend(none_samples_act)

    def __len__(self):
        return len(self.samples) + len(self.none_samples)
    
    def __getitem__(self, idx):
        if idx < len(self.samples):
            fn, _, start, end = self.samples[idx]
            return self.cqt[fn][start:end, :], self.instances_gt[fn][start:end, :]
        else:
            idx -= len(self.samples)
            fn, start, end = self.none_samples[idx]
            return self.cqt[fn][start:end, :], torch.zeros((end - start, 20))