from pathlib import Path
import math
import random
import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm
from .data_utils import (
    sample_instance_intervals,
    generate_non_overlapping_intervals, 
    sample_non_overlapping_interval
)

class FramewiseDataset:
    def __init__(
            self,
            wav_path:Path,
            instances_path:Path,
            train_versions,
            valid_versions,
            train_acts,
            valid_acts,
            idx2motif,
            include_none_class=False,
            duration_sec=15,
            duration_samples=646,
            split="version",
            mixup_prob=0,
            mixup_alpha=0,
            device = "cuda"
    ):
        assert split in ["version", "act"]
        self.train_versions = train_versions
        self.valid_versions = valid_versions
        self.train_acts = train_acts
        self.valid_acts = valid_acts
        self.idx2motif = idx2motif
        self.motif2idx = {x: i for i, x in enumerate(idx2motif)}
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
        self.num_classes = len(self.idx2motif) + 1 if include_none_class else len(self.idx2motif)

        print("Loading data...")
        for fn in tqdm(self.wav_fns, leave=False, ascii=True):
            version = fn.stem.split("_")[0]
            act = fn.stem.split("_")[1]

            # Load waveform
            with open(fn, "rb") as f:
                self.wavs[fn.stem] = torch.load(f, weights_only=True)
            num_frames = math.ceil(self.wavs[fn.stem].shape[0] / 512)

            # Create ground truth instance tensors
            self.instances_gts[fn.stem] = torch.zeros((num_frames, self.num_classes))
            instances = list(pd.read_csv(
                self.instances_path / f"P-{version}/{act}.csv", sep=";").itertuples(index=False, name=None))
            for instance in instances:
                motif = instance[0]
                if motif not in self.idx2motif:
                    continue
                start = instance[1]
                end = instance[2]
                start_idx = int(round(start * 22050 / 512))
                end_idx = int(round(end * 22050 / 512))
                motif_idx = self.motif2idx[motif]
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
            version = fn.stem.split("_")[0]
            act = fn.stem.split("_")[1]
            instances = list(pd.read_csv(
                self.instances_path / f"P-{fn.stem.split('_')[0]}/{fn.stem.split('_')[1]}.csv", sep=";").itertuples(index=False, name=None))
            instances = [x for x in instances if x[0] in self.idx2motif]
            total_duration = self.wavs[fn.stem].shape[0] // 22050

            # Sample leitmotif instances
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
        # Create none sample index lookup table
        self.none_samples_by_version = {}
        for version in self.train_versions + self.valid_versions:
            self.none_samples_by_version[version] = [idx for (idx, x) in enumerate(
                self.none_samples) if x[0] == version]
        self.none_samples_by_act = {}
        for act in self.train_acts + self.valid_acts:
            self.none_samples_by_act[act] = [idx for (idx, x) in enumerate(
                self.none_samples) if x[1] == act]

    def get_subset_idxs(self, versions=None, acts=None):
        """
        Returns a list of subset indices for given versions and/or acts.\n
        """
        if versions is None and acts is None:
            return list(range(len(self.samples))), list(range(len(self.none_samples)))
        elif versions is None:
            samples = [idx for (idx, x) in enumerate(
                self.samples) if x[1] in acts]
            none_samples = [idx + len(self.samples) for (idx, x)
                            in enumerate(self.none_samples) if x[1] in acts]
            return samples, none_samples
        elif acts is None:
            samples = [idx for (idx, x) in enumerate(
                self.samples) if x[0] in versions]
            none_samples = [idx + len(self.samples) for (idx, x)
                            in enumerate(self.none_samples) if x[0] in versions]
            return samples, none_samples
        else:
            samples = [idx for (idx, x) in enumerate(
                self.samples) if x[0] in versions and x[1] in acts]
            none_samples = [idx + len(self.samples) for (idx, x) in enumerate(
                self.none_samples) if x[0] in versions and x[1] in acts]
            return samples, none_samples

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
    def __init__(self, dataset, indices:tuple, non_sample_ratio:float=2.0):
        self.dataset = dataset
        self.sample_indices, self.non_sample_indices = indices
        self.non_sample_ratio = int(len(self.sample_indices) * non_sample_ratio)

    def __len__(self):
        return len(self.sample_indices) + min(len(self.non_sample_indices), self.non_sample_ratio)

    def __getitem__(self, idx):
        if idx < len(self.sample_indices):
            return self.dataset[self.sample_indices[idx]]
        else:
            idx -= len(self.sample_indices)
            return self.dataset[self.non_sample_indices[idx]]


def collate_fn(batch):
    wav, leitmotif_gt = zip(*batch)
    wav = torch.stack(wav)
    leitmotif_gt = torch.stack(leitmotif_gt)
    return wav, leitmotif_gt

class YOLODataset:
    def __init__(
            self,
            wav_path:Path,
            instances_path:Path,
            train_versions,
            valid_versions,
            train_acts,
            valid_acts,
            idx2motif,
            anchors,
            test_versions=[],
            test_acts=[],
            use_merged_data=False,
            duration_sec=15,
            overlap_sec=3,
            duration_frames=646,
            include_threshold=0.5,
            S=11,
            split="version",
            mixup_prob=0,
            mixup_alpha=0,
            pitchshift_prob=0,
            pitchshift_semitones=3,
            eval=False,
            device = "cuda"
    ):
        assert split in ["version", "act"]
        self.train_versions = train_versions
        self.valid_versions = valid_versions
        self.train_acts = train_acts
        self.valid_acts = valid_acts
        self.test_versions = test_versions
        self.test_acts = test_acts
        self.idx2motif = idx2motif
        self.motif2idx = {x: i for i, x in enumerate(idx2motif)}
        self.anchors = anchors
        self.wav_fns = sorted(list(wav_path.glob("*.pt")))
        self.stems = [x.stem for x in self.wav_fns]
        if use_merged_data:
            self.instances_path = instances_path / "MergedInstances"
        else:
            self.instances_path = instances_path / "Instances"
        self.duration_sec = duration_sec
        self.increment_sec = duration_sec - overlap_sec
        self.duration_frames = duration_frames
        self.include_threshold = include_threshold
        self.S = S
        self.split = split
        self.mixup_prob = mixup_prob
        self.cur_mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.pitchshift_prob = pitchshift_prob
        self.cur_pitchshift_prob = pitchshift_prob
        self.pitchshift_semitones = pitchshift_semitones
        self.eval = eval
        self.device = device
        self.grid_w = 1 / S

        self.wavs = {}
        self.samples = []
        self.none_samples = []
        self.num_classes = len(self.idx2motif)
        self.framewise_gts = {}

        print("Loading data...")
        for fn in tqdm(self.wav_fns, leave=False, ascii=True):
            version = fn.stem.split("_")[0]
            act = fn.stem.split("_")[1]
            if self.split == "version" and version not in self.train_versions + self.valid_versions + self.test_versions:
                continue
            if self.split == "act" and act not in self.train_acts + self.valid_acts + self.test_acts:
                continue
            if self.eval and self.split == "version" and version in self.train_versions:
                continue
            if self.eval and self.split == "act" and act in self.train_acts:
                continue
            
            # Load waveform and instance list
            with open(fn, "rb") as f:
                self.wavs[fn.stem] = torch.load(f, weights_only=True)
            length_sec = self.wavs[fn.stem].shape[0] / 22050
            instances = list(pd.read_csv(
                self.instances_path / f"P-{version}/{act}.csv", sep=";").itertuples(index=False, name=None))
            instances.sort(key=lambda x: x[1])

            if self.eval:
                num_frames = math.ceil(self.wavs[fn.stem].shape[0] / 512)
                # Create framewise ground truth tensors
                self.framewise_gts[fn.stem] = torch.zeros((num_frames, self.num_classes))
                instances = list(pd.read_csv(
                    self.instances_path / f"P-{version}/{act}.csv", sep=";").itertuples(index=False, name=None))
                for instance in instances:
                    motif = instance[0]
                    if motif not in self.idx2motif: continue
                    start = instance[1]
                    end = instance[2]
                    start_idx = int(round(start * 22050 / 512))
                    end_idx = int(round(end * 22050 / 512))
                    motif_idx = self.motif2idx[motif]
                    self.framewise_gts[fn.stem][start_idx:end_idx, motif_idx] = 1

            # Slice instances and create ground truth tensors
            for i in range(0, math.floor(length_sec), self.increment_sec):
                start, end = i, i + duration_sec
                if end > length_sec:
                    break
                gt = torch.zeros((len(self.anchors), self.S, 4))

                contains_instance = False
                for instance in instances:
                    if instance[2] < start: continue
                    if instance[1] > end: break
                    if instance[0] not in self.idx2motif: continue
                    fragment_start = max(start, instance[1])
                    fragment_end = min(end, instance[2])
                    fragment_length = fragment_end - fragment_start
                    if fragment_length / (instance[2] - instance[1]) > include_threshold:
                        contains_instance = True
                        fragment_start = (fragment_start - start) / self.duration_sec
                        fragment_end = (fragment_end - start) / self.duration_sec # now we're relative to sample duration
                        midpoint = (fragment_start + fragment_end) / 2
                        grid_idx = int(midpoint // self.grid_w)
                        midpoint_remainder = midpoint % self.grid_w
                        anchors_moved = [(midpoint - (0.5 * anchor), midpoint + (0.5 * anchor)) for anchor in self.anchors]
                        ious = [self.iou_start_end((fragment_start, fragment_end), anchor) for anchor in anchors_moved]
                        iou_rank = self.argsort(ious)
                        for anchor_idx in iou_rank:
                            if gt[anchor_idx, grid_idx, 0] == 1: continue
                            gt[anchor_idx, grid_idx, 0] = 1
                            gt[anchor_idx, grid_idx, 1] = midpoint_remainder / self.grid_w
                            gt[anchor_idx, grid_idx, 2] = (fragment_end - fragment_start) / self.anchors[anchor_idx]
                            gt[anchor_idx, grid_idx, 3] = self.motif2idx[instance[0]]
                            break
                
                if contains_instance:
                    self.samples.append((version, act, start*22050, end*22050, gt))
                else:
                    self.none_samples.append((version, act, start*22050, end*22050, gt))
        
        print("Shuffling and truncating none samples...")
        random.shuffle(self.none_samples)

        # Create none sample index lookup table
        self.none_samples_by_version = {}
        for version in self.train_versions + self.valid_versions:
            self.none_samples_by_version[version] = [idx for (idx, x) in enumerate(
                self.none_samples) if x[0] == version]
        self.none_samples_by_act = {}
        for act in self.train_acts + self.valid_acts:
            self.none_samples_by_act[act] = [idx for (idx, x) in enumerate(
                self.none_samples) if x[1] == act]
            
    def enable_augmentations(self):
        self.cur_mixup_prob = self.mixup_prob
        self.cur_pitchshift_prob = self.pitchshift_prob
    
    def disable_augmentations(self):
        self.cur_mixup_prob = 0
        self.cur_pitchshift_prob = 0

    def apply_augmentations(self, wav, version, act):
        if random.random() < self.cur_mixup_prob:
            mixup_idx = 0
            if self.split == "version":
                mixup_idx = random.choice(self.none_samples_by_version[version[0]])
            else:
                mixup_idx = random.choice(self.none_samples_by_act[act("_")[1]])
            mixup_fn, mixup_start, mixup_end, _ = self.none_samples[mixup_idx]
            mixup_wav = self.wavs[mixup_fn][mixup_start:mixup_end]
            wav = (1 - self.mixup_alpha) * wav + self.mixup_alpha * mixup_wav
        
        if random.random() < self.cur_pitchshift_prob:
            semitones = 0
            while semitones == 0:
                semitones = random.randint(-self.pitchshift_semitones, self.pitchshift_semitones) * 0.5
            wav = torchaudio.functional.pitch_shift(wav, 22050, semitones, n_fft=2048)
        return wav
            
    def iou_start_end(self, b1:tuple, b2:tuple):
        start1, end1 = b1
        start2, end2 = b2
        intersection = max(0, min(end1, end2) - max(start1, start2))
        union = (end1 - start1) + (end2 - start2) - intersection
        return intersection / (union + 1e-16)
    
    def argsort(self, seq):
        return sorted(range(len(seq)), key=lambda x: seq[x], reverse=True)
    
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
            return samples, none_samples
        elif acts is None:
            samples = [idx for (idx, x) in enumerate(
                self.samples) if x[0] in versions]
            none_samples = [idx + len(self.samples) for (idx, x)
                            in enumerate(self.none_samples) if x[0] in versions]
            return samples, none_samples
        else:
            samples = [idx for (idx, x) in enumerate(
                self.samples) if x[0] in versions and x[1] in acts]
            none_samples = [idx + len(self.samples) for (idx, x) in enumerate(
                self.none_samples) if x[0] in versions and x[1] in acts]
            return samples, none_samples
        
    def __len__(self):
        return len(self.samples) + len(self.none_samples)
    
    def __getitem__(self, idx):
        if idx < len(self.samples):
            version, act, start, end, gt = self.samples[idx]
            wav = self.wavs[f"{version}_{act}"][start:end]
            if random.random() < self.cur_mixup_prob:
                mixup_idx = 0
                if self.split == "version":
                    mixup_idx = random.choice(self.none_samples_by_version[version[0]])
                else:
                    mixup_idx = random.choice(self.none_samples_by_act[act("_")[1]])
                mixup_fn, mixup_start, mixup_end, _ = self.none_samples[mixup_idx]
                mixup_wav = self.wavs[mixup_fn][mixup_start:mixup_end]
                wav = (1 - self.mixup_alpha) * wav + self.mixup_alpha * mixup_wav
            if self.eval:
                frame_start, frame_end = start // 512, end // 512
                framewise_gt = self.framewise_gts[f"{version}_{act}"][frame_start:frame_end]
                return wav, gt, framewise_gt
            else:
                return wav, gt
        else:
            idx -= len(self.samples)
            version, act, start, end, gt = self.none_samples[idx]
            wav = self.wavs[f"{version}_{act}"][start:end]
            if self.eval:
                frame_start, frame_end = start // 512, end // 512
                framewise_gt = self.framewise_gts[f"{version}_{act}"][frame_start:frame_end]
                return wav, gt, framewise_gt
            else:
                return wav, gt


if __name__ == "__main__":
    dummy_anchors = [0.1, 0.2, 0.5]
    TRAIN_VERSIONS = ['Bo', 'Bh', 'Fu', 'Ja', 'Ke', 'Kr', 'Le', 'Ne', 'Sw', 'Sa', 'Th']
    VALID_VERSIONS = ['Ka', 'Sw']
    TEST_VERSIONS = ['We']
    TRAIN_ACTS = ['A', 'B-1', 'B-2', 'B-3', 'C-1', 'C-2', 'C-3', 'D-0', 'D-2']
    VALID_ACTS = ['D-1', 'D-3']

    dataset = YOLODataset(
        Path("./wav-22050"),
        Path("./LeitmotifOccurrencesInstances/Instances"),
        TRAIN_VERSIONS,
        VALID_VERSIONS,
        TRAIN_ACTS,
        VALID_ACTS,
        [
            'Nibelungen',
            'Ring',
            'Nibelungenhass',
            'Ritt',
            'Waldweben',
            'Waberlohe',
            'Horn',
            'Schwert',
            'Walhall-b',
            'Feuerzauber',
            'Unmuth',
            'Siegfried',
            'Vertrag'
        ],
        dummy_anchors
    )
    print(dataset.samples[200])
    print(len(dataset.samples))
    print(len(dataset.none_samples))