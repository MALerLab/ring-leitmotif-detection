from pathlib import Path
import torch
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from dataset import LeitmotifDataset, Subset, collate_fn
from data_utils import get_binary_f1, id2version, idx2motif
from modules import RNNModel, CNNModel
import constants as C
# TODO: add version prediction

def infer_rnn(model, cqt):
    leitmotif_out = torch.zeros((cqt.shape[0], 21))
    singing_out = torch.zeros((cqt.shape[0], 1))
    model.eval()
    with torch.inference_mode():
        hidden = None
        for i, x in enumerate(tqdm(cqt, leave=False)):
            x = x.unsqueeze(0).unsqueeze(0)
            lstm_out, h = model.lstm(x, hidden)
            lstm_out = lstm_out.transpose(1, 2)
            lstm_out = model.batch_norm(lstm_out)
            lstm_out = lstm_out.transpose(1, 2)
            leitmotif_pred = model.proj(lstm_out).sigmoid()
            singing_pred = model.singing_mlp(lstm_out)
            hidden = (h[0].detach(), h[1].detach())

            leitmotif_out[i, :] = leitmotif_pred.squeeze()
            singing_out[i, :] = singing_pred.squeeze()
    return leitmotif_out, singing_out

def infer_cnn(model, cqt, duration_samples=6460, overlap=236):
    increment = duration_samples - overlap
    leitmotif_out = torch.zeros((cqt.shape[0], 21))
    singing_out = torch.zeros((cqt.shape[0], 1))
    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(0, cqt.shape[0], increment), leave=False):
            x = cqt[i:i+duration_samples, :]
            if x.shape[0] <= overlap//2:
                break
            x = x.unsqueeze(0)
            leitmotif_pred, singing_pred, _ = model(x)

            # target and source slice positions
            t_start = i + (overlap//2)
            t_end = min(i + duration_samples - (overlap//2), cqt.shape[0])
            s_start = overlap//2
            s_end = -overlap//2 if t_end < cqt.shape[0] else None

            leitmotif_out[t_start:t_end] = leitmotif_pred[0, s_start:s_end]
            singing_out[t_start:t_end] = singing_pred[0, s_start:s_end]
    return leitmotif_out, singing_out

def medfilt(x, k=21):
    assert x.dim() == 1
    k2 = (k - 1) // 2
    y = torch.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i+1)] = x[j:]
        y[-j:, -(i+1)] = x[-1]
    return torch.median(y, dim=1).values

@hydra.main(config_path="config", config_name="inference_config", version_base=None)
def main(config: DictConfig):
    cfg = config.cfg
    hyperparams = config.hyperparams
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = LeitmotifDataset(Path("data/CQT"),
                                Path("data/LeitmotifOccurrencesInstances/Instances"),
                                Path("data/WagnerRing_Public/02_Annotations/ann_audio_singing"))

    files = []
    cqt = {}
    instances_gt = {}
    singing_gt = {}
    if cfg.split == "version":
        cqt = {k: v for k, v in dataset.cqt.items() if k.split("_")[0] in C.VALID_VERSIONS}
        instances_gt = {k: v for k, v in dataset.instances_gt.items() if k.split("_")[0] in C.VALID_VERSIONS}
        singing_gt = {k: v for k, v in dataset.singing_gt.items() if k.split("_")[0] in C.VALID_VERSIONS}
        files = [k for k in cqt.keys()]
    elif cfg.split == "act":
        cqt = {k: v for k, v in dataset.cqt.items() if k.split("_")[1] in C.VALID_ACTS}
        instances_gt = {k: v for k, v in dataset.instances_gt.items() if k.split("_")[1] in C.VALID_ACTS}
        singing_gt = {k: v for k, v in dataset.singing_gt.items() if k.split("_")[1] in C.VALID_ACTS}
        files = [k for k in cqt.keys()]

    model = None
    if cfg.model == "RNN":
        model = RNNModel()
    elif cfg.model == "CNN":
        model = CNNModel()
    else:
        raise ValueError("Invalid model name")
    model.load_state_dict(torch.load(cfg.load_checkpoint)["model"])
    model.to(DEV)

    print(f'Inferring on {len(files)} files with {cfg.model}.')
    leitmotif_preds = []
    singing_preds = []
    leitmotif_gts = []
    singing_gts = []
    for fn in tqdm(files):
        cqt_fn = cqt[fn]
        cqt_fn = cqt_fn.to(DEV)
        if cfg.model == "RNN":
            leitmotif_pred, singing_pred = infer_rnn(model, cqt_fn)
        elif cfg.model == "CNN":
            leitmotif_pred, singing_pred = infer_cnn(model, cqt_fn)

        # Apply median filter
        for i in range(leitmotif_pred.shape[1]):
            leitmotif_pred[:, i] = medfilt(leitmotif_pred[:, i])
        
        # Subdivide measures
        version = fn.split("_")[0]
        act = fn.split("_")[1]
        measures = pd.read_csv(f"data/WagnerRing_Public/02_Annotations/ann_audio_measure/Wagner_WWV086{act}_{id2version[version]}.csv", sep=';').itertuples(index=False, name=None)
        subdivision_points = []
        prev_rec_start = 0
        prev_measure_idx = 0
        for measure in measures:
            recording_start_sec, measure_idx = measure
            measure_diff = measure_idx - prev_measure_idx
            prev_measure_idx = measure_idx

            diff = recording_start_sec - prev_rec_start
            if prev_rec_start == 0:
                prev_rec_start = recording_start_sec
                continue
            subdiv_length = diff / cfg.discretization
            num_subdiv = round(cfg.discretization * measure_diff)

            for i in range(num_subdiv):
                subdiv_point = round((prev_rec_start + i * subdiv_length) * 22050 / 512)
                subdivision_points.append(subdiv_point)
            prev_rec_start = recording_start_sec

        # Take max over subdivisions
        leitmotif_out = torch.zeros(len(subdivision_points)-1, leitmotif_pred.shape[1])
        leitmotif_gt = torch.zeros(len(subdivision_points)-1, leitmotif_pred.shape[1])
        for i in range(len(subdivision_points)-1):
            start = subdivision_points[i]
            end = subdivision_points[i+1]
            if start == end: continue
            if start > leitmotif_pred.shape[0]: break
            leitmotif_out[i, :] = leitmotif_pred[start:end, :].max(dim=0).values
            leitmotif_gt[i, :] = instances_gt[fn][start:end, :].max(dim=0).values

        leitmotif_preds.append(leitmotif_out)
        singing_preds.append(singing_pred)
        leitmotif_gts.append(leitmotif_gt)
        singing_gts.append(singing_gt[fn])
    
    leitmotif_preds = torch.cat(leitmotif_preds, dim=0)
    singing_preds = torch.cat(singing_preds, dim=0)
    leitmotif_gts = torch.cat(leitmotif_gts, dim=0)
    singing_gts = torch.cat(singing_gts, dim=0)

    print("Performing threshold grid search...")
    # Threshold grid search
    thresholds = list(range(0.1, 0.9, 0.1))
    best_f1 = [0 for _ in range(21)]
    best_threshold = [0 for _ in range(21)]
    precision = [0 for _ in range(21)]
    recall = [0 for _ in range(21)]
    for threshold in tqdm(thresholds, leave=False):
        for i in range(21):
            f1, precision, recall = get_binary_f1(leitmotif_preds[:, i], leitmotif_gts[:, i], threshold)
            if f1 > best_f1[i]:
                best_f1[i] = f1
                best_threshold[i] = threshold
                precision[i] = precision
                recall[i] = recall

    print(f"=== Evaluation Results ===")
    for i in range(20):
        print(f"{idx2motif[i]:>{16}} | P: {precision[i]:3d}, R: {recall[i]}, F1: {best_f1[i]}, Threshold: {best_threshold[i]}")

if __name__ == "__main__":
    main()