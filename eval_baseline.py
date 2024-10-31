from pathlib import Path
import datetime
import torch
import pandas as pd
from nnAudio.features.cqt import CQT1992v2
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm
from data.dataset import FramewiseDataset
from data.data_utils import get_binary_f1, id2version
from modules import CNNModel
import constants as C

def infer_cnn(model, cqt, duration_samples=6460, overlap=236, num_classes=21):
    increment = duration_samples - overlap
    leitmotif_out = torch.zeros((cqt.shape[0], num_classes))
    for i in tqdm(range(0, cqt.shape[0], increment), leave=False, ascii=True):
        x = cqt[i:i+duration_samples, :]
        if x.shape[0] <= overlap//2:
            break
        x = x.unsqueeze(0)
        cnn_out = model.stack(x)
        leitmotif_pred = model.proj(cnn_out).sigmoid()

        # target and source slice positions
        t_start = i + (overlap//2)
        t_end = min(i + duration_samples - (overlap//2), cqt.shape[0])
        s_start = overlap//2
        s_end = duration_samples - (overlap//2) if t_end < cqt.shape[0] else None

        leitmotif_out[t_start:t_end] = leitmotif_pred[0, s_start:s_end]
    return leitmotif_out


def infer_bbox(model, cqt, duration_samples=646, overlap=440, num_classes=21):
    increment = duration_samples - overlap
    leitmotif_out = torch.zeros((cqt.shape[0], num_classes))
    for i in tqdm(range(0, cqt.shape[0], increment), leave=False, ascii=True):
        x = cqt[i:i+duration_samples, :]
        if x.shape[0] <= overlap//2:
            break
        elif x.shape[0] < duration_samples:
            x = torch.cat([x, torch.zeros(duration_samples - x.shape[0], x.shape[1]).to(x.device)], dim=0)
        x = x.unsqueeze(0)

        out = model.stack(x)
        if model.apply_attn:
            out = out + model.pos_enc(out)
            out = model.encoder(out)
        out = out.flatten(1, 2)
        bbox_pred = model.proj(out)
        bbox_pred = bbox_pred.reshape(bbox_pred.shape[0], -1, 2).squeeze(0)
        bbox_pred = torch.round(bbox_pred)

        for j in range(bbox_pred.shape[0]):
            if bbox_pred[j, :].max() < 0:
                continue
            start, end = bbox_pred[j, :]
            start = int(torch.clamp(start, 0, duration_samples).item())
            end = int(torch.clamp(end, 0, duration_samples).item())
            leitmotif_out[i+start:i+end, j] += 1
        
    leitmotif_out = leitmotif_out / leitmotif_out.max()
    return leitmotif_out


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

@hydra.main(config_path="config", config_name="baseline_config", version_base=None)
def main(cfg: DictConfig):
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = FramewiseDataset(
        Path("data/wav-22050"),
        Path("data/LeitmotifOccurrencesInstances/Instances"),
        C.TRAIN_VERSIONS,
        C.VALID_VERSIONS,
        C.TRAIN_ACTS,
        C.VALID_ACTS,
        C.MOTIFS,
        include_none_class=True,
        mixup_prob=0,
        mixup_alpha=0,
        device=DEV
    )

    files = []
    wavs = {}
    instances_gts = {}
    if cfg.dataset.split == "version":
        wavs = {k: v for k, v in dataset.wavs.items() if k.split("_")[0] in C.VALID_VERSIONS}
        instances_gts = {k: v for k, v in dataset.instances_gts.items() if k.split("_")[0] in C.VALID_VERSIONS}
        files = [k for k in wavs.keys()]
    elif cfg.dataset.split == "act":
        wavs = {k: v for k, v in dataset.wavs.items() if k.split("_")[1] in C.VALID_ACTS}
        instances_gts = {k: v for k, v in dataset.instances_gts.items() if k.split("_")[1] in C.VALID_ACTS}
        files = [k for k in wavs.keys()]
    transform = CQT1992v2().to(DEV)

    model = None
    if cfg.model.architecture == "CNN":
        model = CNNModel(
            num_classes=dataset.num_classes,
            base_hidden=cfg.model.base_hidden,
            dropout=cfg.model.dropout
        )
    else:
        raise ValueError("Invalid model name")
    model.load_state_dict(torch.load(cfg.eval.checkpoint, weights_only=True)["model"], strict=False)
    model.to(DEV)
    model.eval()

    print(f'Inferring on {len(files)} files with {cfg.model.architecture}.')
    leitmotif_preds = []
    leitmotif_gts = []
    with torch.inference_mode():
        for fn in tqdm(files, ascii=True):
            wav = wavs[fn]
            cqt = transform(wav.to(DEV)).squeeze(0)
            cqt = (cqt / cqt.max()).T
            leitmotif_pred = infer_cnn(model, cqt, num_classes=dataset.num_classes)
            gt = instances_gts[fn]

            # # Apply median filter
            # for i in range(leitmotif_pred.shape[1]):
            #     leitmotif_pred[:, i] = medfilt(leitmotif_pred[:, i])
            
            # # Subdivide measures
            # version = fn.split("_")[0]
            # act = fn.split("_")[1]
            # measures = pd.read_csv(f"data/WagnerRing_Public/02_Annotations/ann_audio_measure/Wagner_WWV086{act}_{id2version[version]}.csv", sep=';').itertuples(index=False, name=None)
            # subdivision_points = []
            # prev_rec_start = 0
            # prev_measure_idx = 0
            # for measure in measures:
            #     recording_start_sec, measure_idx = measure
            #     measure_diff = measure_idx - prev_measure_idx
            #     prev_measure_idx = measure_idx

            #     diff = recording_start_sec - prev_rec_start
            #     if prev_rec_start == 0:
            #         prev_rec_start = recording_start_sec
            #         continue
            #     subdiv_length = diff / cfg.eval.discretization
            #     num_subdiv = round(cfg.eval.discretization * measure_diff)

            #     for i in range(num_subdiv):
            #         subdiv_point = round((prev_rec_start + i * subdiv_length) * 22050 / 512)
            #         subdivision_points.append(subdiv_point)
            #     prev_rec_start = recording_start_sec

            # # Take max over subdivisions
            # leitmotif_out = torch.zeros(len(subdivision_points)-1, leitmotif_pred.shape[1])
            # leitmotif_gt = torch.zeros(len(subdivision_points)-1, leitmotif_pred.shape[1])
            # for i in range(len(subdivision_points)-1):
            #     start = subdivision_points[i]
            #     end = subdivision_points[i+1]
            #     if start == end: continue
            #     if start > leitmotif_pred.shape[0]: break
            #     leitmotif_out[i, :] = leitmotif_pred[start:end, :].max(dim=0).values
            #     leitmotif_gt[i, :] = instances_gts[fn][start:end, :].max(dim=0).values

            leitmotif_preds.append(leitmotif_pred)
            leitmotif_gts.append(gt)
    
    leitmotif_preds = torch.cat(leitmotif_preds, dim=0)
    leitmotif_gts = torch.cat(leitmotif_gts, dim=0)

    # Maximum filtering
    # pool = torch.nn.MaxPool1d(
    #     kernel_size=cfg.eval.discretization,
    #     stride=1,
    #     padding=cfg.eval.discretization//2
    # )
    # leitmotif_preds = pool(leitmotif_preds.T).T
    # leitmotif_gts = pool(leitmotif_gts.T).T

    print("Performing threshold grid search...")
    # Threshold grid search
    thresholds = [x * 0.01 for x in range(1, 100, 1)]
    best_f1 = [0 for _ in range(dataset.num_classes)]
    best_threshold = [0 for _ in range(dataset.num_classes)]
    best_precision = [0 for _ in range(dataset.num_classes)]
    best_recall = [0 for _ in range(dataset.num_classes)]
    for threshold in tqdm(thresholds, leave=False, ascii=True):
        for i in range(dataset.num_classes - 1):
            f1, precision, recall = get_binary_f1(leitmotif_preds[:, i], leitmotif_gts[:, i], threshold)
            if f1 > best_f1[i]:
                best_f1[i] = f1
                best_threshold[i] = threshold
                best_precision[i] = precision
                best_recall[i] = recall
        # Get matrix mean
        f1, precision, recall = get_binary_f1(leitmotif_preds[:-1, :], leitmotif_gts[:-1, :], threshold)
        if f1 > best_f1[-1]:
            best_f1[-1] = f1
            best_threshold[-1] = threshold
            best_precision[-1] = precision
            best_recall[-1] = recall

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"=========== Evaluation Results ============")
    with open(f"inference-log-{now}.txt", "w") as f:
        f.write(f"Model: {cfg.model.architecture}\n")
        f.write(f"Split: {cfg.dataset.split}\n")
        f.write(f"Checkpoint: {cfg.eval.checkpoint}\n")
        f.write(f"Number of files: {len(files)}\n")
        f.write(f"=========== Evaluation Results ============\n")
        for i in range(dataset.num_classes-1):
            f.write(f"{C.MOTIFS[i]:>{16}} | P: {best_precision[i]:.3f}, R: {best_recall[i]:.3f}, F1: {best_f1[i]:.3f}, Threshold: {best_threshold[i]:.3f}\n")
            print(f"{C.MOTIFS[i]:>{16}} | P: {best_precision[i]:.3f}, R: {best_recall[i]:.3f}, F1: {best_f1[i]:.3f}, Threshold: {best_threshold[i]:.3f}")
        f.write(f"{'Matrix Mean':>{16}} | P: {best_precision[-1]:.3f}, R: {best_recall[-1]:.3f}, F1: {best_f1[-1]:.3f}, Threshold: {best_threshold[-1]:.3f}\n")
        print(f"{'Matrix Mean':>{16}} | P: {best_precision[-1]:.3f}, R: {best_recall[-1]:.3f}, F1: {best_f1[-1]:.3f}, Threshold: {best_threshold[-1]:.3f}\n")

if __name__ == "__main__":
    main()