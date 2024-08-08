from pathlib import Path
import datetime
import torch
import pandas as pd
from nnAudio.features.cqt import CQT1992v2
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm
from data.dataset import OTFDataset
from data.data_utils import get_binary_f1, id2version, idx2motif
from modules import CNNModel, FiLMModel, FiLMAttnModel, CNNAttnModel, BBoxModel
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

def infer_film(model, cqt, duration_samples=646, overlap=236, num_classes=21):
    increment = duration_samples - overlap
    leitmotif_out = torch.zeros((cqt.shape[0], num_classes))
    for motif in tqdm(range(num_classes), leave=False, ascii=True):
        for i in tqdm(range(0, cqt.shape[0], increment), leave=False, ascii=True):
            x = cqt[i:i+duration_samples, :]
            if x.shape[0] <= overlap//2:
                break
            x = x.unsqueeze(0)
            cnn_out = model.cnn_forward(x, torch.tensor([motif]))
            leitmotif_pred = model.proj(cnn_out).sigmoid().squeeze(-1)

            # target and source slice positions
            t_start = i + (overlap//2)
            t_end = min(i + duration_samples - (overlap//2), cqt.shape[0])
            s_start = overlap//2
            s_end = duration_samples - (overlap//2) if t_end < cqt.shape[0] else None

            leitmotif_out[t_start:t_end, motif] = leitmotif_pred[0, s_start:s_end]
    return leitmotif_out

def infer_filmattn(model, cqt, duration_samples=646, overlap=236, num_classes=21):
    increment = duration_samples - overlap
    leitmotif_out = torch.zeros((cqt.shape[0], num_classes)).to(cqt.device)
    for motif in tqdm(range(num_classes), leave=False, ascii=True):
        for i in tqdm(range(0, cqt.shape[0], increment), leave=False, ascii=True):
            x = cqt[i:i+duration_samples, :]
            if x.shape[0] <= overlap//2:
                break
            x = x.unsqueeze(0)
            cnn_out = model.cnn_forward(x, torch.tensor([motif]).to(cqt.device))
            cnn_out = cnn_out + model.pos_enc(cnn_out)
            label_emb = model.film_gen.emb(torch.tensor([motif]).to(cqt.device))
            label_emb += model.bias_for_emb.unsqueeze(0)
            cat_emb = torch.cat([label_emb.unsqueeze(1), cnn_out], dim=1)
            encoder_out = model.encoder(cat_emb)
            leitmotif_pred = model.proj(encoder_out).sigmoid()[:, 1:].squeeze(-1)

            # target and source slice positions
            t_start = i + (overlap//2)
            t_end = min(i + duration_samples - (overlap//2), cqt.shape[0])
            s_start = overlap//2
            s_end = duration_samples - (overlap//2) if t_end < cqt.shape[0] else None

            leitmotif_out[t_start:t_end, motif] = leitmotif_pred[0, s_start:s_end]
    return leitmotif_out

def infer_cnnattn(model, cqt, duration_samples=646, overlap=236, num_classes=21):
    increment = duration_samples - overlap
    leitmotif_out = torch.zeros((cqt.shape[0], num_classes)).to(cqt.device)
    for i in tqdm(range(0, cqt.shape[0], increment), leave=False, ascii=True):
        x = cqt[i:i+duration_samples, :]
        if x.shape[0] <= overlap//2:
            break
        x = x.unsqueeze(0)
        cnn_out = model.stack(x)
        cnn_out = cnn_out + model.pos_enc(cnn_out)
        encoder_out = model.encoder(cnn_out)
        leitmotif_pred = model.proj(encoder_out).sigmoid()

        # target and source slice positions
        t_start = i + (overlap//2)
        t_end = min(i + duration_samples - (overlap//2), cqt.shape[0])
        s_start = overlap//2
        s_end = duration_samples - (overlap//2) if t_end < cqt.shape[0] else None

        leitmotif_out[t_start:t_end] = leitmotif_pred[0, s_start:s_end]
    return leitmotif_out

def infer_bbox(model, cqt, duration_samples=646, overlap=440, num_classes=21):
    increment = duration_samples - overlap
    leitmotif_out = torch.zeros((cqt.shape[0], num_classes)).to(cqt.device)
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
        class_pred = model.class_proj(out).sigmoid().squeeze(0)
        bbox_pred = model.bbox_proj(out)
        bbox_pred = bbox_pred.reshape(bbox_pred.shape[0], -1, 2).squeeze(0)
        bbox_pred = torch.round(bbox_pred)

        for j in range(bbox_pred.shape[0]):
            if bbox_pred[j, :].max() < 0:
                continue
            start, end = bbox_pred[j, :]
            start = int(torch.clamp(start, 0, duration_samples).item())
            end = int(torch.clamp(end, 0, duration_samples).item())
            leitmotif_out[i+start:i+end, j] += class_pred[j]
        
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

@hydra.main(config_path="config", config_name="inference_config", version_base=None)
def main(config: DictConfig):
    cfg = config.cfg
    hyperparams = config.hyperparams
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = OTFDataset(Path("data/wav-22050"),
                         Path("data/LeitmotifOccurrencesInstances/Instances"),
                         include_none_class=hyperparams.include_none_class,
                         mixup_prob=0,
                         mixup_alpha=0,
                         device=DEV)

    files = []
    wavs = {}
    instances_gts = {}
    if cfg.split == "version":
        wavs = {k: v for k, v in dataset.wavs.items() if k.split("_")[0] in C.VALID_VERSIONS}
        instances_gts = {k: v for k, v in dataset.instances_gts.items() if k.split("_")[0] in C.VALID_VERSIONS}
        files = [k for k in wavs.keys()]
    elif cfg.split == "act":
        wavs = {k: v for k, v in dataset.wavs.items() if k.split("_")[1] in C.VALID_ACTS}
        instances_gts = {k: v for k, v in dataset.instances_gts.items() if k.split("_")[1] in C.VALID_ACTS}
        files = [k for k in wavs.keys()]
    transform = CQT1992v2().to(DEV)

    model = None
    if cfg.model == "CNN":
        model = CNNModel(num_classes=dataset.num_classes)
    elif cfg.model == "FiLM":
        model = FiLMModel(num_classes=dataset.num_classes,
                          filmgen_emb=hyperparams.filmgen_emb,
                          filmgen_hidden=hyperparams.filmgen_hidden)
    elif cfg.model == "FiLMAttn":
        model = FiLMAttnModel(num_classes=dataset.num_classes,
                              filmgen_emb=hyperparams.filmgen_emb,
                              filmgen_hidden=hyperparams.filmgen_hidden,
                              attn_dim=hyperparams.attn_dim,
                              attn_depth=hyperparams.attn_depth,
                              attn_heads=hyperparams.attn_heads)
    elif cfg.model == "CNNAttn":
        model = CNNAttnModel(num_classes=dataset.num_classes,
                             attn_dim=hyperparams.attn_dim,
                             attn_depth=hyperparams.attn_depth,
                             attn_heads=hyperparams.attn_heads)
    elif cfg.model == "BBox":
        model = BBoxModel(num_classes=dataset.num_classes,
                          apply_attn=hyperparams.apply_attn,
                          attn_dim=hyperparams.attn_dim,
                          attn_depth=hyperparams.attn_depth,
                          attn_heads=hyperparams.attn_heads)
    else:
        raise ValueError("Invalid model name")
    model.load_state_dict(torch.load(cfg.load_checkpoint)["model"], strict=False)
    model.to(DEV)
    model.eval()

    print(f'Inferring on {len(files)} files with {cfg.model}.')
    leitmotif_preds = []
    leitmotif_gts = []
    with torch.inference_mode():
        for fn in tqdm(files, ascii=True):
            wav = wavs[fn]
            cqt = transform(wav.to(DEV)).squeeze(0)
            cqt = (cqt / cqt.max()).T
            if cfg.model == "CNN":
                leitmotif_pred = infer_cnn(model, cqt, num_classes=dataset.num_classes)
            elif cfg.model == "FiLM":
                leitmotif_pred = infer_film(model, cqt, num_classes=dataset.num_classes)
            elif cfg.model == "FiLMAttn":
                leitmotif_pred = infer_filmattn(model, cqt, num_classes=dataset.num_classes)
            elif cfg.model == "CNNAttn":
                leitmotif_pred = infer_cnnattn(model, cqt, num_classes=dataset.num_classes)
            elif cfg.model == "BBox":
                leitmotif_pred = infer_bbox(model, cqt, num_classes=dataset.num_classes)

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
                leitmotif_gt[i, :] = instances_gts[fn][start:end, :].max(dim=0).values

            leitmotif_preds.append(leitmotif_out)
            leitmotif_gts.append(leitmotif_gt)
    
    leitmotif_preds = torch.cat(leitmotif_preds, dim=0)
    leitmotif_gts = torch.cat(leitmotif_gts, dim=0)

    torch.save(leitmotif_preds, "bbox-pred.pt")
    torch.save(leitmotif_gts, "bbox-gt.pt")

    # Maximum filtering
    pool = torch.nn.MaxPool1d(kernel_size=cfg.discretization, stride=1, padding=cfg.discretization//2)
    leitmotif_preds = pool(leitmotif_preds.T).T
    leitmotif_gts = pool(leitmotif_gts.T).T

    torch.save(leitmotif_preds, "bbox-pred-max.pt")
    torch.save(leitmotif_gts, "bbox-gt-max.pt")

    print("Performing threshold grid search...")
    # Threshold grid search
    thresholds = [x * 0.001 for x in range(1, 1000, 1)]
    best_f1 = [0 for _ in range(dataset.num_classes + 1)]
    best_threshold = [0 for _ in range(dataset.num_classes + 1)]
    best_precision = [0 for _ in range(dataset.num_classes + 1)]
    best_recall = [0 for _ in range(dataset.num_classes + 1)]
    for threshold in tqdm(thresholds, leave=False, ascii=True):
        for i in range(dataset.num_classes):
            f1, precision, recall = get_binary_f1(leitmotif_preds[:, i], leitmotif_gts[:, i], threshold)
            if f1 > best_f1[i]:
                best_f1[i] = f1
                best_threshold[i] = threshold
                best_precision[i] = precision
                best_recall[i] = recall
        # Get matrix mean
        f1, precision, recall = get_binary_f1(leitmotif_preds, leitmotif_gts, threshold)
        if f1 > best_f1[-1]:
            best_f1[-1] = f1
            best_threshold[-1] = threshold
            best_precision[-1] = precision
            best_recall[-1] = recall

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"=========== Evaluation Results ============")
    with open(f"inference-log-{now}.txt", "w") as f:
        f.write(f"Model: {cfg.model}\n")
        f.write(f"Split: {cfg.split}\n")
        f.write(f"Checkpoint: {cfg.load_checkpoint}\n")
        f.write(f"Number of files: {len(files)}\n")
        f.write(f"=========== Evaluation Results ============\n")
        for i in range(dataset.num_classes if hyperparams.include_none_class else dataset.num_classes-1):
            f.write(f"{idx2motif[i]:>{16}} | P: {best_precision[i]:.3f}, R: {best_recall[i]:.3f}, F1: {best_f1[i]:.3f}, Threshold: {best_threshold[i]:.3f}\n")
            print(f"{idx2motif[i]:>{16}} | P: {best_precision[i]:.3f}, R: {best_recall[i]:.3f}, F1: {best_f1[i]:.3f}, Threshold: {best_threshold[i]:.3f}")
        f.write(f"{'Matrix Mean':>{16}} | P: {best_precision[-1]:.3f}, R: {best_recall[-1]:.3f}, F1: {best_f1[-1]:.3f}, Threshold: {best_threshold[-1]:.3f}\n")
        print(f"{'Matrix Mean':>{16}} | P: {best_precision[-1]:.3f}, R: {best_recall[-1]:.3f}, F1: {best_f1[-1]:.3f}, Threshold: {best_threshold[-1]:.3f}\n")

if __name__ == "__main__":
    main()