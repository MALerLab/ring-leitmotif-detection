from pathlib import Path
import random
from copy import deepcopy
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import torch
from data import YOLODataset, Subset, collate_fn, get_binary_f1
from modules import YOLO, nms, get_iou, grid_to_absolute
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP
import constants as C

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

@hydra.main(config_path="config", config_name="yolo_config", version_base=None)
def main(cfg: DictConfig):
    DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
    random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    if cfg.debug:
        torch.autograd.set_detect_anomaly(True)
        c = {
            "TRAIN_VERSIONS": ["Bo"],
            "VALID_VERSIONS": ["Ka"],
            "TRAIN_ACTS": ["A"],
            "VALID_ACTS": ["D-1"],
        }
        constants = OmegaConf.create(c)
    else:
        constants = C

    base_set = YOLODataset(
        Path(cfg.dataset.wav_dir), 
        Path("data/LeitmotifOccurrencesInstances/Instances"),
        constants.TRAIN_VERSIONS,
        constants.VALID_VERSIONS,
        constants.TRAIN_ACTS,
        constants.VALID_ACTS,
        C.MOTIFS,
        C.ANCHORS,
        overlap_sec = 0,
        include_threshold = 0.5,
        max_none_samples = 0,
        split = cfg.dataset.split,
        mixup_prob = 0,
        mixup_alpha = 0,
        pitchshift_prob = 0,
        pitchshift_semitones = 0,
        eval=True,
        device = DEV
    )

    valid_set = None
    if cfg.dataset.split == "version":
        valid_set = Subset(base_set, base_set.get_subset_idxs(versions=constants.VALID_VERSIONS))
    elif cfg.dataset.split == "act":
        valid_set = Subset(base_set, base_set.get_subset_idxs(acts=constants.VALID_ACTS))
    else:
        raise ValueError("Invalid split method")
    
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size = cfg.batch_size,
        shuffle = False,
        collate_fn = collate_fn
    )

    model = YOLO(
        num_anchors=len(C.ANCHORS),
        C=len(C.MOTIFS),
        base_hidden=cfg.model.base_hidden,
        dropout=cfg.model.dropout
    )
    model.to(DEV)
    ckpt = torch.load(cfg.eval.checkpoint)
    model.load_state_dict(ckpt["model"])

    targets = []
    preds = []

    anchors = torch.tensor(C.ANCHORS).to(DEV)
    anchors = anchors.reshape(1, 3, 1, 1)
    print("Starting inference...")
    with torch.inference_mode():
        for batch in tqdm(valid_loader, ascii=True):
            wav, gt = batch
            pred = model(wav.to(DEV))
            gt = gt.to(DEV)
            pred = nms(pred, torch.tensor(C.ANCHORS).to(DEV))

            gt[..., 1:2] = grid_to_absolute(gt[..., 1:2], batched=True)
            gt[..., 2:3] = gt[..., 2:3] * anchors

            for batch_idx in range(gt.shape[0]):
                framewise_gt = torch.zeros(200, base_set.num_classes)
                for t in gt[batch_idx]:
                    t_boxes = t[t[..., 0] == 1]
                    for i in range(t_boxes.shape[0]):
                        conf, x, w, c = t_boxes[i].tolist()
                        x_start = int(round((x - w / 2) * 200))
                        x_end = int(round((x + w / 2) * 200))
                        framewise_gt[x_start:x_end, int(c)] += conf
                targets.append(framewise_gt)

            for batch_idx, p_boxes in enumerate(pred):
                framewise_pred = torch.zeros(200, base_set.num_classes)
                for p_box in p_boxes:
                    conf, x, w, c = p_box
                    x_start = int(round((x - w / 2) * 200))
                    x_end = int(round((x + w / 2) * 200))
                    framewise_pred[x_start:x_end, int(c)] += conf
                preds.append(framewise_pred)

    targets = torch.cat(targets, dim=0)
    preds = torch.cat(preds, dim=0)
    preds /= preds.max()

    for i in range(preds.shape[1]):
        preds[:, i] = medfilt(preds[:, i])

    print("Performing threshold grid search...")
    # Threshold grid search
    thresholds = [x * 0.01 for x in range(1, 100, 1)]
    best_f1 = [0 for _ in range(base_set.num_classes + 1)]
    best_threshold = [0 for _ in range(base_set.num_classes + 1)]
    best_precision = [0 for _ in range(base_set.num_classes + 1)]
    best_recall = [0 for _ in range(base_set.num_classes + 1)]
    for threshold in tqdm(thresholds, leave=False, ascii=True):
        for i in range(base_set.num_classes):
            f1, precision, recall = get_binary_f1(preds[:, i], targets[:, i], threshold)
            if f1 > best_f1[i]:
                best_f1[i] = f1
                best_threshold[i] = threshold
                best_precision[i] = precision
                best_recall[i] = recall
        # Get matrix mean
        f1, precision, recall = get_binary_f1(preds, targets, threshold)
        if f1 > best_f1[-1]:
            best_f1[-1] = f1
            best_threshold[-1] = threshold
            best_precision[-1] = precision
            best_recall[-1] = recall

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"=========== Evaluation Results ============")
    with open(f"inference-log-{now}.txt", "w") as f:
        f.write(f"Split: {cfg.dataset.split}\n")
        f.write(f"Checkpoint: {cfg.eval.checkpoint}\n")
        f.write(f"=========== Evaluation Results ============\n")
        for i in range(base_set.num_classes):
            f.write(f"{base_set.idx2motif[i]:>{16}} | P: {best_precision[i]:.3f}, R: {best_recall[i]:.3f}, F1: {best_f1[i]:.3f}, Threshold: {best_threshold[i]:.3f}\n")
            print(f"{base_set.idx2motif[i]:>{16}} | P: {best_precision[i]:.3f}, R: {best_recall[i]:.3f}, F1: {best_f1[i]:.3f}, Threshold: {best_threshold[i]:.3f}")
        f.write(f"{'Matrix Mean':>{16}} | P: {best_precision[-1]:.3f}, R: {best_recall[-1]:.3f}, F1: {best_f1[-1]:.3f}, Threshold: {best_threshold[-1]:.3f}\n")
        print(f"{'Matrix Mean':>{16}} | P: {best_precision[-1]:.3f}, R: {best_recall[-1]:.3f}, F1: {best_f1[-1]:.3f}, Threshold: {best_threshold[-1]:.3f}\n")

if __name__ == "__main__":
    main()


