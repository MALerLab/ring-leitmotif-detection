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
        Path(cfg.dataset.instances_dir),
        constants.TRAIN_VERSIONS,
        constants.VALID_VERSIONS,
        constants.TRAIN_ACTS,
        constants.VALID_ACTS,
        C.MOTIFS,
        C.ANCHORS,
        use_merged_data=cfg.dataset.use_merged_data,
        overlap_sec = 0,
        include_threshold = 0.5,
        # max_none_samples = 0,
        split = cfg.dataset.split,
        mixup_prob = 0,
        mixup_alpha = 0,
        pitchshift_prob = 0,
        pitchshift_semitones = 0,
        device = DEV
    )

    valid_set = None
    if cfg.dataset.split == "version":
        valid_set = Subset(base_set, base_set.get_subset_idxs(versions=constants.VALID_VERSIONS))
        test_set = Subset(base_set, base_set.get_subset_idxs(versions=constants.TEST_VERSIONS))
    elif cfg.dataset.split == "act":
        valid_set = Subset(base_set, base_set.get_subset_idxs(acts=constants.VALID_ACTS))
        test_set = Subset(base_set, base_set.get_subset_idxs(acts=constants.TEST_ACTS))
    else:
        raise ValueError("Invalid split method")
    
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size = cfg.batch_size,
        shuffle = False,
        collate_fn = collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
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
    model.eval()

    targets = []
    preds = []

    anchors = torch.tensor(C.ANCHORS).to(DEV)
    anchors = anchors.reshape(1, 3, 1, 1)
    print("Starting inference...")
    with torch.inference_mode():
        for batch in tqdm(test_loader, ascii=True):
            wav, gt = batch
            p = model(wav.to(DEV))
            gt = gt.to(DEV)
            gt[..., 1:2] = grid_to_absolute(gt[..., 1:2], batched=True)
            gt[..., 2:3] = gt[..., 2:3] * anchors
            preds.append(p)
            targets.append(gt)

    print("Performing threshold grid search...")
    conf_thresholds = [i * 0.05 for i in range(20)]
    nms_iou_thresholds = [i * 0.05 for i in range(18)]
    iou_threshold = 0.5

    best_f1 = [0 for _ in range(base_set.num_classes + 1)]
    best_thresholds = [[0, 0] for _ in range(base_set.num_classes + 1)]
    best_precision = [0 for _ in range(base_set.num_classes + 1)]
    best_recall = [0 for _ in range(base_set.num_classes + 1)]

    best_thresholds = [
        [0.500, 0.350],
        [0.550, 0.400],
        [0.500, 0.650],
        [0.450, 0.400],
        [0.350, 0.300],
        [0.600, 0.600],
        [0.550, 0.300],
        [0.500, 0.300],
        [0.550, 0.300],
        [0.600, 0.450],
        [0.600, 0.500],
        [0.550, 0.450],
        [0.650, 0.500],
        [0.550, 0.350]
    ]
    for i in range(base_set.num_classes + 1):
        conf_thresh, nms_iou_thresh = best_thresholds[i]
        tp = torch.tensor([0 for _ in range(base_set.num_classes + 1)]).to(DEV)
        fp = torch.tensor([0 for _ in range(base_set.num_classes + 1)]).to(DEV)
        fn = torch.tensor([0 for _ in range(base_set.num_classes + 1)]).to(DEV)
        for pred, gt in zip(preds, targets):
            p = nms(pred, torch.tensor(C.ANCHORS).to(DEV), conf_threshold=conf_thresh, iou_threshold=nms_iou_thresh)
            for b in range(len(pred)):
                t = gt[b][gt[b][..., 0] == 1].tolist()
                checked = [False for _ in range(len(t))]
                for p_box in p[b]:
                    for i, t_box in enumerate(t):
                        if (p_box[3] == t_box[3] and
                            get_iou(
                                torch.tensor(t_box[1:3]).to(DEV), 
                                torch.tensor(p_box[1:3]).to(DEV)
                            ).item() > iou_threshold):
                            if not checked[i]:
                                tp[int(t_box[3])] += 1
                                tp[-1] += 1
                                checked[i] = True
                            break
                    else:
                        fp[int(p_box[3])] += 1
                        fp[-1] += 1
                for i, c in enumerate(checked):
                    if not c:
                        fn[int(t[i][3])] += 1
                        fn[-1] += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        best_f1[i] = f1[i].item()
        best_precision[i] = precision[i].item()
        best_recall[i] = recall[i].item()

    # for conf_thresh in tqdm(conf_thresholds, ascii=True):
    #     for nms_iou_thresh in tqdm(nms_iou_thresholds, ascii=True, leave=False):
    #         tp = torch.tensor([0 for _ in range(base_set.num_classes + 1)]).to(DEV)
    #         fp = torch.tensor([0 for _ in range(base_set.num_classes + 1)]).to(DEV)
    #         fn = torch.tensor([0 for _ in range(base_set.num_classes + 1)]).to(DEV)
    #         for pred, gt in zip(preds, targets):
    #             p = nms(pred, torch.tensor(C.ANCHORS).to(DEV), conf_threshold=conf_thresh, iou_threshold=nms_iou_thresh)
    #             for b in range(len(pred)):
    #                 t = gt[b][gt[b][..., 0] == 1].tolist()
    #                 checked = [False for _ in range(len(t))]
    #                 for p_box in p[b]:
    #                     for i, t_box in enumerate(t):
    #                         if (p_box[3] == t_box[3] and
    #                             get_iou(
    #                                 torch.tensor(t_box[1:3]).to(DEV), 
    #                                 torch.tensor(p_box[1:3]).to(DEV)
    #                             ).item() > iou_threshold):
    #                             if not checked[i]:
    #                                 tp[int(t_box[3])] += 1
    #                                 tp[-1] += 1
    #                                 checked[i] = True
    #                             break
    #                     else:
    #                         fp[int(p_box[3])] += 1
    #                         fp[-1] += 1
    #                 for i, c in enumerate(checked):
    #                     if not c:
    #                         fn[int(t[i][3])] += 1
    #                         fn[-1] += 1

    #         precision = tp / (tp + fp)
    #         recall = tp / (tp + fn)
    #         f1 = 2 * (precision * recall) / (precision + recall)
    #         for i in range(base_set.num_classes + 1):
    #             if f1[i].item() > best_f1[i]:
    #                 best_f1[i] = f1[i].item()
    #                 best_thresholds[i] = [conf_thresh, nms_iou_thresh]
    #                 best_precision[i] = precision[i].item()
    #                 best_recall[i] = recall[i].item()

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"=========== Evaluation Results ============")
    with open(f"inference-log-{now}.txt", "w") as f:
        f.write(f"Method: NMS threshold grid search\n")
        f.write(f"Split: {cfg.dataset.split}\n")
        f.write(f"Checkpoint: {cfg.eval.checkpoint}\n")
        f.write(f"=========== Evaluation Results ============\n")
        for i in range(base_set.num_classes):
            f.write(f"{base_set.idx2motif[i]:>{16}} | P: {best_precision[i]:.3f}, R: {best_recall[i]:.3f}, F1: {best_f1[i]:.3f}, Conf/IOU: {best_thresholds[i][0]:.3f}/{best_thresholds[i][1]}\n")
            print(f"{base_set.idx2motif[i]:>{16}} | P: {best_precision[i]:.3f}, R: {best_recall[i]:.3f}, F1: {best_f1[i]:.3f}, Conf/IOU: {best_thresholds[i][0]:.3f}/{best_thresholds[i][1]}")
        f.write(f"{'Matrix Mean':>{16}} | P: {best_precision[-1]:.3f}, R: {best_recall[-1]:.3f}, F1: {best_f1[-1]:.3f}, Conf/IOU: {best_thresholds[-1][0]:.3f}/{best_thresholds[-1][1]}\n")
        print(f"{'Matrix Mean':>{16}} | P: {best_precision[-1]:.3f}, R: {best_recall[-1]:.3f}, F1: {best_f1[-1]:.3f}, Conf/IOU: {best_thresholds[-1][0]:.3f}/{best_thresholds[-1][1]}\n")

if __name__ == "__main__":
    main()


