from pathlib import Path
import random
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import wandb
from data import YOLODataset, Subset, collate_fn
from modules import YOLO, YOLOLoss, nms, get_acc
import constants as C


class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        dataset, 
        train_loader, 
        valid_loader, 
        device, 
        cfg,
        log_to_wandb
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.cfg = cfg
        self.loss = YOLOLoss(
            torch.tensor(C.ANCHORS).to(device),
            lambda_class = cfg.loss.lambda_class,
            lambda_noobj = cfg.loss.lambda_noobj,
            lambda_obj = cfg.loss.lambda_obj,
            lambda_coord = cfg.loss.lambda_coord,
        )
        self.cur_epoch = 0
        self.ckpt_dir = Path(f"checkpoints/YOLO")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.patience = cfg.trainer.patience
        self.log_to_wandb = log_to_wandb
    
    def save_checkpoint(self, ckpt_path):
        ckpt = {
            "epoch": self.cur_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(ckpt, ckpt_path)

    def load_checkpoint(self):
        ckpt = torch.load(self.cfg.load_checkpoint)
        self.cur_epoch = ckpt["epoch"]
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def randomize_none_samples(self, gt):
        label_sum = gt.sum(dim=1)
        labels = label_sum.argmax(dim=-1)
        is_non = label_sum.max(dim=-1).values == 0
        random_label = torch.randint(0, gt.shape[-1], (sum(is_non),)).to(self.device)
        labels[is_non] = random_label
        return labels
    
    def step(self, batch):
        wav, gt = batch
        wav = wav.to(self.device)
        gt = gt.to(self.device)
        pred = self.model(wav)
        loss, loss_dict = self.loss(pred, gt)

        suppressed_pred = nms(pred, torch.tensor(C.ANCHORS).to(self.device))
        acc = get_acc(suppressed_pred, gt, torch.tensor(C.ANCHORS).to(self.device))
        return loss, loss_dict, acc

    def train(self):
        if self.log_to_wandb:
            wandb.init(
                project=self.cfg.trainer.wandb.project,
                entity=self.cfg.trainer.wandb.entity,
                name=self.cfg.trainer.wandb.run_name, 
                config={**OmegaConf.to_container(self.cfg)}
            )
        
        self.model.to(self.device)
        self.loss.to(self.device)
        num_iter = 0
        best_valid_loss = float("inf")
        best_ckpt_path = None
        last_ckpt_path = None
        for epoch in tqdm(range(self.cur_epoch, self.cfg.trainer.num_epochs), ascii=True):
            if self.log_to_wandb:
                wandb.log({"epoch": epoch}, step=num_iter)
            self.cur_epoch = epoch
            self.model.train()
            self.dataset.enable_mixup()
            total_acc = 0
            num_total = 0
            for batch in tqdm(self.train_loader, leave=False, ascii=True):
                loss, loss_dict, acc = self.step(batch)
                if acc != -1:
                    total_acc += acc
                    num_total += 1
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                if self.log_to_wandb:
                    wandb.log({"train/loss": loss.item()}, step=num_iter)
                    wandb.log({f"train/{k}": v.item() for k, v in loss_dict.items()}, step=num_iter)
                num_iter += 1

            avg_acc = total_acc / num_total
            if self.log_to_wandb:
                wandb.log({"train/acc": avg_acc}, step=num_iter)

            self.model.eval()
            self.dataset.disable_mixup()
            with torch.inference_mode():
                total_loss = 0
                total_loss_items = [0, 0, 0, 0]
                total_acc = 0
                num_total = 0
                for batch in tqdm(self.valid_loader, leave=False, ascii=True):
                    loss, loss_dict, acc = self.step(batch)
                    if acc != -1:
                        total_acc += acc
                        num_total += 1
                    total_loss += loss.item()
                    total_loss_items = [a + b.item() for a, b in zip(total_loss_items, loss_dict.values())]
                    total_acc += acc

                avg_loss = total_loss / num_total
                avg_loss_items = [a / num_total for a in total_loss_items]
                avg_acc = total_acc / num_total

                if self.log_to_wandb:
                    wandb.log({"valid/loss": avg_loss}, step=num_iter)
                    wandb.log({f"valid/{k}": v for k, v in zip(["noobj", "obj", "coord", "class"], avg_loss_items)}, step=num_iter)
                    wandb.log({"valid/acc": avg_acc}, step=num_iter)
                    wandb.log({"valid/patience": self.patience}, step=num_iter)

                # Save checkpoints
                if avg_loss < best_valid_loss:
                    best_valid_loss = avg_loss
                    ckpt_path = self.ckpt_dir / f"{self.cfg.trainer.wandb.run_name}_best_epoch{self.cur_epoch}_{avg_loss:.4f}.pt"
                    if best_ckpt_path is not None:
                        best_ckpt_path.unlink(missing_ok=True)
                    best_ckpt_path = ckpt_path
                    self.save_checkpoint(ckpt_path)
                
                if last_ckpt_path is not None:
                    last_ckpt_path.unlink(missing_ok=True)
                last_ckpt_path = self.ckpt_dir / f"{self.cfg.trainer.wandb.run_name}_last.pt"
                self.save_checkpoint(last_ckpt_path)

                # Early stopping
                if avg_loss > best_valid_loss:
                    if self.patience == 0:
                        break
                    self.patience -= 1
                else:
                    self.patience = self.cfg.trainer.patience
        
        if self.log_to_wandb:
            wandb.finish()

@hydra.main(config_path="config", config_name="yolo_config", version_base=None)
def main(cfg: DictConfig):
    DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
    random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    log_to_wandb = cfg.trainer.wandb.log_to_wandb
    if cfg.debug:
        torch.autograd.set_detect_anomaly(True)
        c = {
            "TRAIN_VERSIONS": ["Bo"],
            "VALID_VERSIONS": ["Ka"],
            "TRAIN_ACTS": ["A"],
            "VALID_ACTS": ["D-1"],
        }
        constants = OmegaConf.create(c)
        log_to_wandb = False
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
        max_none_samples=cfg.dataset.max_none_samples,
        split = cfg.dataset.split,
        mixup_prob = cfg.dataset.mixup_prob,
        mixup_alpha = cfg.dataset.mixup_alpha,
        device = DEV
    )
    train_set, valid_set, = None, None
    if cfg.dataset.split == "version":
        train_set = Subset(base_set, base_set.get_subset_idxs(versions=constants.TRAIN_VERSIONS))
        valid_set = Subset(base_set, base_set.get_subset_idxs(versions=constants.VALID_VERSIONS))
    elif cfg.dataset.split == "act":
        train_set = Subset(base_set, base_set.get_subset_idxs(acts=constants.TRAIN_ACTS))
        valid_set = Subset(base_set, base_set.get_subset_idxs(acts=constants.VALID_ACTS))
    else:
        raise ValueError("Invalid split method")

    rng = torch.Generator().manual_seed(cfg.random_seed)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=rng,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        pin_memory_device=DEV
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=rng,
        collate_fn = collate_fn,
        num_workers=4,
        pin_memory=True,
        pin_memory_device=DEV
    )

    model = YOLO(
        num_anchors=len(C.ANCHORS), 
        C=len(C.MOTIFS),
        base_hidden=cfg.model.base_hidden,
        dropout=cfg.model.dropout    
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    trainer = Trainer(model, optimizer, base_set, train_loader, valid_loader, DEV, cfg, log_to_wandb)
    
    trainer.train()

if __name__ == "__main__":
    main()