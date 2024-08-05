from pathlib import Path
import random
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import wandb
from data.dataset import OTFDataset, Subset, collate_fn
from modules import CNNModel, CRNNModel, FiLMModel, FiLMAttnModel, CNNAttnModel, BBoxModel
from data.data_utils import get_binary_f1, get_boundaries, get_diou_loss
import constants as C

class Trainer:
    def __init__(self, model, optimizer, dataset, train_loader, valid_loader, device, cfg, hyperparams):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.cfg = cfg
        self.hyperparams = hyperparams
        self.bce = torch.nn.BCELoss()
        self.cur_epoch = 0
        self.ckpt_dir = Path(f"checkpoints/{self.cfg.model}")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
    
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

    def train(self):
        if self.cfg.log_to_wandb:
            wandb.init(project=self.cfg.wandb_project,
                       name=self.cfg.run_name, 
                       config={**OmegaConf.to_container(self.cfg),
                               **OmegaConf.to_container(self.hyperparams)})
        
        self.model.to(self.device)
        num_iter = 0
        best_valid_loss = float("inf")
        best_ckpt_path = None
        last_ckpt_path = None
        for epoch in tqdm(range(self.cur_epoch, self.hyperparams.num_epochs), ascii=True):
            self.cur_epoch = epoch
            self.model.train()
            self.dataset.enable_mixup()
            for batch in tqdm(self.train_loader, leave=False, ascii=True):
                # Leitmotif train loop
                wav, gt = batch
                wav = wav.to(self.device)
                gt = gt.to(self.device)

                if self.cfg.model in ["FiLM", "FiLMAttn"]:
                    labels = self.randomize_none_samples(gt)
                    pred = self.model(wav, labels).squeeze(-1)
                    pred = pred[torch.arange(pred.shape[0]), :, labels]
                    gt = gt[torch.arange(gt.shape[0]), :, labels]
                    loss = self.bce(pred, gt)
                elif self.cfg.model == "BBox":
                    pred = self.model(wav)
                    pred = pred.reshape(-1, 2)
                    gt = get_boundaries(gt, device=self.device)
                    loss, iou = diou_loss(pred, gt)
                else:
                    pred = self.model(wav)
                    loss = self.bce(pred, gt)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                if self.cfg.log_to_wandb:
                    if self.cfg.model == "BBox":
                        wandb.log({"train/loss": loss.item(), "train/iou": iou}, step=num_iter)
                    else:
                        f1, precision, recall = get_binary_f1(pred, gt, 0.5)
                        wandb.log({"train/loss": loss.item(), "train/precision": precision, "train/recall": recall, "train/f1": f1}, step=num_iter)
                        wandb.log({"train/total_loss": loss.item()}, step=num_iter)
                num_iter += 1

            self.model.eval()
            self.dataset.disable_mixup()
            with torch.inference_mode():
                total_loss = 0
                total_f1, total_p, total_r, total_iou = 0, 0, 0, 0
                for batch in tqdm(self.valid_loader, leave=False, ascii=True):
                    wav, gt = batch
                    wav = wav.to(self.device)
                    gt = gt.to(self.device)

                    if self.cfg.model in ["FiLM", "FiLMAttn"]:
                        labels = self.randomize_none_samples(gt)
                        pred = self.model(wav, labels).squeeze(-1)
                        gt = gt[torch.arange(gt.shape[0]), :, labels]
                        loss = self.bce(pred, gt)
                    elif self.cfg.model == "BBox":
                        pred = self.model(wav)
                        pred = pred.reshape(-1, 2)
                        gt = get_boundaries(gt, device=self.device)
                        loss, iou = diou_loss(pred, gt)
                    else:
                        pred = self.model(wav)
                        loss = self.bce(pred, gt)

                    total_loss += loss.item()
                    if self.cfg.log_to_wandb:
                        if self.cfg.model == "BBox":
                            total_iou += iou
                        else:
                            f1, p, r = get_binary_f1(pred, gt, 0.5)
                            total_f1 += f1
                            total_p += p
                            total_r += r
                    
                avg_loss = total_loss / len(self.valid_loader)
                if self.cfg.log_to_wandb:
                    if self.cfg.model == "BBox":
                        avg_iou = total_iou / len(self.valid_loader)
                        wandb.log({"valid/loss": avg_loss, "valid/iou": avg_iou}, step=num_iter)
                    else:
                        p = total_p / len(self.valid_loader)
                        r = total_r / len(self.valid_loader)
                        f1 = total_f1 / len(self.valid_loader)
                        wandb.log({"valid/loss": avg_loss, "valid/precision": p, "valid/recall": r, "valid/f1": f1}, step=num_iter)

                if avg_loss < best_valid_loss:
                    best_valid_loss = avg_loss
                    ckpt_path = self.ckpt_dir / f"{self.cfg.run_name}_best_epoch{self.cur_epoch}_{avg_loss:.4f}.pt"
                    if best_ckpt_path is not None:
                        best_ckpt_path.unlink(missing_ok=True)
                    best_ckpt_path = ckpt_path
                    self.save_checkpoint(ckpt_path)
                
                if last_ckpt_path is not None:
                    last_ckpt_path.unlink(missing_ok=True)
                last_ckpt_path = self.ckpt_dir / f"{self.cfg.run_name}_last.pt"
                self.save_checkpoint(last_ckpt_path)
        
        if self.cfg.log_to_wandb:
            wandb.finish()

@hydra.main(config_path="config", config_name="train_config", version_base=None)
def main(config: DictConfig):
    cfg = config.cfg
    hyperparams = config.hyperparams
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(cfg.random_seed)
    base_set = OTFDataset(Path("data/wav-22050"), 
                          Path("data/LeitmotifOccurrencesInstances/Instances"),
                          C.TRAIN_VERSIONS,
                          C.VALID_VERSIONS,
                          C.TRAIN_ACTS,
                          C.VALID_ACTS,
                          C.MOTIFS,
                          include_none_class = hyperparams.include_none_class,
                          split = cfg.split,
                          mixup_prob = hyperparams.mixup_prob,
                          mixup_alpha = hyperparams.mixup_alpha,
                          device = DEV)
    train_set, valid_set, = None, None
    if cfg.split == "version":
        train_set = Subset(base_set, base_set.get_subset_idxs(versions=C.TRAIN_VERSIONS))
        valid_set = Subset(base_set, base_set.get_subset_idxs(versions=C.VALID_VERSIONS))
    elif cfg.split == "act":
        train_set = Subset(base_set, base_set.get_subset_idxs(acts=C.TRAIN_ACTS))
        valid_set = Subset(base_set, base_set.get_subset_idxs(acts=C.VALID_ACTS))
    else:
        raise ValueError("Invalid split method")

    rng = torch.Generator().manual_seed(cfg.random_seed)
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=32, 
                                               shuffle=True, 
                                               generator=rng, 
                                               collate_fn=collate_fn, 
                                               num_workers=4,
                                               pin_memory=True,
                                               pin_memory_device=DEV)
    valid_loader = torch.utils.data.DataLoader(valid_set, 
                                               batch_size=32, 
                                               shuffle=False, 
                                               collate_fn = collate_fn,
                                               num_workers=4,
                                               pin_memory=True,
                                               pin_memory_device=DEV)

    model = None
    if cfg.model == "CNN":
        model = CNNModel(num_classes=base_set.num_classes)
    elif cfg.model == "CRNN":
        model = CRNNModel(num_classes=base_set.num_classes)
    elif cfg.model == "FiLM":
        model = FiLMModel(num_classes=base_set.num_classes,
                          filmgen_emb=hyperparams.filmgen_emb,
                          filmgen_hidden=hyperparams.filmgen_hidden)
    elif cfg.model == "FiLMAttn":
        model = FiLMAttnModel(num_classes=base_set.num_classes,
                              filmgen_emb=hyperparams.filmgen_emb,
                              filmgen_hidden=hyperparams.filmgen_hidden,
                              attn_dim=hyperparams.attn_dim,
                              attn_depth=hyperparams.attn_depth,
                              attn_heads=hyperparams.attn_heads)
    elif cfg.model == "CNNAttn":
        model = CNNAttnModel(num_classes=base_set.num_classes,
                             attn_dim=hyperparams.attn_dim,
                             attn_depth=hyperparams.attn_depth,
                             attn_heads=hyperparams.attn_heads)
    elif cfg.model == "BBox":
        model = BBoxModel(num_classes=base_set.num_classes,
                          apply_attn=hyperparams.apply_attn,
                          attn_dim=hyperparams.attn_dim,
                          attn_depth=hyperparams.attn_depth,
                          attn_heads=hyperparams.attn_heads)
    else:
        raise ValueError("Invalid model type")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr)
    trainer = Trainer(model, optimizer, base_set, train_loader, valid_loader, DEV, cfg, hyperparams)
    
    trainer.train()

if __name__ == "__main__":
    main()