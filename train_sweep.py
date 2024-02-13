from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import wandb
import yaml
from dataset import AudioDataset, Subset, collate_fn
from modules import RNNModel, CNNModel
from data_utils import get_binary_f1, get_multiclass_acc
import constants as C

class Trainer:
    def __init__(self, model, optimizer, train_loader, valid_loader, device, cfg):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.cfg = cfg
        self.bce = torch.nn.BCELoss()
        self.ce = torch.nn.CrossEntropyLoss()
        self.cur_epoch = 0
    
    def save_checkpoint(self):
        ckpt = {
            "epoch": self.cur_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        ckpt_dir = Path(f"checkpoints/{wandb.config.model}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, ckpt_dir / f"{self.cfg.name}_epoch{self.cur_epoch}.pt")

    # def load_checkpoint(self):
    #     # ckpt = torch.load(self.cfg.load_checkpoint)
    #     self.cur_epoch = ckpt["epoch"]
    #     self.model.load_state_dict(ckpt["model"])
    #     self.optimizer.load_state_dict(ckpt["optimizer"])
    #     for state in self.optimizer.state.values():
    #         for k, v in state.items():
    #             if torch.is_tensor(v):
    #                 state[k] = v.to(self.device)
    
    def _train_mlp_submodules(self, num_epochs=1, train_singing=False):
        self.model.train()
        self.model.freeze_backbone()
        for epoch in tqdm(range(num_epochs), leave=False):
            for batch in tqdm(self.train_loader, leave=False):        
                cqt, _, singing_gt, version_gt = batch
                cqt = cqt.to(self.device)
                singing_gt = singing_gt.to(self.device)
                version_gt = version_gt.to(self.device)

                _, singing_pred, version_pred = self.model(cqt)
                version_pred = version_pred.permute(0, 2, 1)
                loss = self.ce(version_pred, version_gt)
                if train_singing:
                    loss += self.bce(singing_pred, singing_gt)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
        self.model.unfreeze_backbone()

    def train(self):
        self.model.to(self.device)
        num_iter = 0
        adv_iter = 0
        for epoch in tqdm(range(self.cur_epoch, wandb.config.num_epochs)):
            self.cur_epoch = epoch
            self.model.train()
            for batch in tqdm(self.train_loader, leave=False):
                # Leitmotif train loop
                cqt, leitmotif_gt, singing_gt, version_gt = batch
                cqt = cqt.to(self.device)
                leitmotif_gt = leitmotif_gt.to(self.device)
                singing_gt = singing_gt.to(self.device)
                version_gt = version_gt.to(self.device)
                leitmotif_pred, singing_pred, version_pred = self.model(cqt)
                leitmotif_loss = self.bce(leitmotif_pred, leitmotif_gt)
                loss = leitmotif_loss

                # Adversarial train loop
                version_pred = version_pred.permute(0, 2, 1)
                version_loss = self.ce(version_pred, version_gt)
                adv_loss = version_loss

                singing_loss = None
                if wandb.config.train_singing:
                    singing_loss = self.bce(singing_pred, singing_gt)
                    adv_loss += singing_loss
                adv_loss_multiplier = min(1, adv_iter / wandb.config.adv_grad_iter)
                loss += adv_loss_multiplier * adv_loss
                wandb.log({"adv_loss_multiplier": adv_loss_multiplier}, step=num_iter)
                adv_iter += 1

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                f1, precision, recall = get_binary_f1(leitmotif_pred, leitmotif_gt, 0.5)
                wandb.log({"train_loss": leitmotif_loss.item(), "train_precision": precision, "train_recall": recall, "train_f1": f1}, step=num_iter)
                wandb.log({"total_loss": loss.item()}, step=num_iter)
                wandb.log({"adv_version_loss": version_loss.item()}, step=num_iter)
                wandb.log({"adv_version_acc": get_multiclass_acc(version_pred, version_gt)}, step=num_iter)
                if wandb.config.train_singing:
                    f1, precision, recall = get_binary_f1(singing_pred, singing_gt, 0.5)
                    wandb.log({"adv_singing_loss": singing_loss.item(), "adv_singing_train_f1": f1}, step=num_iter)
                num_iter += 1

            self.model.eval()
            with torch.inference_mode():
                total_loss = 0
                total_precision = 0
                total_recall = 0
                total_f1 = 0
                for batch in tqdm(self.valid_loader, leave=False):
                    cqt, leitmotif_gt, singing_gt, version_gt = batch
                    cqt = cqt.to(self.device)
                    leitmotif_gt = leitmotif_gt.to(self.device)
                    singing_gt = singing_gt.to(self.device)
                    leitmotif_pred, singing_pred, version_pred = self.model(cqt)
                    leitmotif_loss = self.bce(leitmotif_pred, leitmotif_gt)
                    total_loss += leitmotif_loss.item()

                    f1, precision, recall = get_binary_f1(leitmotif_pred, leitmotif_gt, 0.5)
                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1
                    
                avg_loss = total_loss / len(self.valid_loader)
                avg_precision = total_precision / len(self.valid_loader)
                avg_recall = total_recall / len(self.valid_loader)
                avg_f1 = total_f1 / len(self.valid_loader)
                wandb.log({"valid_loss": avg_loss, "valid_precision": avg_precision, "valid_recall": avg_recall, "valid_f1": avg_f1})
            
            self.save_checkpoint()

        wandb.finish()

def main():
    with open("config/sweep_config.yaml") as f:
        sweep_config = yaml.safe_load(f)
    run = wandb.init(project="ring-leitmotif", config=sweep_config)

    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    base_set = AudioDataset(Path("data/wav-22050"), 
                            Path("data/LeitmotifOccurrencesInstances/Instances"),
                            Path("data/WagnerRing_Public/02_Annotations/ann_audio_singing"),
                            mixup_prob=wandb.config.mixup_prob,
                            mixup_alpha=wandb.config.mixup_alpha,
                            device=DEV)
    train_set, valid_set, test_set = None, None, None
    if wandb.config.split == "version":
        train_set = Subset(base_set, base_set.get_subset_idxs(versions=C.TRAIN_VERSIONS))
        valid_set = Subset(base_set, base_set.get_subset_idxs(versions=C.VALID_VERSIONS))
    elif wandb.config.split == "act":
        train_set = Subset(base_set, base_set.get_subset_idxs(acts=C.TRAIN_ACTS))
        valid_set = Subset(base_set, base_set.get_subset_idxs(acts=C.VALID_ACTS))
    else:
        raise ValueError("Invalid split method")

    rng = torch.Generator().manual_seed(1204)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, generator=rng, collate_fn = collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=False, collate_fn = collate_fn)

    model = None
    if wandb.config.model == "RNN":
        model = RNNModel(mlp_hidden_size=128 * wandb.config.mlp_hidden_size_multiplier,
                         adv_grad_multiplier=wandb.config.adv_grad_multiplier)
    elif wandb.config.model == "CNN":
        model = CNNModel(mlp_hidden_size=64 * wandb.config.mlp_hidden_size_multiplier,
                         adv_grad_multiplier=wandb.config.adv_grad_multiplier)
    
    mlp_params = [param for name, param in model.named_parameters() if 'mlp' in name]
    backbone_params = [param for name, param in model.named_parameters() if 'mlp' not in name]
    optimizer = torch.optim.Adam([
        {'params': mlp_params, 'lr': 0.02},
        {'params': backbone_params, 'lr': 0.002}
    ])
    trainer = Trainer(model, optimizer, train_loader, valid_loader, DEV, run)
    
    trainer.train()

if __name__ == "__main__":
    main()




