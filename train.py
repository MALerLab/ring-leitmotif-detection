from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import wandb
from dataset import LeitmotifDataset, Subset, collate_fn
from modules import RNNModel, CNNModel
from data_utils import get_binary_f1, get_multiclass_acc
import constants as C

class Trainer:
    def __init__(self, model, optimizer, train_loader, valid_loader, device, cfg, hyperparams):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.cfg = cfg
        self.hyperparams = hyperparams
        self.bce = torch.nn.BCELoss()
        self.ce = torch.nn.CrossEntropyLoss()
        self.cur_epoch = 0
    
    def save_checkpoint(self):
        ckpt = {
            "epoch": self.cur_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        ckpt_dir = Path(f"checkpoints/{self.cfg.model}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, ckpt_dir / f"{self.cfg.run_name}_epoch{self.cur_epoch}.pt")

    def load_checkpoint(self):
        ckpt = torch.load(self.cfg.load_checkpoint)
        self.cur_epoch = ckpt["epoch"]
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
    
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
        if self.cfg.log_to_wandb:
            wandb.init(project="ring-leitmotif", name=self.cfg.run_name, config=OmegaConf.to_container(self.cfg))
        
        self.model.to(self.device)
        num_iter = 0
        adv_iter = 0
        for epoch in tqdm(range(self.cur_epoch, self.hyperparams.num_epochs)):
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

                if epoch > 10:
                    # Adversarial train loop
                    version_pred = version_pred.permute(0, 2, 1)
                    version_loss = self.ce(version_pred, version_gt)
                    adv_loss = version_loss

                    singing_loss = None
                    if self.cfg.train_singing:
                        singing_loss = self.bce(singing_pred, singing_gt)
                        adv_loss += singing_loss
                    adv_loss_multiplier = min(1, adv_iter / self.hyperparams.adv_grad_iter)
                    loss += adv_loss_multiplier * adv_loss
                    if self.cfg.log_to_wandb:
                        wandb.log({"adv_loss_multiplier": adv_loss_multiplier}, step=num_iter)
                    adv_iter += 1

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                if self.cfg.log_to_wandb:
                    f1, precision, recall = get_binary_f1(leitmotif_pred, leitmotif_gt, 0.5)
                    wandb.log({"train_loss": leitmotif_loss.item(), "train_precision": precision, "train_recall": recall, "train_f1": f1}, step=num_iter)
                    wandb.log({"total_loss": loss.item()}, step=num_iter)
                    if epoch > 10:
                        wandb.log({"adv_version_loss": version_loss.item()}, step=num_iter)
                        wandb.log({"adv_version_acc": get_multiclass_acc(version_pred, version_gt)}, step=num_iter)
                        if self.cfg.train_singing:
                            f1, precision, recall = get_binary_f1(singing_pred, singing_gt, 0.5)
                            wandb.log({"adv_singing_loss": singing_loss.item(), "adv_singing_train_f1": f1}, step=num_iter)
                num_iter += 1

            if epoch == 10:
                print("Training MLP submodules...")
                self._train_mlp_submodules(num_epochs=self.hyperparams.adv_num_epochs,
                                           train_singing=self.cfg.train_singing)

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

                    if self.cfg.log_to_wandb:
                        f1, precision, recall = get_binary_f1(leitmotif_pred, leitmotif_gt, 0.5)
                        total_precision += precision
                        total_recall += recall
                        total_f1 += f1
                    
                if self.cfg.log_to_wandb:
                    avg_loss = total_loss / len(self.valid_loader)
                    avg_precision = total_precision / len(self.valid_loader)
                    avg_recall = total_recall / len(self.valid_loader)
                    avg_f1 = total_f1 / len(self.valid_loader)
                    wandb.log({"valid_loss": avg_loss, "valid_precision": avg_precision, "valid_recall": avg_recall, "valid_f1": avg_f1})
            
            self.save_checkpoint()
        
        if self.cfg.log_to_wandb:
            wandb.finish()

@hydra.main(config_path="config", config_name="train_config", version_base=None)
def main(config: DictConfig):
    cfg = config.cfg
    hyperparams = config.hyperparams
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    base_set = LeitmotifDataset(Path("data/CQT"), 
                                Path("data/LeitmotifOccurrencesInstances/Instances"),
                                Path("data/WagnerRing_Public/02_Annotations/ann_audio_singing"))
    train_set, valid_set, test_set = None, None, None
    if cfg.split == "version":
        train_set = Subset(base_set, base_set.get_subset_idxs(versions=C.TRAIN_VERSIONS))
        valid_set = Subset(base_set, base_set.get_subset_idxs(versions=C.VALID_VERSIONS))
    elif cfg.split == "act":
        train_set = Subset(base_set, base_set.get_subset_idxs(acts=C.TRAIN_ACTS))
        valid_set = Subset(base_set, base_set.get_subset_idxs(acts=C.VALID_ACTS))
    else:
        raise ValueError("Invalid split method")

    rng = torch.Generator().manual_seed(cfg.random_seed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, generator=rng, collate_fn = collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=False, collate_fn = collate_fn)

    model = None
    if cfg.model == "RNN":
        model = RNNModel(hidden_size=hyperparams.hidden_size,
                         num_layers=hyperparams.num_layers,
                         adv_grad_multiplier=hyperparams.adv_grad_multiplier)
    elif cfg.model == "CNN":
        model = CNNModel(adv_grad_multiplier=hyperparams.adv_grad_multiplier)
    else:
        raise ValueError("Invalid model name")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr)
    trainer = Trainer(model, optimizer, train_loader, valid_loader, DEV, cfg, hyperparams)
    
    trainer.train()

if __name__ == "__main__":
    main()




