from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import wandb
from dataset import LeitmotifDataset, Subset, collate_fn
from modules import RNNModel, CNNModel
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
        self.criterion = torch.nn.BCELoss()
        self.cur_epoch = 0
    
    def save_checkpoint(self):
        ckpt = {
            "epoch": self.cur_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        ckpt_dir = Path(f"checkpoints/{self.cfg['model']}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, ckpt_dir / f"{self.cfg['run_name']}_epoch{self.cur_epoch}.pt")

    def load_checkpoint(self):
        ckpt = torch.load(self.cfg["load_checkpoint"])
        self.cur_epoch = ckpt["epoch"]
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
    
    def train(self):
        if self.cfg["log_to_wandb"]:
            wandb.init(project="ring-leitmotif", name=self.cfg["run_name"], config=self.hyperparams)
        
        self.model.to(self.device)
        for epoch in tqdm(range(self.cur_epoch, self.hyperparams["num_epochs"])):
            self.cur_epoch = epoch
            self.model.train()
            for batch in tqdm(self.train_loader, leave=False):
                cqt, gt = batch
                cqt = cqt.to(self.device)
                gt = gt.to(self.device)
                pred = self.model(cqt)
                loss = self.criterion(pred, gt)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.cfg["log_to_wandb"]:
                    tp = ((pred > 0.5) & (gt == 1)).sum().item()
                    if tp == 0:
                        tp = 0.0001
                    fp = ((pred > 0.5) & (gt == 0)).sum().item()
                    fn = ((pred <= 0.5) & (gt == 1)).sum().item()
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    f1 = 2 * precision * recall / (precision + recall)
                    wandb.log({"train_loss": loss.item(), "train_precision": precision, "train_recall": recall, "train_f1": f1})
            
            self.model.eval()
            with torch.inference_mode():
                total_loss = 0
                total_precision = 0
                total_recall = 0
                total_f1 = 0
                for batch in tqdm(self.valid_loader, leave=False):
                    cqt, gt = batch
                    cqt = cqt.to(self.device)
                    gt = gt.to(self.device)
                    pred = self.model(cqt)
                    loss = self.criterion(pred, gt)
                    total_loss += loss.item()

                    if self.cfg["log_to_wandb"]:
                        tp = ((pred > 0.5) & (gt == 1)).sum().item()
                        if tp == 0:
                            tp = 0.0001
                        fp = ((pred > 0.5) & (gt == 0)).sum().item()
                        fn = ((pred <= 0.5) & (gt == 1)).sum().item()
                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                        f1 = 2 * precision * recall / (precision + recall)
                        total_precision += precision
                        total_recall += recall
                        total_f1 += f1
                    
                if self.cfg["log_to_wandb"]:
                    avg_loss = total_loss / len(self.valid_loader)
                    avg_precision = total_precision / len(self.valid_loader)
                    avg_recall = total_recall / len(self.valid_loader)
                    avg_f1 = total_f1 / len(self.valid_loader)
                    wandb.log({"valid_loss": avg_loss, "valid_precision": avg_precision, "valid_recall": avg_recall, "valid_f1": avg_f1})
            
            self.save_checkpoint()
        
        if self.cfg["log_to_wandb"]:
            wandb.finish()

@hydra.main(config_path=".", config_name="train_config", version_base=None)
def main(config: DictConfig):
    cfg = OmegaConf.to_container(config.cfg)
    hyperparams = OmegaConf.to_container(config.hyperparams)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    base_set = LeitmotifDataset(Path("data/CQT"), Path("data/LeitmotifOccurrencesInstances/Instances"))
    train_set, valid_set, test_set = None, None, None
    if cfg["split"] == "version":
        train_set = Subset(base_set, base_set.get_subset_idxs(versions=C.TRAIN_VERSION))
        valid_set = Subset(base_set, base_set.get_subset_idxs(versions=C.VALID_VERSION))
        test_set = Subset(base_set, base_set.get_subset_idxs(versions=C.TEST_VERSION))
    elif cfg["split"] == "act":
        train_set = Subset(base_set, base_set.get_subset_idxs(acts=C.TRAIN_ACT))
        valid_set = Subset(base_set, base_set.get_subset_idxs(acts=C.VALID_ACT))
        test_set = Subset(base_set, base_set.get_subset_idxs(acts=C.TEST_ACT))
    else:
        raise ValueError("Invalid split method")

    rng = torch.Generator().manual_seed(cfg["random_seed"])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, generator=rng, collate_fn = collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=False, collate_fn = collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, collate_fn = collate_fn)

    model = None
    if cfg["model"] == "RNN":
        model = RNNModel(hidden_size=hyperparams["hidden_size"], num_layers=hyperparams["num_layers"])
    elif cfg["model"] == "CNN":
        model = CNNModel()
    else:
        raise ValueError("Invalid model name")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    trainer = Trainer(model, optimizer, train_loader, valid_loader, DEV, cfg, hyperparams)
    
    trainer.train()

if __name__ == "__main__":
    main()




