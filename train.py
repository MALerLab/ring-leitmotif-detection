from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import wandb
from dataset import LeitmotifDataset, Subset, collate_fn
from modules import RNNModel, CNNModel
from constants import TRAIN_VERSION, VALID_VERSION, TEST_VERSION

# TODO: add metrics

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
            for batch in tqdm(self.train_loader):
                cqt, gt = batch
                cqt = cqt.to(self.device)
                gt = gt.to(self.device)
                pred = self.model(cqt)
                loss = self.criterion(pred, gt)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.cfg["log_to_wandb"]:
                    wandb.log({"train_loss": loss.item()})
            
            self.model.eval()
            with torch.inference_mode():
                total_loss = 0
                for batch in tqdm(self.valid_loader):
                    cqt, gt = batch
                    cqt = cqt.to(self.device)
                    gt = gt.to(self.device)
                    pred = self.model(cqt)
                    loss = self.criterion(pred, gt)
                    total_loss += loss.item()
                    
                if self.cfg["log_to_wandb"]:
                    avg_loss = total_loss / len(self.valid_loader)
                    wandb.log({"valid_loss": avg_loss})
            
            self.save_checkpoint()
        
        if self.cfg["log_to_wandb"]:
            wandb.finish()

@hydra.main(config_path=".", config_name="train_config", version_base=None)
def main(config: DictConfig):
    cfg = OmegaConf.to_container(config.cfg)
    hyperparams = OmegaConf.to_container(config.hyperparams)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    base_set = LeitmotifDataset(Path("data/CQT"), Path("data/LeitmotifOccurrencesInstances/Instances"))
    train_set = Subset(base_set, base_set.get_subset_idxs(versions=TRAIN_VERSION))
    valid_set = Subset(base_set, base_set.get_subset_idxs(versions=VALID_VERSION))
    test_set = Subset(base_set, base_set.get_subset_idxs(versions=TEST_VERSION))

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




