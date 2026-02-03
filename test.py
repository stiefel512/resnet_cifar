import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from utils import get_device
from train import build_model

from omegaconf import OmegaConf


def load_model(cfg: OmegaConf) -> nn.Module:
    model = build_model(cfg)
    state_dict = torch.load(f"{cfg.experiment.output_dir}/best.pt", map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    return model

def get_test_dataloader(cfg: OmegaConf) -> DataLoader:
    transforms = T.Compose([T.ToTensor(),
                            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # Load the dataset with and without transforms. 
    # The dataset with transforms will be used for the training, and the dataset without will be used for the validation
    if cfg.data.dataset == 'CIFAR10':
        ds = CIFAR10(root=cfg.data.data_dir,
                     train=False,
                     transform=transforms,
                     download=True)
    elif cfg.data.dataset == 'CIFAR100':
        ds = CIFAR100(root=cfg.data.data_dir,
                      train=False,
                      transform=transforms,
                      download=True)
    else:
        raise Exception("Unsupported Dataset")
    
    dataloader = DataLoader(
        dataset = ds,
        batch_size = cfg.data.batch_size,
        shuffle = False,
        num_workers = cfg.data.num_workers
    )
    return dataloader


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
):
    model.to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    
    device = get_device()
    
    model = load_model(cfg)
    
    criterion = nn.CrossEntropyLoss()
    
    dataloader = get_test_dataloader(cfg)
    
    avg_loss, acc = evaluate(model, dataloader, criterion, device)
    
    print(f"TEST LOSS: {avg_loss}, TEST ACCURACY {acc}")