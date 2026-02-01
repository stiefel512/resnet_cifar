import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms as T

from model import resnet_config, CIFARResNet

from omegaconf import OmegaConf
from typing import Tuple
from parse import parse

import random
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def set_seed(seed: int) -> None:
    # Set python's seed
    random.seed(seed)
    
    # Numpy's seed
    np.random.seed(seed)
    
    # PyTorch's seed
    torch.manual_seed(seed)
    
    # PyTorch cuda's seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # cuDNN/Backend determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
def build_transform(cfg: OmegaConf) -> T.transforms:
    transforms = []
    if cfg.data.augmentations.random_crop:
        transforms.append(T.RandomCrop([cfg.data.input_size[1], cfg.data.input_size[2]], padding=cfg.data.augmentations.random_crop_padding))
    if cfg.data.augmentations.horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if cfg.data.augmentations.color_jitter:
        transforms.append(T.ColorJitter())
    
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))  # Standard ImageNet Normalization
    transforms = T.Compose(transforms)
    return transforms


def build_dataloaders(cfg: OmegaConf) -> Tuple[DataLoader, DataLoader]:
    
    # Set the trainsforms:
    transforms = build_transform(cfg)
    val_transforms = T.Compose([T.ToTensor(),
                                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # Load the dataset with and without transforms. 
    # The dataset with transforms will be used for the training, and the dataset without will be used for the validation
    if cfg.data.dataset == 'CIFAR10':
        val_ds = CIFAR10(root=cfg.data.data_dir,
                          train=True,
                          transform=val_transforms,
                          download=True)
        train_ds = CIFAR10(root=cfg.data.data_dir,
                          train=True,
                          transform=transforms,
                          download=False)
    elif cfg.data.dataset == 'CIFAR100':
        val_ds = CIFAR100(root=cfg.data.data_dir,
                          train=True,
                          transform=val_transforms,
                          download=True)
        train_ds = CIFAR100(root=cfg.data.data_dir,
                            train=True,
                            transform=transforms,
                            download=False)
    else:
        raise Exception("Unsupported Dataset")
        
    # Create the train/validation split
    ds_size = len(train_ds)
    val_size = int(ds_size * cfg.data.train_val_split)
    train_size = ds_size - val_size
    g = torch.Generator()
    g.manual_seed(cfg.experiment.seed)
    train_inds, val_inds = random_split(train_ds, [train_size, val_size], generator=g)
    train_dataset = Subset(train_ds, train_inds.indices)
    val_dataset = Subset(val_ds, val_inds.indices)    
    
    # Ensure the same shuffle order and random augmentations per epoch
    g = torch.Generator()
    g.manual_seed(cfg.experiment.seed)
    
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = cfg.data.batch_size,
        shuffle = True,
        num_workers = cfg.data.num_workers,
        worker_init_fn = seed_worker,
        generator = g
    )
    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = cfg.data.batch_size,
        shuffle=False,
        num_workers = cfg.data.num_workers,
        worker_init_fn = seed_worker,
        generator = g
    )
    return train_dataloader, val_dataloader
    
    
def build_model(cfg: OmegaConf) -> nn.Module:
    arch = parse("resnet{configuration}", cfg.model.architecture)
    if arch is None:
        raise Exception("Incorrect Model Architecture")
    ds = parse("CIFAR{num_classes}", cfg.data.dataset)
    if ds is None:
        raise Exception("Unsupported Dataset")
    
    return CIFARResNet(
        configuration = int(arch['configuration']),
        in_channels = cfg.data.input_size[0],
        num_classes = int(ds['num_classes']),
        norm = cfg.model.norm,
        num_groups = cfg.model.num_groups,
        base_channels = cfg.model.base_channels
    )
    
    
def build_optimizer(optimizer_conf: OmegaConf, model: nn.Module) -> torch.optim.Optimizer:
    if optimizer_conf.type == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = optimizer_conf.lr,
            momentum = optimizer_conf.momentum,
            weight_decay = optimizer_conf.weight_decay
        )
    elif optimizer_conf.type == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = optimizer_conf.lr,
            weight_decay = optimizer_conf.weight_decay
        )
        
    else:
        raise Exception("Optimizer type not supported")
    
    
def build_scheduler(cfg: OmegaConf, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    if cfg.scheduler.warmup is not None and cfg.scheduler.warmup.epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=cfg.scheduler.warmup.start_factor,
            end_factor=1.0,
            total_iters=cfg.scheduler.warmup.epochs
        )
        remaining_epochs = cfg.training.max_epochs - cfg.scheduler.warmup.epochs
    else:
        warmup_scheduler = None
        remaining_epochs = cfg.training.max_epochs
    
    if cfg.scheduler.type == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=remaining_epochs,
            eta_min=cfg.scheduler.eta_min,
            last_epoch=remaining_epochs
        )
    elif cfg.scheduler.type == 'step':
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, 
            step_size=cfg.scheduler.step_size,
            gamma=0.1,
            last_epoch=remaining_epochs
        )
    else:
        raise Exception("Unsupported LR Scheduler")
    
    if warmup_scheduler is None:
        return main_scheduler
    else:
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, main_scheduler], 
            milestones=[cfg.scheduler.warmup.epochs]
        )
        

def get_loss_fn(cfg: OmegaConf) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=cfg.regularization.label_smoothing)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> None:
    model.train()
    device = get_device()
    
    for batch_idx, batch in enumerate(loader):
        # Unpack the batch
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        # Compute the predictions
        pred = model(x)
        
        # Compute the loss
        loss = loss_fn(pred, y)
        
        # Backpropagation:
        # Clear the gradients
        optimizer.zero_grad()
        # Backprop
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(loader.dataset):>5d}]")
    
    
def validate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> None:
    model.eval()
    
    val_loss = 0.0
    correct = 0
    total = 0
    
    device = get_device()
    
    with torch.no_grad():  # Disable Gradient Calculation
        for batch_idx, batch in enumerate(loader):
            # Unpack the batch
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # Compute the predictions
            preds = model(x)
            
            # Compute the loss
            loss = loss_fn(preds, y)
            
            val_loss += loss.item() * x.size(0)
            
            _, predicted = torch.max(preds, 1)
            
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    avg_loss = val_loss / len(loader.dataset)
    accuracy = 100 * correct / total
            
    return avg_loss, accuracy


def train(cfg: OmegaConf) -> None:
    # Save the exact configuration used in the experiment
    out_dir = cfg.experiment.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, f"{out_dir}/config_used.yaml")
    
    # Set all seeds
    set_seed(cfg.experiment.seed)
    
    # Get the dataloader objects
    train_loader, val_loader = build_dataloaders(cfg)
    
    # Build the model
    model = build_model(cfg)
    model.to(get_device())
    
    # Get the optimizer
    optimizer = build_optimizer(cfg.optimizer, model)
    
    # Get the LR scheduler
    scheduler = build_scheduler(cfg, optimizer)
    
    # Get the Loss Function
    loss_fn = get_loss_fn(cfg)
    
    # Train
    for epoch in range(cfg.training.max_epochs):
        # Train on the training data
        train_one_epoch(model, train_loader, optimizer, loss_fn)
        # Validate on the validation data
        val_loss, val_acc = validate(model, val_loader, loss_fn)
        print(f"Epoch {epoch}:\tLoss {val_loss}, Acc {val_acc}")
    
        # Advance the LR Scheduler
        scheduler.step()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    
    train(cfg)