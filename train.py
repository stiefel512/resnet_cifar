import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10, CIFAR100, VisionDataset
from torchvision import transforms as T

from model import resnet_config, CIFARResNet
from utils import get_device

from omegaconf import OmegaConf
from typing import Tuple
from parse import parse
import pandas as pd
import random
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


device = get_device()

def set_seed(seed: int) -> None:
    """Set all the different seeds, for experiment reproducibility

    Args:
        seed (int): The seed
    """
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
    """Seed DataLoader workers, to ensure consistent training order and augmentations across runs

    Args:
        worker_id (_type_): The ID of the worker to be seeded
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
def build_transform(cfg: OmegaConf) -> T.transforms:
    """Build the training transform

    Args:
        cfg (OmegaConf): The configuration

    Returns:
        T.transforms: The composed transform
    """
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


def stratified_split(dataset: VisionDataset, val_fraction: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a stratified training/validation split

    Args:
        dataset (VisionDataset): The dataset to split
        val_fraction (float): The fraction of items for the validation set
        seed (int): The random number seed

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The indices of dataset for training and validation
    """
    targets = torch.tensor(dataset.targets)

    g = torch.Generator().manual_seed(seed)

    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[int(label)].append(idx)

    train_idx = []
    val_idx = []

    for cls, idxs in class_indices.items():
        idxs = torch.tensor(idxs)
        perm = idxs[torch.randperm(len(idxs), generator=g)]
        n_val = int(len(idxs) * val_fraction)
        val_idx.append(perm[:n_val])
        train_idx.append(perm[n_val:])

    return torch.cat(train_idx), torch.cat(val_idx)


def build_dataloaders(cfg: OmegaConf) -> Tuple[DataLoader, DataLoader]:
    """Build the DataLoader objects for training and validation

    Args:
        cfg (OmegaConf): The Configuration

    Raises:
        Exception: Unsupported Dataset

    Returns:
        Tuple[DataLoader, DataLoader]: Training and Validation DataLoader objects
    """
    
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
    if cfg.data.stratify:
        # Create a completely random training/validation split
        ds_size = len(train_ds)
        val_size = int(ds_size * cfg.data.train_val_split)
        train_size = ds_size - val_size
        g = torch.Generator()
        g.manual_seed(cfg.experiment.seed)
        # Instead of using random split, use a random permutation and take in the new order
        indices = torch.randperm(ds_size, generator=g)
        train_inds = indices[:train_size]
        val_inds = indices[train_size:]
        train_dataset = Subset(train_ds, train_inds)
        val_dataset = Subset(val_ds, val_inds)    
    else:
        # Instead of using a purely random split, stratify.
        train_idx, val_idx = stratified_split(
            train_ds, 
            cfg.data.train_val_split, 
            cfg.experiment.seed
        )
        train_dataset = Subset(train_ds, train_idx)
        val_dataset = Subset(val_ds, val_idx)
    
    # Ensure the same shuffle order and random augmentations per epoch
    train_g = torch.Generator()
    train_g.manual_seed(cfg.experiment.seed)
    val_g = torch.Generator()
    val_g.manual_seed(cfg.experiment.seed + 1)
    
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = cfg.data.batch_size,
        shuffle = True,
        num_workers = cfg.data.num_workers,
        worker_init_fn = seed_worker,
        generator = train_g
    )
    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = cfg.data.batch_size,
        shuffle=False,
        num_workers = cfg.data.num_workers,
        worker_init_fn = seed_worker,
        generator = val_g
    )
    return train_dataloader, val_dataloader
    
    
def build_model(cfg: OmegaConf) -> nn.Module:
    """Build the Model

    Args:
        cfg (OmegaConf): The configuration

    Raises:
        Exception: Incorrect Model Architecture
        Exception: Unsupported Dataset

    Returns:
        nn.Module: ResNet
    """
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
    
    
def kaiming_init(model: nn.Module) -> None:
    """Explicitly initialize the model weights with kaiming initialization

    Args:
        model (nn.Module): The model to initialize
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight,
                mode='fan_out',
                nonlinearity='relu'
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight,
                mode='fan_in',
                nonlinearity='relu'
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            
    
def build_optimizer(optimizer_conf: OmegaConf, model: nn.Module) -> torch.optim.Optimizer:
    """Build the Optimizer

    Args:
        optimizer_conf (OmegaConf): The configuration
        model (nn.Module): The model to be optimized

    Raises:
        Exception: Unsupported Optimizer

    Returns:
        torch.optim.Optimizer: SGD or Adam optimizer
    """
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
    """Build the LR Scheduler

    Args:
        cfg (OmegaConf): The configuration
        optimizer (torch.optim.Optimizer): The optimizer

    Raises:
        Exception: Unsupported LR Scheduler

    Returns:
        torch.optim.lr_scheduler.LRScheduler: The scheduler
    """
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
            eta_min=cfg.scheduler.eta_min
        )
    elif cfg.scheduler.type == 'step':
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, 
            step_size=cfg.scheduler.step_size,
            gamma=0.1
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
    """Create the loss function

    Args:
        cfg (OmegaConf): The configuration

    Returns:
        nn.Module: CrossEntropyLoss object
    """
    return nn.CrossEntropyLoss(label_smoothing=cfg.regularization.label_smoothing)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> Tuple[float, float]:
    """Train the model for one epoch

    Args:
        model (nn.Module): The model to be trained
        loader (DataLoader): The data on which to train
        optimizer (torch.optim.Optimizer): The optimizer to update the model weights
        loss_fn (nn.Module): The loss function

    Returns:
        Tuple[float, float]: Average model loss and accuracy across the training epoch
    """
    global device
    
    model.train()
    
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(loader):
        # Unpack the batch
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        # Compute the predictions
        pred = model(x)
        
        # Compute the loss
        loss = loss_fn(pred, y)
        train_loss += loss.item() * x.size(0)
        correct += (pred.argmax(dim=1) == y).sum().item()
        total += y.size(0)
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
    
    avg_loss = train_loss / len(loader.dataset)
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc
    
    
def validate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> Tuple[float, float]:
    """Validate the model

    Args:
        model (nn.Module): The model
        loader (DataLoader): The validation data
        loss_fn (nn.Module): The loss function

    Returns:
        Tuple[float, float]: The model's average validation loss and accuracy for the epoch
    """
    global device
    
    model.eval()
    
    val_loss = 0.0
    correct = 0
    total = 0
    
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
    """Train a model, based on parameters in the config

    Args:
        cfg (OmegaConf): The configuration
    """
    global device
    
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
    kaiming_init(model)
    model.to(device)
    
    # Get the optimizer
    optimizer = build_optimizer(cfg.optimizer, model)
    
    # Get the LR scheduler
    scheduler = build_scheduler(cfg, optimizer)
    
    # Get the Loss Function
    loss_fn = get_loss_fn(cfg)
    
    best_acc = 0
    
    results = []
    
    # Train
    for epoch in range(cfg.training.max_epochs):
        # Train on the training data
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        # Validate on the validation data
        val_loss, val_acc = validate(model, val_loader, loss_fn)
        print(f"Epoch {epoch}:\tTrain Loss {train_loss}, Train Acc {train_acc}")
        print(f"\t\t\t:Val Loss {val_loss}, Val Acc {val_acc}")

        results.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{cfg.experiment.output_dir}/best.pt")
        # Advance the LR Scheduler
        scheduler.step()
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{cfg.experiment.output_dir}/log.csv")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    
    train(cfg)