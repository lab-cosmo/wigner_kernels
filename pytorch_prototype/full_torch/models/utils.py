import os
import pytorch_lightning as pl

from torch_geometric.data import DataLoader

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import mse_loss, l1_loss


import numpy as np
from sklearn.model_selection import train_test_split


def train_val_test_split(dset_len,val_ratio,test_ratio, seed=None, order=None):
    shuffle = True if order is None else False
    valtest_ratio = val_ratio + test_ratio
    idx_train = list(range(dset_len))
    idx_test = []
    idx_val = []
    if valtest_ratio > 0 and dset_len > 0:
        idx_train, idx_tmp = train_test_split(range(dset_len), test_size=valtest_ratio, random_state=seed, shuffle=shuffle)
        if test_ratio == 0:
            idx_val = idx_tmp
        elif val_ratio == 0:
            idx_test = idx_tmp
        else:
            test_val_ratio = test_ratio / (test_ratio + val_ratio)
            idx_val, idx_test = train_test_split(idx_tmp, test_size=test_val_ratio, random_state=seed, shuffle=shuffle)

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def make_splits(dataset_len, val_ratio, test_ratio, seed=None, filename=None, splits=None, order=None):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits['idx_train']
        idx_val = splits['idx_val']
        idx_test = splits['idx_test']
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, val_ratio, test_ratio, seed, order
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return torch.from_numpy(idx_train), torch.from_numpy(idx_val), torch.from_numpy(idx_test)


class Model(pl.LightningModule):
    def __init__(self, #model: pl.LightningModule,
    lr:float =1e-4, weight_decay:float=0, lr_factor:float=0.8,
    lr_patience:int=10, lr_min:float=1e-7, target_name='forces',
    lr_warmup_steps:int=0,
    test_interval:int=1,
):
        super(Model, self).__init__()
        #self.model = model
        self.losses = None
        # self.derivative = model.derivative
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_min = lr_min
        self.target_name = target_name
        self.lr_warmup_steps = lr_warmup_steps
        self.test_interval = test_interval

    def configure_optimizers(self):
        optimizer = AdamW(#self.model.parameters(),
        self.parameters(),
        lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            'min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.lr_min
        )
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'validation_loss',
                        'interval': 'epoch',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def training_step(self, data, batch_idx):
        loss = self.step(data, 'training')
        return loss


    def validation_step(self, data, batch_idx):
        loss = self.step(data, 'validation')
        return loss

    def test_step(self, data, batch_idx):
        loss = self.step(data, 'test')
        return loss


    def step(self, data, stage):
        with torch.set_grad_enabled(stage == 'train' or self.derivative):
            pred = self(data.z, data.pos, data.batch)

        loss = 0
        facs = {'forces': 1.}
        for k,fac in facs.items():
            loss += fac * (pred[k] - data[k]).pow(2).mean()

        # Add sync_dist=True to sync logging across all GPU workers
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

