# lightning-bolts related imports
import pytorch_lightning as pl
import numpy as np
import cv2
import os
import torch
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
from torch.utils.data import Dataset, DataLoader

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    import torchvision
else:  # pragma: no cover
    warn_missing_pkg('torchvision')
from typing import Any, Callable, List, Optional, Union
from face_recognition.utils.utils import str2list

__all__ = ['LitDataModule']


def LitDataModule(hparams):
    dm = FaceRecogDataModule(
        data_dir=hparams.data_dir,
        dataset_name=hparams.dataset_name,
        input_size=hparams.input_size,
        val_data_dir_list=hparams.val_data_dir_list,
        num_workers=hparams.num_workers,
        pin_memory=hparams.pin_memory,
        batch_size=int(hparams.batch_size/len(str2list(hparams.gpu_id))) if hparams.accelerator=='ddp' else hparams.batch_size,
        normalize=True,
        shuffle=True,
        drop_last=bool(hparams.drop_last)
    )

    dm.prepare_data()
    dm.setup()

    return dm


def img_loader(path):
    try:
        with open(path, 'rb') as f:

            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            return img
    except IOError:
        print('Cannot load image ' + path)


class Val_Dataset(Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader, n_fold=10):

        self.root = root
        self.file_list = file_list
        self.transform = transform
        self.loader = loader
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []

        with open(file_list) as f:
            pairs = f.read().splitlines()
        fold_len = len(pairs)/10
        for i, p in enumerate(pairs):
            p = p.split(' ')
            nameL = p[0]
            nameR = p[1]
            fold = i // fold_len
            flag = int(p[2])

            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)

    def __getitem__(self, index):

        img_l = self.loader(os.path.join(self.root, self.nameLs[index]))
        img_r = self.loader(os.path.join(self.root, self.nameRs[index]))
        imglist = [img_l, img_r]

        if self.transform is not None:
            imgs = [self.transform(i) for i in imglist]
        else:
            imgs = [torch.from_numpy(i) for i in imglist]

        imgs.append(torch.tensor(self.folds[index]))
        imgs.append(torch.tensor(self.flags[index]))

        return imgs

    def __len__(self):
        return len(self.nameLs)


class FaceRecogDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        dataset_name,
        input_size,
        val_data_dir_list,
        num_workers: int = 4,
        normalize: bool = False,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.val_data_dir_list = val_data_dir_list
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.dims = (3, input_size[0], input_size[1])

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, val, and test dataset
        """
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms(train=True)
            val_transforms = self.default_transforms(train=False)

            self.dataset_train = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, self.dataset_name), train_transforms)
            self.dataset_val = []
            for item in self.val_data_dir_list:
                val_data_dir = os.path.join(self.data_dir, item, item)
                file_list = os.path.join(self.data_dir, item, item+'_pair.txt')
                self.dataset_val.append(Val_Dataset(val_data_dir, file_list, transform=val_transforms))

        # if stage == "test" or stage is None:
        #     raise NotImplementedError
        #     self.dataset_test = None

    def default_transforms(self, train) -> Callable:
        if self.normalize:
            if train:
                defau_transforms = transform_lib.Compose([
                    transform_lib.RandomHorizontalFlip(),
                    transform_lib.ToTensor(),
                    transform_lib.Normalize(mean=(0.5,), std=(0.5,))
                ])
            else:
                defau_transforms = transform_lib.Compose([
                    transform_lib.ToTensor(),
                    transform_lib.Normalize(mean=(0.5,), std=(0.5,))
                ])
        else:
            defau_transforms = transform_lib.Compose([transform_lib.ToTensor()])
        return defau_transforms

    @property
    def num_classes(self) -> int:
        return len(self.dataset_train.class_to_idx)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        return self._data_loader(self.dataset_train, shuffle=self.shuffle, drop_last=self.drop_last, pin_memory=self.pin_memory)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        val_dataloader_list = []
        for item in self.dataset_val:
            val_dataloader_list.append(self._data_loader(item, pin_memory=self.pin_memory))
        return val_dataloader_list

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False, drop_last: bool = False, pin_memory: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory
        )

