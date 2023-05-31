import torch
from torch.utils.data import Dataset
from typing import Tuple

TRAIN_TYPE = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]

VAL_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]

SHOW_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


class Base_Dataset(Dataset):
    """


    """

    def __init__(self,
                 args,
                 mode='train'):
        assert mode in ['train', 'val', 'show'], 'dataset mode is error'
        self.mode = mode

    def get_train_data(self, item: int) -> TRAIN_TYPE:

        pass

    def get_val_data(self, item: int) -> VAL_TYPE:
        pass

    def get_show_data(self, item: int) -> SHOW_TYPE:
        pass

    def __getitem__(self, item):
        if self.mode == 'train':
            return self.get_train_data(item)
        elif self.mode == 'val':
            return self.get_val_data(item)
        elif self.mode == 'show':
            return self.get_show_data(item)

    def __len__(self):
        pass
