import torch
import glob
import os.path as osp


class Scannet(torch.utils.data.Dataset):
    def __init__(self, data_root, dataset, filename_suffix, triple=False):
        self.data_root = data_root
        self.dataset = dataset
        self.filename_suffix = filename_suffix
        self.filenames = sorted(
            glob.glob(
                osp.join(self.data_root, self.dataset, "*" + self.filename_suffix)
            )
        )
        # self.filenames = self.filenames[:100]
        self.filenames_three = []
        for x in self.filenames:
            self.filenames_three.append(x)
            if triple:
                self.filenames_three.append(x)
                self.filenames_three.append(x)
        
        

    def __len__(self) -> int:
        return len(self.filenames_three)

    def __getitem__(self, index):
        file = self.filenames_three[index]
        return torch.load(file)