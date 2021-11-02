
import numpy as np

import matplotlib.pyplot as plt
from plyfile import PlyData
from plyfile import PlyElement

colors=[[0, 0, 0],
[174, 199, 232],
[152, 223, 138],
[31, 119, 180 ],
[255, 187, 120],
[188, 189, 34],
[140, 86, 75 ],
[255, 152, 150],
[214, 39, 40],
[197, 176, 213],
[148, 103, 189],
[196, 156, 148],
[23, 190, 207],
[247, 182, 210],
[66, 188, 102],
[219, 219, 141],
[140, 57, 197],
[202, 185, 52],
[51, 176, 203],
[200, 54, 131],
[92, 193, 61],
[78,71, 183],
[172, 114, 82],
[255, 127, 14],
[91, 163, 138],
[153, 98, 156],
[140, 153, 101],
[158, 218, 229],
[100, 125, 154],
[178, 127, 135],
[146, 111, 194],
[44, 160, 44],
[112, 128, 144],
[96, 207, 209],
[227, 119, 194],
[213, 92, 176],
[94, 106, 211],
[82, 84, 163],
[100, 85, 144],
[0, 0, 255],
[0, 255, 0]]

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def save_as_ply(pointcloud, colors, file_path):
    num_vertices = pointcloud.shape[0]
    vertices = np.zeros(
        (num_vertices),
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("red", np.ubyte),
            ("green", np.ubyte),
            ("blue", np.ubyte),
        ],
    )
    for index in range(num_vertices):
        point = pointcloud[index,:]
        color = colors[index]
        vertices[index] = (*point, *color)
    el = PlyElement.describe(vertices, "vertex")
    PlyData([el], text=True).write(file_path)

if __name__ == '__main__':
    import os.path as osp
    from util.config import cfg
    cfg.task = 'test'
    from relation.data.scannetv2_inst import Dataset
    from data.dataset import Scannet
    dataset = Dataset(test=False)
    dataset.trainLoader()
    dataloader = dataset.train_data_loader

    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]
    cfg.dataset='train_weakly'
    train_files = Scannet(cfg.data_root, cfg.dataset, cfg.filename_suffix)
    for idx in range(20):
        xyz, rgb, label, _, _ = train_files[idx]

        unique_label = np.unique(label)
        label_to_cmap = {}
        for i, l in enumerate(unique_label):
            label_to_cmap[l] = i
        colored_label = [colors[label_to_cmap[l]] for l in label]
        save_as_ply(xyz, colored_label, osp.join("/result", f"base_data_{idx}.ply"))