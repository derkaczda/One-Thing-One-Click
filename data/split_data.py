import os.path as osp
import argparse
import os
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, help="Path to the preprocessed data")
    parser.add_argument("--split-file", type=str, help="Path to the txt file containing the split information")
    parser.add_argument("--dest", type=str, help="Path where to store the data")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    file_postfix = "_vh_clean_2.ply_inst_nostuff.pth"
    with open(args.split_file, "r") as f:
        lines = f.readlines()

    for l in lines:
        print(l[:-1])
        l = l[:-1]
        src = osp.join(args.data_root, f"{l}{file_postfix}")
        dst = osp.join(args.dest, f"{l}{file_postfix}")
        shutil.copyfile(src, dst)



