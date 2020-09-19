import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from tqdm import tqdm
import random
import shutil
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Split large pointclouds in Semantic3D.')
    parser.add_argument('--dataset_path',
                        help='path to Semantic3D',
                        required=True)
    parser.add_argument('--out_path',
                        help='Output path', default=None)

    parser.add_argument('--size_limit',
                        help='Maximum size of processed pointcloud in Megabytes.', default=2000, type=int)

    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args

def preprocess(args):
    # Split large pointclouds into multiple point clouds.

    dataset_path = args.dataset_path
    out_path = args.out_path
    size_limit = args.size_limit  #Size in mega bytes.

    if out_path is None:
        print("out_path not give, Saving output in {}".format(dataset_path))
        out_path = dataset_path

    all_files = glob.glob(str(Path(dataset_path) / '*.txt'))

    train_files = [
        f for f in all_files if exists(
            str(Path(f).parent / Path(f).name.replace('.txt', '.labels')))
    ]

    files = {}
    for f in train_files:
        size = Path(f).stat().st_size / 1e6
        if size <= size_limit:
            files[f] = 1
            continue
        else:
            parts = int(size / size_limit) + 1
            files[f] = parts

    os.makedirs(out_path, exist_ok=True)

    for key, parts in tqdm(files.items()):
        if parts == 1:
            if dataset_path != out_path:
                shutil.copyfile(key, join(out_path, Path(key).name))
                shutil.copyfile(key.replace('.txt', '.labels'), join(out_path, Path(key).name.replace('.txt', '.labels')))
            continue
        print("Splitting {} into {} parts".format(Path(key).name, parts))
        pc = pd.read_csv(key,
                         header=None,
                         delim_whitespace=True,
                         dtype=np.float32).values

        labels = pd.read_csv(key.replace(".txt", ".labels"),
                             header=None,
                             delim_whitespace=True,
                             dtype=np.int32).values
        labels = np.array(labels, dtype=np.int32).reshape((-1,))

        axis = 1  # Longest axis.

        inds = pc[:, axis].argsort()
        pc = pc[inds]
        labels = labels[inds]
        pcs = np.array_split(pc, parts)
        lbls = np.array_split(labels, parts)
        for i in range(parts):
            name = join(
                out_path,
                Path(key).name.replace('.txt', '_part_{}.txt'.format(i)))
            name_lbl = name.replace('.txt', '.labels')

            shuf = np.arange(pcs[i].shape[0])
            np.random.shuffle(shuf)

            np.savetxt(name, pcs[i][shuf])
            np.savetxt(name_lbl, lbls[i][shuf])


if __name__ == '__main__':
    args = parse_args()
    preprocess(args)