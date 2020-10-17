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
from open3d.ml.datasets import utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split large pointclouds in Semantic3D.')
    parser.add_argument('--dataset_path',
                        help='path to Semantic3D',
                        required=True)
    parser.add_argument('--out_path', help='Output path', default=None)

    parser.add_argument(
        '--size_limit',
        help='Maximum size of processed pointcloud in Megabytes.',
        default=2000,
        type=int)

    args = parser.parse_args()

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
        out_path = Path(dataset_path) / 'processed'
        print("out_path not give, Saving output in {}".format(out_path))

    all_files = glob.glob(str(Path(dataset_path) / '*.txt'))

    train_files = [
        f for f in all_files
        if exists(str(Path(f).parent / Path(f).name.replace('.txt', '.labels')))
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

    sub_grid_size = 0.01

    for key, parts in tqdm(files.items()):

        if parts == 1:
            if dataset_path != out_path:
                pc = pd.read_csv(key,
                                 header=None,
                                 delim_whitespace=True,
                                 dtype=np.float32).values

                labels = pd.read_csv(key.replace(".txt", ".labels"),
                                     header=None,
                                     delim_whitespace=True,
                                     dtype=np.int32).values
                labels = np.array(labels, dtype=np.int32).reshape((-1,))

                print(pc.shape, labels.shape)
                points, feat, labels = utils.DataProcessing.grid_subsampling(
                    pc[:, :3],
                    features=pc[:, 3:],
                    labels=labels,
                    grid_size=sub_grid_size)
                pc = np.concatenate([points, feat], 1)
                print(pc.shape, labels.shape)

                name = join(out_path, Path(key).name.replace('.txt', '.txt'))
                name_lbl = name.replace('.txt', '.labels')

                np.savetxt(name, pc, fmt='%.3f %.3f %.3f %i %i %i %i')
                np.savetxt(name_lbl, labels, fmt='%i')

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

        for i, pc in enumerate(pcs):
            lbl = lbls[i]
            print(pc.shape, lbl.shape)
            pc, feat, lbl = utils.DataProcessing.grid_subsampling(
                pc[:, :3],
                features=pc[:, 3:],
                labels=lbl,
                grid_size=sub_grid_size)
            pcs[i] = np.concatenate([pc, feat], 1)
            lbls[i] = lbl
            print(pc.shape, lbl.shape)

        for i in range(parts):
            name = join(
                out_path,
                Path(key).name.replace('.txt', '_part_{}.txt'.format(i)))
            name_lbl = name.replace('.txt', '.labels')

            shuf = np.arange(pcs[i].shape[0])
            np.random.shuffle(shuf)

            np.savetxt(name, pcs[i][shuf], fmt='%.3f %.3f %.3f %i %i %i %i')
            np.savetxt(name_lbl, lbls[i][shuf], fmt='%i')


if __name__ == '__main__':
    args = parse_args()
    preprocess(args)
