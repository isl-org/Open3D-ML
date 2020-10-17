from open3d.ml.datasets import (SemanticKITTI, ParisLille3D, Semantic3D, S3DIS,
                                Toronto3D)
import argparse
import yaml
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Read from datasets')
    parser.add_argument('--path_semantickitti',
                        help='path to semantiSemanticKITTI')
    parser.add_argument('--path_semantick3d', help='path to Semantic3D')
    parser.add_argument('--path_parislille3d', help='path to ParisLille3D')
    parser.add_argument('--path_toronto3d', help='path to Toronto3D')
    parser.add_argument('--path_s3dis', help='path to S3DIS')

    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


def demo_dataset(args):
    # read data from datasets
    datasets = []
    if args.path_semantickitti is not None:
        datasets.append(
            SemanticKITTI(dataset_path=args.path_semantickitti,
                          use_cache=False))
    if args.path_parislille3d is not None:
        datasets.append(
            ParisLille3D(dataset_path=args.path_parislille3d, use_cache=False))
    if args.path_toronto3d is not None:
        datasets.append(
            Toronto3D(dataset_path=args.path_toronto3d, use_cache=False))
    if args.path_semantick3d is not None:
        datasets.append(
            Semantic3D(dataset_path=args.path_semantick3d, use_cache=False))
    if args.path_s3dis is not None:
        datasets.append(S3DIS(dataset_path=args.path_s3dis, use_cache=False))

    for dataset in datasets:
        print(dataset.name)
        cat_num = len(dataset.label_to_names)
        num_labels = np.zeros([cat_num])

        split = dataset.get_split('train')
        for i in range(len(split)):
            data = split.get_data(i)
            labels = data['label']
            for l in range(cat_num):
                num_labels[l] += (labels == l).sum()

        print(num_labels)

    for dataset in datasets:
        print(dataset.label_to_names)
        # print names of all pointcould
        split = dataset.get_split('test')
        for i in range(len(split)):
            attr = split.get_attr(i)
            print(attr['name'])

        split = dataset.get_split('train')
        for i in range(len(split)):
            data = split.get_data(i)
            print(data['point'].shape)


if __name__ == '__main__':
    args = parse_args()
    demo_dataset(args)
