import logging
from os.path import join
import argparse
import pickle
from open3d.ml.datasets import utils
from open3d.ml import datasets
import multiprocessing


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect bounding boxes for augmentation.')
    parser.add_argument('--dataset_path',
                        help='path to Dataset root',
                        required=True)
    parser.add_argument(
        '--out_path',
        help='Output path to store pickle (default to dataet_path)',
        default=None,
        required=False)
    parser.add_argument('--dataset_type',
                        help='Name of dataset class',
                        default="KITTI",
                        required=False)
    parser.add_argument('--num_cpus',
                        help='Name of dataset class',
                        type=int,
                        default=multiprocessing.cpu_count(),
                        required=False)

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


def process_boxes(i):
    data = train.get_data(i)
    bbox = data['bounding_boxes']
    flat_bbox = [box.to_xyzwhlr() for box in bbox]
    indices = utils.operations.points_in_box(data['point'], flat_bbox)
    bboxes = []
    for i, box in enumerate(bbox):
        pts = data['point'][indices[:, i]]
        box.points_inside_box = pts
        bboxes.append(box)
    return bboxes


if __name__ == '__main__':
    """Collect bboxes

    This script constructs a bbox dictionary for later data augmentation.

    Args:
        dataset_path (str): Directory to load dataset data.
        out_path (str): Directory to save pickle file (infos).
        dataset_type (str): Name of dataset object under `ml3d/datasets` to use to 
                            load the test data split from the dataset folder. Uses 
                            reflection to dynamically import this dataset object by name. 
                            Default: KITTI
        num_cpus (int): Number of threads to use.
                        Default: All CPU cores.

    Example usage:

    python scripts/collect_bboxes.py --dataset_path /path/to/data --dataset_type MyCustomDataset
    """

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
    )

    args = parse_args()
    out_path = args.out_path
    if out_path is None:
        out_path = args.dataset_path
    classname = getattr(datasets, args.dataset_type)
    dataset = classname(args.dataset_path)
    train = dataset.get_split('train')

    print("Found", len(train), "traning samples")
    print("This may take a few minutes...")
    with multiprocessing.Pool(args.num_cpus) as p:
        bboxes = p.map(process_boxes, range(len(train)))
        bboxes = [e for l in bboxes for e in l]
        file = open(join(out_path, 'bboxes.pkl'), 'wb')
        pickle.dump(bboxes, file)
