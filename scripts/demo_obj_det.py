import logging
import open3d.ml as _ml3d

from open3d.ml.vis import Visualizer, BoundingBox3D, LabelLUT
from open3d.ml import datasets

import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo for inference of object detection')
    parser.add_argument('framework',
                        help='deep learning framework: tf or torch')
    parser.add_argument('--dataset_type',
                        help='Name of dataset class',
                        default="KITTI",
                        required=False)
    parser.add_argument('--dataset_path', help='path to dataset', required=True)
    parser.add_argument('--path_ckpt_pointpillars',
                        help='path to PointPillars checkpoint')
    parser.add_argument('--device',
                        help='device to run the pipeline',
                        default='gpu')

    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


def main(args):

    framework = _ml3d.utils.convert_framework_name(args.framework)
    args.device = _ml3d.utils.convert_device_name(args.device)
    if framework == 'torch':
        import open3d.ml.torch as ml3d
        from ml3d.torch.dataloaders import TorchDataloader as Dataloader
    else:
        import tensorflow as tf
        import open3d.ml.tf as ml3d

        from ml3d.tf.dataloaders import TFDataloader as Dataloader

        device = args.device
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if device == 'cpu':
                    tf.config.set_visible_devices([], 'GPU')
                elif device == 'gpu':
                    tf.config.set_visible_devices(gpus[0], 'GPU')
                else:
                    idx = device.split(':')[1]
                    tf.config.set_visible_devices(gpus[int(idx)], 'GPU')
            except RuntimeError as e:
                print(e)

    classname = getattr(datasets, args.dataset_type)
    dataset = classname(args.dataset_path)

    ObjectDetection = _ml3d.utils.get_module("pipeline", "ObjectDetection",
                                             framework)
    PointPillars = _ml3d.utils.get_module("model", "PointPillars", framework)
    cfg = _ml3d.utils.Config.load_from_file("ml3d/configs/pointpillars_" +
                                            args.dataset_type.lower() + ".yml")

    model = PointPillars(device=args.device, **cfg.model)

    pipeline = ObjectDetection(model, dataset, device=args.device)

    # load the parameters.
    pipeline.load_ckpt(ckpt_path=args.path_ckpt_pointpillars)

    test_split = Dataloader(dataset=dataset.get_split('training'),
                            preprocess=model.preprocess,
                            transform=None,
                            use_cache=False,
                            shuffle=False)

    # run inference on a single example.
    data = test_split[5]['data']
    result = pipeline.run_inference(data)[0]

    boxes = data['bbox_objs']
    boxes.extend(result)

    vis = Visualizer()

    lut = LabelLUT()
    for val in sorted(dataset.label_to_names.keys()):
        lut.add_label(val, val)

    # Uncommenting this assigns bbox color according to lut
    # for key, val in sorted(dataset.label_to_names.items()):
    #     lut.add_label(key, val)

    vis.visualize([{
        "name": args.dataset_type,
        'points': data['point']
    }],
                  lut,
                  bounding_boxes=boxes)

    # run inference on a multiple examples
    vis = Visualizer()
    lut = LabelLUT()
    for val in sorted(dataset.label_to_names.keys()):
        lut.add_label(val, val)

    boxes = []
    data_list = []
    for idx in tqdm(range(100)):
        data = test_split[idx]['data']

        result = pipeline.run_inference(data)[0]

        boxes = data['bbox_objs']
        boxes.extend(result)

        data_list.append({
            "name": args.dataset_type + '_' + str(idx),
            'points': data['point'],
            'bounding_boxes': boxes
        })

    vis.visualize(data_list, lut, bounding_boxes=None)


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
    )

    args = parse_args()
    main(args)
