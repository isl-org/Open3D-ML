import numpy as np
import glob, os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import read_ply
from helper_tool import Plot

if __name__ == '__main__':
    base_dir = '/data/S3DIS/results'
    original_data_dir = '/data/S3DIS/original_ply'
    data_path = glob.glob(os.path.join(base_dir, '*.ply'))
    data_path = np.sort(data_path)

    test_total_correct = 0
    test_total_seen = 0
    gt_classes = [0 for _ in range(13)]
    positive_classes = [0 for _ in range(13)]
    true_positive_classes = [0 for _ in range(13)]
    visualization = False

    for file_name in data_path:
        pred_data = read_ply(file_name)
        pred = pred_data['pred']
        original_data = read_ply(os.path.join(original_data_dir, file_name.split('/')[-1][:-4] + '.ply'))
        labels = original_data['class']
        points = np.vstack((original_data['x'], original_data['y'], original_data['z'])).T

        ##################
        # Visualize data #
        ##################
        if visualization:
            colors = np.vstack((original_data['red'], original_data['green'], original_data['blue'])).T
            xyzrgb = np.concatenate([points, colors], axis=-1)
            Plot.draw_pc(xyzrgb)  # visualize raw point clouds
            Plot.draw_pc_sem_ins(points, labels)  # visualize ground-truth
            Plot.draw_pc_sem_ins(points, pred)  # visualize prediction

        correct = np.sum(pred == labels)
        print(str(file_name.split('/')[-1][:-4]) + '_acc:' + str(correct / float(len(labels))))
        test_total_correct += correct
        test_total_seen += len(labels)

        for j in range(len(labels)):
            gt_l = int(labels[j])
            pred_l = int(pred[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l == pred_l)

    iou_list = []
    for n in range(13):
        iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
        iou_list.append(iou)
    mean_iou = sum(iou_list) / 13.0
    print('eval accuracy: {}'.format(test_total_correct / float(test_total_seen)))
    print('mean IOU:{}'.format(mean_iou))
    print(iou_list)

    acc_list = []
    for n in range(13):
        acc = true_positive_classes[n] / float(gt_classes[n])
        acc_list.append(acc)
    mean_acc = sum(acc_list) / 13.0
    print('mAcc value is :{}'.format(mean_acc))
