import torch
import numpy as np

from .base_model import BaseModel
from ...utils import MODEL
from ..dataloaders import DefaultBatcher
from ..modules.losses import filter_valid_label
from ...datasets.utils import transforms


class PointNet(BaseModel):

    def __init__(self,
                 name='PointNet',
                 num_points=1024,
                 num_classes=16,
                 ignored_label_inds=[],
                 dim_input=3,
                 dim_feature=0,
                 normalize=True,
                 augment='uniform',
                 feature_transform_regularization=0.001,
                 batcher='DefaultBatcher',
                 ckpt_path=None,
                 task='classification',
                 **kwargs):
        super().__init__(
            name=name,
            num_points=num_points,
            num_classes=num_classes,
            ignored_label_inds=ignored_label_inds,
            dim_input=dim_input,
            dim_feature=dim_feature,
            normalize=normalize,
            augment=augment,
            feature_transform_regularization=feature_transform_regularization,
            batcher=batcher,
            ckpt_path=ckpt_path,
            task=task,
            **kwargs)

        if task == 'classification':
            self.net = Classifier(num_points, num_classes, dim_input,
                                  dim_feature)
        elif task == 'segmentation':
            self.net = SceneSegmenter(num_points, num_classes, dim_input,
                                      dim_feature)
        else:
            raise ValueError(f"Invalid task {task}")
        assert augment in ['gaussian',
                           'uniform'], f"Invalid augmentation {augment}"

        self.num_points = num_points
        self.normalize = normalize
        self.augment = augment
        self.inference_data = None
        self.inference_result = None
        self.feature_transform_regularization = feature_transform_regularization
        self.task = task
        self.batcher = DefaultBatcher()

    def forward(self, inputs):
        return self.net(inputs['point'].to(self.device))

    def get_loss(self, Loss, results, inputs, device):
        cfg = self.cfg
        labels = inputs['data']['labels']

        # Todo: Is this valid for classification?
        scores, labels = filter_valid_label(results[0], labels, cfg.num_classes,
                                            cfg.ignored_label_inds, device)

        loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        if self.net.feat.feature_transform:
            loss += transform_regularizer(
                results[2], reg_weight=self.feature_transform_regularization)

        return loss, labels, scores

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg_pipeline.adam_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg_pipeline.max_epoch // 10,
            gamma=cfg_pipeline.scheduler_gamma)
        return optimizer, scheduler

    # Todo: Parameters not consistent for base class and usage
    def preprocess(self, data, attr):
        return data

    # Todo: Parameters not consistent for base class and usage
    def transform(self, data, attr):
        point = data['point'].copy()

        # Sub/super-sample to get consistent point cloud size for batching
        # Todo: How to run semantic segmentation inference on different point cloud sizes?
        choice = np.random.choice(len(point), self.num_points, replace=True)
        point = point[choice, :]

        if self.task == 'classification':
            # Normalize and center
            if self.normalize:
                point, _ = transforms.trans_normalize(
                    point, feat=None, t_normalize={'method': 'unit_sphere'})

            # Data augmentation for classification training
            if self.augment and attr['split'] in ['train', 'training']:
                # Paper: "[...] Gaussian noise with zero mean and 0.02 standard deviation."
                if self.augment == 'gaussian':
                    point = transforms.trans_augment(
                        point, {
                            'rotation_method': 'vertical',
                            'noise_type': 'gaussian',
                            'noise_level': 0.002
                        })
                # Following original implementation at
                # https://github.com/charlesq34/pointnet/blob/master/train.py
                elif self.augment == 'uniform':
                    point = transforms.trans_augment(point, {
                        'rotation_method': 'vertical',
                        'noise_type': 'uniform_clip'
                    })

        if data.get('feat') is not None:
            point = np.concatenate([point, data['feat'][choice].copy()], axis=1)

        label = data['label'] if self.task == 'classification' else data[
            'label'][choice]

        return {'point': point.T, 'labels': label}

    def inference_begin(self, data):
        attr = {'split': 'test'}
        self.inference_data = self.preprocess(data, attr)

    def inference_preprocess(self):
        attr = {'split': 'test'}
        data = self.transform(self.inference_data, attr)
        inputs = {'data': data, 'attr': attr}
        return self.batcher.collate_fn([inputs])

    def inference_end(self, inputs, results):
        # Todo: Correct?
        if self.task == 'classification':
            inference_result = {
                'predict_labels':
                    np.array([results[0].argmax().detach().cpu().numpy()]),
                'predict_scores':
                    torch.nn.functional.softmax(results[0],
                                                dim=1).detach().cpu().numpy()
            }
        else:
            inference_result = {
                'predict_labels':
                    results[0].argmax(dim=-1).detach().cpu().numpy().squeeze(),
                'predict_scores':
                    torch.nn.functional.softmax(
                        results[0], dim=-1).detach().cpu().numpy().squeeze()
            }
        self.inference_result = inference_result
        return True


#######################################################################
#                            PointNet                                 #
#######################################################################
# Adapted from https://github.com/fxia22/pointnet.pytorch


def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class Transform(torch.nn.Module):
    # Following original implementation at https://github.com/charlesq34/pointnet/blob/master/models/transform_nets.py
    def __init__(self, num_points=1024, k=3):
        super(Transform, self).__init__()

        self.num_points = num_points
        self.k = k

        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp = torch.nn.MaxPool1d(num_points)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k * k)
        self.relu = torch.nn.ReLU()

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)

        # Initialize weights with Xavier as in original implementation (PyTorch default is Kaiming He)
        self.apply(init_weights)

        # Initialize output of transformation as identity matrix
        self.fc3.weight.data.fill_(0.0)
        self.fc3.bias.data = torch.eye(self.k).view(-1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.mp(x)
        x = x.view(-1, 1024)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Todo: Is this identical to original implementation?
        eye = torch.eye(self.k,
                        device=x.device).view(1, -1).repeat(x.size()[0], 1)
        x = x + eye
        return x.view(-1, self.k, self.k)


class Feature(torch.nn.Module):
    # Following original implementation at https://github.com/charlesq34/pointnet/blob/master/models/pointnet_cls.py
    def __init__(self,
                 num_points=1024,
                 dim_input=3,
                 dim_feature=0,
                 global_feature=True,
                 input_transform=True,
                 feature_transform=True):
        super(Feature, self).__init__()

        # "Shared MLP" implemented as Conv 1D with 1x1 kernels and "channels first" inputs
        self.conv1 = torch.nn.Conv1d(dim_input + dim_feature, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(1024)
        self.mp = torch.nn.MaxPool1d(num_points)
        self.relu = torch.nn.ReLU()

        # Initialize weights with Xavier as in original implementation (PyTorch default is Kaiming He)
        self.apply(init_weights)

        self.in_trans = Transform(num_points=num_points, k=dim_input)
        self.feat_trans = Transform(num_points=num_points, k=64)

        self.num_points = num_points
        self.dim_input = dim_input
        self.dim_feature = dim_feature
        self.global_feature = global_feature
        self.input_transform = input_transform
        self.feature_transform = feature_transform

    def forward(self, x):
        if self.input_transform:
            # Only transform spatial input dimensions
            if self.dim_feature:
                x_feat = x[:, self.dim_input:]
                x = x[:, :self.dim_input]

            t_in = self.in_trans(x)
            x = torch.bmm(t_in, x)

            if self.dim_feature:
                x = torch.cat([x, x_feat], dim=1)
        else:
            t_in = None

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        if self.feature_transform:
            t_feat = self.feat_trans(x)
            x = torch.bmm(t_feat, x)
        else:
            t_feat = None

        point_feature = x

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.mp(x)
        x = x.view(-1, 1024)

        if self.global_feature:
            return x, t_in, t_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            x = torch.cat([x, point_feature], dim=1)
            return x, t_in, t_feat


class Classifier(torch.nn.Module):
    # Following original implementation at https://github.com/charlesq34/pointnet/blob/master/models/pointnet_cls.py
    def __init__(self,
                 num_points=1024,
                 num_classes=16,
                 dim_input=3,
                 dim_feature=0,
                 dropout=0.3):
        super(Classifier, self).__init__()

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, num_classes)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()

        # Initialize weights with Xavier as in original implementation (PyTorch default is Kaiming He)
        self.apply(init_weights)

        self.feat = Feature(num_points, dim_input, dim_feature)

        self.num_points = num_points

    def forward(self, x):
        x, t_in, t_feat = self.feat(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x, t_in, t_feat


class SceneSegmenter(torch.nn.Module):
    # Following original implementation at https://github.com/charlesq34/pointnet/blob/master/models/pointnet_seg.py
    def __init__(self,
                 num_points=3000,
                 num_classes=50,
                 dim_input=3,
                 dim_feature=0):
        super(SceneSegmenter, self).__init__()

        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, num_classes, 1)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.relu = torch.nn.ReLU()

        # Initialize weights with Xavier as in original implementation (PyTorch default is Kaiming He)
        self.apply(init_weights)

        self.feat = Feature(num_points,
                            dim_input,
                            dim_feature,
                            global_feature=False)

        self.num_points = num_points
        self.num_classes = num_classes

    def forward(self, x):
        x, t_in, t_feat = self.feat(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        x = x.transpose(2, 1).contiguous()
        x = x.view(-1, self.num_points, self.num_classes)
        return x, t_in, t_feat


class PartSegmenter(torch.nn.Module):

    def __init__(self):
        super(PartSegmenter, self).__init__()
        raise NotImplementedError


def transform_regularizer(transform, reg_weight=0.001):
    eye = torch.eye(transform.size()[1], device=transform.device)[None, :, :]
    return reg_weight * torch.mean(
        torch.norm(torch.bmm(transform, transform.transpose(2, 1)) - eye,
                   dim=(1, 2)))


MODEL._register_module(PointNet, 'torch')
