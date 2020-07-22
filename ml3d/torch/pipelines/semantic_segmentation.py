#coding: future_fstrings
import torch, pickle
import torch.nn as nn
import helper_torch_util 
import numpy as np
from pprint import pprint
from datetime import datetime
from tqdm import tqdm
from sklearn.neighbors import KDTree
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler

from os.path import exists, join, isfile, dirname, abspath

from ml3d.torch.datasets import BaseDataset
from ml3d.datasets.semantickitti import DataProcessing
from ml3d.torch.utils import make_dir


import yaml


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)

def intersection_over_union(scores, labels):
    r"""
        Compute the per-class IoU and the mean IoU # TODO: complete doc

        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels

        Returns
        -------
        list of floats of length num_classes+1 (last item is mIoU)
    """
    num_classes = scores.size(-2) 
    predictions = torch.max(scores, dim=-2).indices

    ious = []

    for label in range(num_classes):
        pred_mask = predictions == label
        labels_mask = labels == label
        iou = (pred_mask & labels_mask).float().sum() 
        iou = iou / (pred_mask | labels_mask).float().sum()
        ious.append(iou.cpu().item())
    ious.append(np.nanmean(ious))
    return ious


def accuracy(scores, labels):
    r"""
        Compute the per-class accuracies and the overall accuracy 

        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels

        Returns
        -------
        list of floats of length num_classes+1 
        (last item is overall accuracy)
    """
    num_classes = scores.size(-2) 

    predictions = torch.max(scores, dim=-2).indices

    accuracies = []

    accuracy_mask = predictions == labels
    for label in range(num_classes):
        label_mask = labels == label
        per_class_accuracy = (accuracy_mask & label_mask).float().sum()
        per_class_accuracy /= label_mask.float().sum()
        accuracies.append(per_class_accuracy.cpu().item())
    # overall accuracy
    accuracies.append(accuracy_mask.float().mean().cpu().item())
    #accuracies = np.array(accuracies)
    return accuracies


class SemanticSegmentation():
    def __init__(self, model, dataset, cfg):
        self.model      = model
        self.dataset    = dataset
        self.cfg        = cfg

        make_dir(cfg.main_log_dir) 
        cfg.logs_dir = join(cfg.main_log_dir, 
                                    cfg.model_name)
        make_dir(cfg.logs_dir) 

        # dataset.cfg.num_points = model.cfg.num_points

    def run_inference(self, points, device):
        cfg     = self.cfg
        model   = self.model

        model.to(device)
        model.eval()

        with torch.no_grad():
            inputs  = model.preprocess_inference(points, device)
            scores  = model(inputs)
            pred    = torch.max(scores.squeeze(0), dim=-1).indices
            # pred    = pred.cpu().data.numpy()

        return pred


    def run_test(self, device):
        #self.device = device
        model   = self.model
        dataset = self.dataset
        cfg     = self.cfg
        model.to(device)
        model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_file_path   = join(cfg.logs_dir, 
                                'log_test_'+ timestamp + '.txt')
        log_file        = open(log_file_path, 'w')
        self.log_file   = log_file


        test_sampler    = dataset.get_sampler(cfg.test_batch_size, 'test')
        test_loader = DataLoader(test_sampler, batch_size=cfg.test_batch_size)
        test_probs  = [np.zeros(shape=[len(l), self.model.cfg.num_classes], 
                        dtype=np.float16) for l in dataset.possibility]

        self.load_ckpt(model.cfg.ckpt_path, False)

        test_smooth = 0.98
        epoch       = 0

        with torch.no_grad():
            while True:
                for batch_data in tqdm(test_loader, desc='test', leave=False):
                    # loader: point_clout, label
                    inputs          = model.preprocess(batch_data, device) 
                    result_torch    = model(inputs)
                    result_torch    = torch.reshape(result_torch,
                                                (-1, model.cfg.num_classes))

                    m_softmax       = nn.Softmax(dim=-1)
                    result_torch    = m_softmax(result_torch)
                    stacked_probs   = result_torch.cpu().data.numpy()

                    stacked_probs = np.reshape(stacked_probs, 
                                                [cfg.test_batch_size, 
                                                model.cfg.num_points, 
                                                model.cfg.num_classes])
                  
                    point_inds  = inputs['input_inds']
                    cloud_inds  = inputs['cloud_inds']

                    for j in range(np.shape(stacked_probs)[0]):
                        probs = stacked_probs[j, :, :]
                        inds = point_inds[j, :]
                        c_i = cloud_inds[j][0]
                        test_probs[c_i][inds] = \
                                    test_smooth * test_probs[c_i][inds] + \
                                    (1 - test_smooth) * probs

               

                new_min = np.min(dataset.min_possibility)
                log_out(f"Epoch {epoch:3d}, end. "
                        f"Min possibility = {new_min:.1f}", log_file)
               
                if np.min(dataset.min_possibility) > 0.5:  # 0.5
                    log_out(f"\nReproject Vote #"
                            f"{int(np.floor(new_min)):d}", log_file)
                    dataset.save_test_result(test_probs, 
                                        str(dataset.cfg.test_split_number))
                    log_out(f"{str(dataset.cfg.test_split_number)}"
                            f" finished", log_file)
                    return
              
                epoch += 1

    def run_train(self, device):
        model   = self.model
        dataset = self.dataset
        cfg     = self.cfg
        model.to(device) 
        model.device = device

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_file_path = join(cfg.logs_dir, 
                        'log_train_'+ timestamp + '.txt')
        log_file = open(log_file_path, 'w')
        self.log_file   = log_file

        n_samples       = torch.tensor(dataset.cfg.class_weights, 
                            dtype=torch.float, device=device)
        ratio_samples   = n_samples / n_samples.sum()
        weights         = 1 / (ratio_samples + 0.02)
        criterion       = nn.CrossEntropyLoss(weight=weights)


        train_split     = BaseDataset(dataset=dataset.get_split('training'), 
                            preprocess=model.preprocess,
                            transform=model.transform, shuffle=True)
        train_loader    = DataLoader(train_split,
                            batch_size=cfg.batch_size, shuffle=True,
                            pin_memory=True)

        '''
        valid_sampler   = dataset.get_sampler(cfg.val_batch_size, 'validation')
        valid_loader    = DataLoader(valid_sampler, 
                                     batch_size=cfg.val_batch_size)
        train_sampler   = dataset.get_sampler(cfg.batch_size, 'training')
        train_loader    = DataLoader(train_sampler, 
                                     batch_size=cfg.batch_size)
        '''

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.adam_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                cfg.scheduler_gamma)

        self.optimizer, self.scheduler = optimizer, scheduler

        if hasattr(model.cfg, 'ckpt_path'):
            first_epoch = self.load_ckpt(model.cfg.ckpt_path, True)
        
        writer = SummaryWriter(join(cfg.logs_dir, 
                                cfg.train_sum_dir))
    
        for epoch in range(0, cfg.max_epoch+1):
            print(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')

            # --------------------- train
            model.train()
            self.losses = []
            self.accs   = []
            self.ious   = []
            step        = 0
            #for batch_data in tqdm(train_loader, 
            #                            desc='Training', leave=False):
            for idx, inputs in enumerate(tqdm(train_loader)):
                
                scores  = model(inputs['data'])

                labels  = inputs['data']['label'].to(device)
                scores, labels = self.filter_valid(scores, labels, device)


                logp = torch.distributions.utils.probs_to_logits(scores, 
                                                        is_binary=False)
                
                optimizer.zero_grad()
                loss = criterion(logp, labels)
                loss.backward()
                optimizer.step()

                acc  = accuracy(scores, labels)
                iou  = intersection_over_union(scores, labels)
                self.losses.append(loss.cpu().item())
                self.accs.append(accuracy(scores, labels))
                self.ious.append(intersection_over_union(scores, labels))

                step = step + 1

            scheduler.step()

            # --------------------- validation
            model.eval()
            self.valid_losses    = []
            self.valid_accs      = []
            self.valid_ious      = []
            step                 = 0
            with torch.no_grad():
                for batch_data in tqdm(valid_loader, 
                                            desc='validation', leave=False):

                    inputs = model.preprocess(batch_data, device) 
                    # scores: B x N x num_classes
                    scores = model(inputs)

                    labels = batch_data[1] 
                    scores, labels = self.filter_valid(scores, labels, device)


                    logp = torch.distributions.utils.probs_to_logits(scores, 
                                                            is_binary=False)
                    
                    loss = criterion(logp, labels)
                    acc  = accuracy(scores, labels)
                    iou  = intersection_over_union(scores, labels)

                    self.valid_losses.append(loss.cpu().item())
                    self.valid_accs.append(accuracy(scores, labels))
                    self.valid_ious.append(intersection_over_union(scores, 
                                                                    labels))

                    step = step + 1

            self.save_logs(writer, epoch)

            if epoch % cfg.save_ckpt_freq == 0:
                path_ckpt = join(self.cfg.logs_dir, 'checkpoint')
                self.save_ckpt(path_ckpt, epoch)


    def save_logs(self, writer, epoch):

        accs = np.nanmean(np.array(self.accs), axis=0)
        ious = np.nanmean(np.array(self.ious), axis=0)

        valid_accs = np.nanmean(np.array(self.valid_accs), axis=0)
        valid_ious = np.nanmean(np.array(self.valid_ious), axis=0)


        loss_dict = {
            'Training loss':    np.mean(self.losses),
            'Validation loss':  np.mean(self.valid_losses)
        }
        acc_dicts = [
            {
                'Training accuracy': acc,
                'Validation accuracy': val_acc
            } for acc, val_acc in zip(accs, valid_accs)
        ]
        iou_dicts = [
            {
                'Training IoU': iou,
                'Validation IoU': val_iou
            } for iou, val_iou in zip(ious, valid_ious)
        ]

        # send results to tensorboard
        writer.add_scalars('Loss', loss_dict, epoch)

        for i in range(self.model.cfg.num_classes):
            writer.add_scalars(f'Per-class accuracy/{i+1:02d}', 
                                                acc_dicts[i], epoch)
            writer.add_scalars(f'Per-class IoU/{i+1:02d}', iou_dicts[i], epoch)
        writer.add_scalars('Per-class accuracy/Overall', acc_dicts[-1], epoch)
        writer.add_scalars('Per-class IoU/Mean IoU', iou_dicts[-1], epoch)

        log_out(f"loss train: {loss_dict['Training loss']:.3f} "
                f" eval: {loss_dict['Validation loss']:.3f}", 
                                    self.log_file)
        log_out(f"acc train: {acc_dicts[-1]['Training accuracy']:.3f} "
                 f" eval: {acc_dicts[-1]['Validation accuracy']:.3f}", 
                                    self.log_file)
        log_out(f"acc train: {iou_dicts[-1]['Training IoU']:.3f} "
                f" eval: {iou_dicts[-1]['Validation IoU']:.3f}", 
                                    self.log_file)
        # print(acc_dicts[-1])

    def load_ckpt(self, ckpt_path, is_train=True):
        if exists(ckpt_path):
            #path = max(list((cfg.ckpt_path).glob('*.pth')))
            log_out(f'Loading checkpoint {ckpt_path}', self.log_file)
            ckpt = torch.load(ckpt_path)
            first_epoch = ckpt['epoch']+1
            self.model.load_state_dict(ckpt['model_state_dict'])
            if is_train:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            first_epoch = 0
            log_out('No checkpoint', self.log_file)

        return first_epoch

    def save_ckpt(self, path_ckpt, epoch):
        make_dir(path_ckpt)
        torch.save(
            dict(
                epoch=epoch,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict()
            ),
            join(path_ckpt, f'ckpt_{epoch:02d}.pth')
        )
        log_out(f'Epoch {epoch:3d}: save ckpt to {path_ckpt:s}', self.log_file)
                    

        
        

    def filter_valid(self, scores, labels, device):
        valid_scores = scores.reshape(-1, self.model.cfg.num_classes)
        valid_labels = labels.reshape(-1).to(device)
                
        ignored_bool = torch.zeros_like(valid_labels, dtype=torch.bool)
        for ign_label in self.dataset.cfg.ignored_label_inds:
            ignored_bool = torch.logical_or(ignored_bool, 
                            torch.eq(valid_labels, ign_label))
           
        valid_idx = torch.where(
            torch.logical_not(ignored_bool))[0].to(device)

        valid_scores = torch.gather(valid_scores, 0, 
            valid_idx.unsqueeze(-1).expand(-1, self.model.cfg.num_classes))
        valid_labels = torch.gather(valid_labels, 0, valid_idx)

        # Reduce label values in the range of logit shape
        reducing_list = torch.arange(0, 
                        self.model.cfg.num_classes, dtype=torch.int64)
        inserted_value = torch.zeros([1], dtype=torch.int64)
        
        for ign_label in self.dataset.cfg.ignored_label_inds:
            reducing_list = torch.cat([reducing_list[:ign_label],
                     inserted_value, reducing_list[ign_label:]], 0)

        valid_labels = torch.gather(reducing_list.to(device), 
                                        0, valid_labels)

        valid_labels = valid_labels.unsqueeze(0)
        valid_scores = valid_scores.unsqueeze(0).transpose(-2,-1)


        return valid_scores, valid_labels
