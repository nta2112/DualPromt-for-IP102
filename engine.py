# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer

import utils
import metrics
import pandas as pd

def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        
        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    if original_model is not None:
        original_model.eval()

    all_features = []
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']
            features = output.get('pre_logits', None)

            if features is not None:
                all_features.append(features.cpu())
            all_targets.append(target.cpu())
            all_logits.append(logits.cpu())

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    if len(all_features) > 0:
        all_features = torch.cat(all_features, dim=0).numpy()
    else:
        all_features = None
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_logits = torch.cat(all_logits, dim=0).numpy()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, all_features, all_targets, all_logits


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    all_features_list = []
    all_targets_list = []
    all_logits_list = []

    for i in range(task_id+1):
        test_stats, features, targets, logits = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
        
        if features is not None:
            all_features_list.append(features)
        all_targets_list.append(targets)
        all_logits_list.append(logits)
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)
    
    # Compute new metrics
    if len(all_features_list) > 0:
        full_features = np.concatenate(all_features_list, axis=0)
        full_features = full_features / (np.linalg.norm(full_features, axis=1, keepdims=True) + 1e-8)
        dist_matrix = full_features @ full_features.T
    else:
        dist_matrix = None
        
    full_targets = np.concatenate(all_targets_list, axis=0)
    full_logits = np.concatenate(all_logits_list, axis=0)
    
    seen_classes = []
    for i in range(task_id + 1):
        if class_mask is not None:
            seen_classes.extend(class_mask[i])
    seen_classes_set = set(seen_classes)
    
    if dist_matrix is not None:
        mAP = metrics.calculate_mAP(dist_matrix, full_targets)
        recalls = metrics.calculate_recall_at_k(dist_matrix, full_targets, [1, 5, 10])
        r1, r5, r10 = recalls[1], recalls[5], recalls[10]
        r1_seen, r1_unseen = metrics.compute_open_world_recall(dist_matrix, full_targets, seen_classes_set)
    else:
        mAP, r1, r5, r10, r1_seen, r1_unseen = 0, 0, 0, 0, 0, 0
        
    auroc, fpr95 = metrics.compute_ood_metrics(full_logits, full_targets, list(seen_classes_set))
    
    # Track lifelong metrics based on mAP
    if not hasattr(args, 'map_matrix'):
        args.map_matrix = np.zeros((args.num_tasks, args.num_tasks))
        
    for i in range(task_id + 1):
        t_targets = all_targets_list[i]
        if dist_matrix is not None:
            t_features = all_features_list[i]
            t_features = t_features / (np.linalg.norm(t_features, axis=1, keepdims=True) + 1e-8)
            t_dist = t_features @ t_features.T
            t_mAP = metrics.calculate_mAP(t_dist, t_targets)
            args.map_matrix[i, task_id] = t_mAP
        
    plasticity = args.map_matrix[task_id, task_id] if task_id >= 0 else 0
    forgetting_map = 0.0
    if task_id > 0:
        forgetting_map = np.mean((np.max(args.map_matrix, axis=1) - args.map_matrix[:, task_id])[:task_id])
    overall_map = np.mean(args.map_matrix[:task_id+1, task_id])
    
    # Save to CSV
    csv_file = os.path.join(args.output_dir if args.output_dir else '.', 'results.csv')
    write_header = not os.path.exists(csv_file)
    with open(csv_file, 'a') as f:
        if write_header:
            f.write("task,numclass,cnn_top1,nme_top1,R@1,R@5,R@10,mAP,AUROC,FPR95,Plasticity,Forgetting,Overall\n")
        f.write(f"{task_id},{len(seen_classes_set)},{avg_stat[0]:.4f},0.0,{r1:.4f},{r5:.4f},{r10:.4f},{mAP:.4f},{auroc if auroc is not None else 'None'},{fpr95 if fpr95 is not None else 'None'},{plasticity:.4f},{forgetting_map:.4f},{overall_map:.4f}\n")
        
    # Save to json
    json_file = os.path.join(args.output_dir if args.output_dir else '.', 'history.json')
    history = {}
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            history = json.load(f)
    history[str(task_id)] = {
        'R@1': r1, 'R@5': r5, 'R@10': r10, 'mAP': mAP,
        'AUROC': auroc, 'FPR95': fpr95,
        'Plasticity': plasticity, 'Forgetting': forgetting_map, 'Overall': overall_map
    }
    with open(json_file, 'w') as f:
        json.dump(history, f, indent=4)

    return test_stats

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):
        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        unwrapped_model = utils.unwrap_model(model)
                        unwrapped_model.e_prompt.prompt.grad.zero_()
                        unwrapped_model.e_prompt.prompt[cur_idx] = unwrapped_model.e_prompt.prompt[prev_idx]
                        optimizer.param_groups[0]['params'] = unwrapped_model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    unwrapped_model = utils.unwrap_model(model)
                    unwrapped_model.e_prompt.prompt_key.grad.zero_()
                    unwrapped_model.e_prompt.prompt_key[cur_idx] = unwrapped_model.e_prompt.prompt_key[prev_idx]
                    optimizer.param_groups[0]['params'] = unwrapped_model.parameters()
     
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        for epoch in range(args.epochs):            
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)
            
            if lr_scheduler:
                lr_scheduler.step(epoch)

        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')