import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

def calculate_mAP(dist_matrix, labels):
    n = len(labels)
    aps = []
    
    for i in range(n):
        y_true = (labels == labels[i]).astype(int)
        y_true[i] = 0 # exclude self
        if y_true.sum() == 0:
            continue
        
        y_score = dist_matrix[i].copy()
        y_score[i] = -1e9 # exclude self
        
        ap = average_precision_score(y_true, y_score)
        aps.append(ap)
        
    return np.mean(aps) if aps else 0.0

def calculate_recall_at_k(dist_matrix, labels, k_list=[1, 5, 10]):
    n = len(labels)
    recalls = {k: 0.0 for k in k_list}
    valid_queries = 0
    
    sorted_indices = np.argsort(-dist_matrix, axis=1)
    
    for i in range(n):
        y_true = (labels == labels[i]).astype(int)
        y_true[i] = 0
        if y_true.sum() == 0:
            continue
            
        valid_queries += 1
        ranks = sorted_indices[i]
        ranks = ranks[ranks != i]
        
        for k in k_list:
            top_k_indices = ranks[:k]
            if np.sum(y_true[top_k_indices]) > 0:
                recalls[k] += 1
                
    if valid_queries == 0:
        return {k: 0.0 for k in k_list}
        
    return {k: v / valid_queries for k, v in recalls.items()}

def compute_fpr95(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        return fpr[idx[0]]
    return None

def compute_ood_metrics(logits, targets, seen_classes):
    # OOD metrics based on Maximum Softmax Probability or Max Logit among seen classes
    # y_true: 1 for Seen (In-Distribution), 0 for Unseen (Out-of-Distribution)
    y_true = np.array([1 if t in seen_classes else 0 for t in targets])
    
    if len(np.unique(y_true)) < 2:
        return None, None # Cannot compute AUROC if only one class exists
        
    # score: max logit among seen classes
    seen_logits = logits[:, seen_classes]
    y_score = np.max(seen_logits, axis=1)
    
    auroc = roc_auc_score(y_true, y_score)
    fpr95 = compute_fpr95(y_true, y_score)
    return auroc, fpr95

def compute_open_world_recall(dist_matrix, labels, seen_classes):
    n = len(labels)
    valid_seen = 0
    valid_unseen = 0
    recall_seen = 0.0
    recall_unseen = 0.0
    
    sorted_indices = np.argsort(-dist_matrix, axis=1)
    
    for i in range(n):
        y_true = (labels == labels[i]).astype(int)
        y_true[i] = 0
        if y_true.sum() == 0:
            continue
            
        ranks = sorted_indices[i]
        ranks = ranks[ranks != i]
        top1_idx = ranks[0]
        
        is_correct = (labels[top1_idx] == labels[i])
        
        if labels[i] in seen_classes:
            valid_seen += 1
            if is_correct:
                recall_seen += 1
        else:
            valid_unseen += 1
            if is_correct:
                recall_unseen += 1
                
    r1_seen = recall_seen / valid_seen if valid_seen > 0 else 0.0
    r1_unseen = recall_unseen / valid_unseen if valid_unseen > 0 else None
    
    return r1_seen, r1_unseen
