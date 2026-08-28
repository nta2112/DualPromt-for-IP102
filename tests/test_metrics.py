import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics

def test_metrics_perfect():
    # Perfect case: features perfectly cluster
    # 5 samples, classes [0, 0, 1, 1, 2]
    # In perfect clustering, dist between same class is 1.0, diff class is 0.0
    labels = np.array([0, 0, 1, 1, 2])
    dist_matrix = np.array([
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0] # class 2 has only 1 sample, recall might be 0, mAP 0
    ])
    
    mAP = metrics.calculate_mAP(dist_matrix, labels)
    # class 0 has 1 true neighbor. dist to it is 1.0. AP=1.0
    # class 1 has 1 true neighbor. dist to it is 1.0. AP=1.0
    # class 2 has 0 true neighbors. It's skipped.
    # Mean AP of [1.0, 1.0, 1.0, 1.0] = 1.0
    assert mAP == 1.0, f"mAP should be 1.0, got {mAP}"
    
    recalls = metrics.calculate_recall_at_k(dist_matrix, labels, [1, 5, 10])
    assert recalls[1] == 1.0, f"R@1 should be 1.0, got {recalls[1]}"
    
    # Perfect OOD: Seen classes [0, 1], Unseen [2]
    seen_classes = [0, 1]
    # Logits: high for seen classes if it's seen, low if it's unseen
    logits = np.array([
        [10.0, 0.0, 0.0], # class 0 (seen), max seen logit=10
        [10.0, 0.0, 0.0], # class 0
        [0.0, 10.0, 0.0], # class 1 (seen), max seen logit=10
        [0.0, 10.0, 0.0], # class 1
        [0.0, 0.0, 10.0]  # class 2 (unseen), max seen logit=0
    ])
    auroc, fpr95 = metrics.compute_ood_metrics(logits, labels, seen_classes)
    assert auroc == 1.0, f"AUROC should be 1.0, got {auroc}"
    assert fpr95 == 0.0, f"FPR95 should be 0.0, got {fpr95}"
    
    print("All metric tests passed!")

if __name__ == '__main__':
    test_metrics_perfect()
