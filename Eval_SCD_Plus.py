"""
SCanNet-Plus 评估脚本
用于评估训练好的模型
"""

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage import io
import numpy as np

from datasets import RS_ST as RS
from models.SCanNet_Plus import SCanNetPlus
from utils.utils import accuracy, SCDD_eval_all
from utils.loss import CrossEntropyLoss2d


def evaluate_model(checkpoint_path, data_root=None):
    """
    评估训练好的模型

    Args:
        checkpoint_path: 模型权重路径
        data_root: 数据集根目录（可选）
    """
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    print(f"Loading model from {checkpoint_path}...")
    net = SCanNetPlus(in_channels=3, num_classes=RS.num_classes).to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()
    print("Model loaded successfully.")

    # 加载数据
    print("Loading validation data...")
    val_set = RS.Data('val')
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    criterion = CrossEntropyLoss2d(ignore_index=0)

    # 评估
    print("\n" + "="*60)
    print("Starting evaluation...")
    print("="*60 + "\n")

    start_time = time.time()
    val_loss = 0.0
    acc_meter = []

    preds_all = []
    labels_all = []

    with torch.no_grad():
        for vi, data in enumerate(val_loader):
            imgs_A, imgs_B, labels_A, labels_B, imgs_id = data

            imgs_A = imgs_A.to(device).float()
            imgs_B = imgs_B.to(device).float()
            labels_A = labels_A.to(device).long()
            labels_B = labels_B.to(device).long()

            # 前向传播
            out_change, outputs_A, outputs_B, _, _, _ = net(imgs_A, imgs_B)

            # 计算损失
            loss_A = criterion(outputs_A, labels_A)
            loss_B = criterion(outputs_B, labels_B)
            loss = loss_A * 0.5 + loss_B * 0.5
            val_loss += loss.item()

            # 计算准确率
            labels_A_np = labels_A.cpu().numpy()
            labels_B_np = labels_B.cpu().numpy()
            outputs_A = outputs_A.cpu()
            outputs_B = outputs_B.cpu()
            change_mask = F.sigmoid(out_change).cpu() > 0.5
            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A * change_mask.squeeze().long()).numpy()
            preds_B = (preds_B * change_mask.squeeze().long()).numpy()

            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A_np, labels_B_np):
                acc_A, _ = accuracy(pred_A, label_A)
                acc_B, _ = accuracy(pred_B, label_B)
                acc = (acc_A + acc_B) * 0.5
                acc_meter.append(acc)

                preds_all.append(pred_A)
                preds_all.append(pred_B)
                labels_all.append(label_A)
                labels_all.append(label_B)

            # 打印进度
            if (vi + 1) % 50 == 0:
                print(f"Processed {vi + 1}/{len(val_loader)} images")

    # 计算最终指标
    val_loss = val_loss / len(val_loader)
    accuracy_avg = np.mean(acc_meter)
    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, RS.num_classes)

    elapsed_time = time.time() - start_time

    # 打印结果
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {checkpoint_path}")
    print(f"-"*60)
    print(f"Validation Loss:   {val_loss:.4f}")
    print(f"Overall Accuracy:  {accuracy_avg*100:.2f}%")
    print(f"Mean IoU:          {IoU_mean*100:.2f}%")
    print(f"SCD Score:         {Sek*100:.2f}%")
    print(f"Fscd:              {Fscd*100:.2f}%")
    print(f"-"*60)
    print(f"Evaluation Time:   {elapsed_time:.1f}s")
    print("="*60 + "\n")

    return {
        'val_loss': val_loss,
        'accuracy': accuracy_avg,
        'mIoU': IoU_mean,
        'Sek': Sek,
        'Fscd': Fscd
    }


def compare_models(baseline_path, plus_path):
    """
    比较基线模型和Plus模型的性能

    Args:
        baseline_path: 基线模型路径
        plus_path: Plus模型路径
    """
    print("\n" + "="*60)
    print("SCanNet vs SCanNet-Plus Comparison")
    print("="*60 + "\n")

    print("Evaluating Baseline Model...")
    baseline_results = evaluate_model(baseline_path)

    print("\nEvaluating SCanNet-Plus Model...")
    plus_results = evaluate_model(plus_path)

    # 打印对比
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"{'Metric':<20} {'Baseline':<15} {'SCanNet-Plus':<15} {'Improvement':<15}")
    print("-"*60)
    print(f"{'mIoU (%)':<20} {baseline_results['mIoU']*100:<15.2f} {plus_results['mIoU']*100:<15.2f} {(plus_results['mIoU']-baseline_results['mIoU'])*100:>+14.2f}")
    print(f"{'Sek (%)':<20} {baseline_results['Sek']*100:<15.2f} {plus_results['Sek']*100:<15.2f} {(plus_results['Sek']-baseline_results['Sek'])*100:>+14.2f}")
    print(f"{'Fscd (%)':<20} {baseline_results['Fscd']*100:<15.2f} {plus_results['Fscd']*100:<15.2f} {(plus_results['Fscd']-baseline_results['Fscd'])*100:>+14.2f}")
    print(f"{'OA (%)':<20} {baseline_results['accuracy']*100:<15.2f} {plus_results['accuracy']*100:<15.2f} {(plus_results['accuracy']-baseline_results['accuracy'])*100:>+14.2f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate SCanNet-Plus')
    parser.add_argument('--checkpoint', type=str, default='',
                       help='Path to model checkpoint')
    parser.add_argument('--baseline', type=str, default='',
                       help='Path to baseline model for comparison')
    parser.add_argument('--data_root', type=str, default=None,
                       help='Dataset root directory (optional)')

    args = parser.parse_args()

    if args.checkpoint:
        evaluate_model(args.checkpoint, args.data_root)
    elif args.baseline:
        # 查找最新的Plus模型
        chkpt_dir = 'checkpoints/ST'
        plus_models = [f for f in os.listdir(chkpt_dir) if 'SCanNet_Plus' in f and f.endswith('.pth')]
        if plus_models:
            plus_path = os.path.join(chkpt_dir, sorted(plus_models)[-1])
            compare_models(args.baseline, plus_path)
        else:
            print("No SCanNet-Plus model found in checkpoints directory.")
    else:
        # 自动查找并比较
        chkpt_dir = 'checkpoints/ST'
        baseline_models = [f for f in os.listdir(chkpt_dir) if 'SCanNet_psd' in f and f.endswith('.pth')]
        plus_models = [f for f in os.listdir(chkpt_dir) if 'SCanNet_Plus' in f and f.endswith('.pth')]

        if baseline_models and plus_models:
            baseline_path = os.path.join(chkpt_dir, sorted(baseline_models)[-1])
            plus_path = os.path.join(chkpt_dir, sorted(plus_models)[-1])
            compare_models(baseline_path, plus_path)
        else:
            print("Please specify --checkpoint or --baseline argument")
