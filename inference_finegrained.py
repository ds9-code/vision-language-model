import os
import argparse
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, auc
import json

from models.fine_grain_cross_attention_textquery import FineGrainedMRITextCrossAttentionModel as FineGrainedMRITextCrossAttentionModelTextQuery
from data.dataset import MRITextDataset, get_validation_transform, collate_fn
from configs.config import config as global_config

from torchmetrics.classification import AUROC, F1Score, Accuracy as TorchMetricsAccuracy
from sklearn.metrics import classification_report, confusion_matrix



## Main inference Loop 
def run_inference(checkpoint_path, threshold, test_dataset, test_loader, output_dir, calibration_dir, use_gpu):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Loading model from checkpoint: {checkpoint_path}")
    print(f"Using threshold: {threshold:.4f}")
    

    ## spinup the model and checkpoint
    model = FineGrainedMRITextCrossAttentionModelTextQuery()
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint_data:
        state_dict = checkpoint_data['state_dict']
    else:
        state_dict = checkpoint_data
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[len("model."):]] = v
        elif k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=True)   
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    test_auroc = AUROC(task="binary").to(device)
    test_f1 = F1Score(task="binary", average='macro').to(device)
    test_acc = TorchMetricsAccuracy(task="binary").to(device)
    all_preds_probs = []
    all_preds_binary = []
    all_labels = []

    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference", unit="batch")):
            volumes, texts, pad_mask, labels = batch
            volumes = volumes.to(device)
            pad_mask = pad_mask.to(device)
            labels = labels.to(device)
            logits, _ = model(volumes, pad_mask, texts)
            logits = logits.squeeze(1)
            probs = torch.sigmoid(logits)
            preds_binary = (probs > threshold).float()
            test_auroc.update(probs, labels)
            test_f1.update(preds_binary, labels)
            test_acc.update(preds_binary, labels)
            all_preds_probs.extend(probs.cpu().numpy())
            all_preds_binary.extend(preds_binary.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("Inference completed.")

    # estimate metrics 
    final_auroc = test_auroc.compute().item()
    final_f1 = test_f1.compute().item()
    final_acc = test_acc.compute().item()
    all_labels_np = np.array(all_labels).astype(int)
    all_preds_binary_np = np.array(all_preds_binary).astype(int)
    all_probs_np = np.array(all_preds_probs)
    report = classification_report(all_labels_np, all_preds_binary_np, target_names=['Class 0', 'Class 1'], output_dict=True)
    cm = confusion_matrix(all_labels_np, all_preds_binary_np)
    
    precision, recall, _ = precision_recall_curve(all_labels_np, all_probs_np)
    auc_pr = auc(recall, precision)
    
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
        specificity = TN / (TN + FP) if (TN + FP) > 0 else float('nan')
        balanced_accuracy = (sensitivity + specificity) / 2 if not (np.isnan(sensitivity) or np.isnan(specificity)) else float('nan')
    else:
        sensitivity = float('nan')
        specificity = float('nan')
        balanced_accuracy = float('nan')

    prob_true, prob_pred = calibration_curve(all_labels_np, all_probs_np, n_bins=10)
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    calibration_plot_path = os.path.join(calibration_dir, f'calibration_{checkpoint_name}.png')
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(calibration_plot_path)
    plt.close()

    if hasattr(test_dataset, 'df'):
        pat_ids = test_dataset.df['pat_id'].tolist()
        scandates = test_dataset.df['scandates'].tolist()
    else:
        pat_ids = [None] * len(all_labels_np)
        scandates = [None] * len(all_labels_np)
    if len(pat_ids) != len(all_labels_np):
        print("Warning: Number of patient IDs does not match number of predictions. Saving without IDs.")
        pat_ids = [None] * len(all_labels_np)
        scandates = [None] * len(all_labels_np)
    
    preds_csv_path = os.path.join(output_dir, f'predictions_{checkpoint_name}.csv')
    results_df = pd.DataFrame({
        'pat_id': pat_ids,
        'scandates': scandates,
        'label': all_labels_np,
        'prediction_binary': all_preds_binary_np,
        'prediction_prob_class1': all_probs_np
    })
    results_df.to_csv(preds_csv_path, index=False)
    print(f"Predictions saved to {preds_csv_path}")

    results = {
        'checkpoint': checkpoint_path,
        'threshold': threshold,
        'auroc': final_auroc,
        'auc_pr': auc_pr,
        'f1_score': final_f1,
        'accuracy': final_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'balanced_accuracy': balanced_accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'calibration_plot_path': calibration_plot_path,
        'predictions_csv_path': preds_csv_path
    }
    return results

def main():
    parser = argparse.ArgumentParser(description="Inference script for FineGrainedMRITextCrossAttentionModel.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint. Overrides config.inference_checkpoint_path.")
    parser.add_argument("--threshold", type=float, default=None, help="Classification threshold. Overrides config.inference_threshold.")
    parser.add_argument("--test_csv_path", type=str, default=None, help="Path to test CSV. Overrides config.inference_test_csv_path.")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to image files. Overrides config.image_dir.")
    parser.add_argument("--text_dir", type=str, default=None, help="Path to text files. Overrides config.text_dir.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size. Overrides config.batch_size.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers. Overrides config.num_workers.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Overrides config.seed.")
    parser.add_argument("--use_gpu", action='store_true', help="Use GPU if available.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path or global_config.inference_checkpoint_path
    threshold = args.threshold if args.threshold is not None else global_config.inference_threshold
    
    if checkpoint_path is None:
        print("Error: Checkpoint path must be specified in config or via --checkpoint_path")
        sys.exit(1)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)

    output_dir = args.output_dir
    calibration_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(calibration_dir, exist_ok=True)

    # get from config if no ars are passed 
    test_csv_path = args.test_csv_path or global_config.inference_test_csv_path
    image_dir = args.image_dir or global_config.image_dir
    text_dir = args.text_dir or global_config.text_dir
    batch_size = args.batch_size or global_config.batch_size
    num_workers = args.num_workers or global_config.num_workers
    seed = args.seed or global_config.seed
    
    if test_csv_path is None or not os.path.exists(test_csv_path):
        print("Error: Test CSV path is not specified or does not exist.")
        sys.exit(1)
    
    pl.seed_everything(seed, workers=True)
    
    spatial_size = global_config.image_size
    if isinstance(spatial_size, (list, tuple)):
        target_spatial_size = tuple(spatial_size)
    else:
        target_spatial_size = (spatial_size, spatial_size, spatial_size)
    
    # spin up dataloader
    test_transform = get_validation_transform(target_spatial_size)
    test_dataset = MRITextDataset(
        csv_path=test_csv_path,
        image_dir=image_dir,
        text_dir=text_dir,
        spatial_transform=test_transform,
        intensity_transform=None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.use_gpu else False
    )

    print(f"\n=== Running inference ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Test CSV: {test_csv_path}")
    
    # infer
    results = run_inference(
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        test_dataset=test_dataset,
        test_loader=test_loader,
        output_dir=output_dir,
        calibration_dir=calibration_dir,
        use_gpu=args.use_gpu
    )

    results_json_path = "inference_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_json_path}")
    
    print("\n=== Results Summary ===")
    print(f"AUROC: {results['auroc']:.4f}")
    print(f"AUC-PR: {results['auc_pr']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Sensitivity: {results['sensitivity']:.4f}")
    print(f"Specificity: {results['specificity']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")

if __name__ == "__main__":
    main()
