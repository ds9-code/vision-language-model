"""
Script to extract ViT Brainiac features from MRI volumes and save to CSV.
Runs the pretrained ViT backbone on all scans in the validation set.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.causal_vit_vision_model import ViT3D
from data.dataset import get_validation_transform


def get_preprocessing_transform(spatial_size=(96, 96, 96)):
    """
    Get the exact same validation transform used in training.
    Uses get_validation_transform from data.dataset (same as CVAE saliency script).
    """
    return get_validation_transform(spatial_size)


def extract_features(
    csv_path: str,
    output_csv: str,
    vit_checkpoint: str = None,
    batch_size: int = 8,
    device: str = "cuda:0"
):
    """
    Extract ViT features for all scans in the CSV.
    
    Args:
        csv_path: Path to CSV with columns [patid, scandate, dataset, dirpth, pathology, ...]
        output_csv: Path to save output CSV with features
        vit_checkpoint: Path to pretrained ViT checkpoint
        batch_size: Batch size for processing
        device: Device to run on
    """
    print("="*80)
    print("ViT Brainiac Feature Extraction")
    print("="*80)
    print(f"Input CSV: {csv_path}")
    print(f"Output CSV: {output_csv}")
    print(f"ViT checkpoint: {vit_checkpoint}")
    print(f"Device: {device}")
    print("="*80)
    
    # Load CSV - expecting columns: patid, scandate, dataset, dirpth, pathology
    df = pd.read_csv(csv_path, dtype={"patid": str, "scandate": str})
    print(f"\nLoaded {len(df)} scans from CSV")
    
    # CSV already has individual scans, no expansion needed
    scan_df = df.copy()
    print(f"Processing {len(scan_df)} individual scans")
    
    print("\nLoading ViT model...")
    vit_model = ViT3D(
        in_channels=1,
        img_size=(96, 96, 96),
        pretrained=True,
        simclr_ckpt_path=vit_checkpoint
    )
    vit_model = vit_model.to(device)
    vit_model.eval()
    print(f"ViT model loaded. Feature dimension: {vit_model.feature_dim}")
    
    # Preprocessing transform (same as CVAE saliency script)
    transform = get_preprocessing_transform(spatial_size=(96, 96, 96))
    
    # Extract features
    print("\nExtracting features...")
    all_features = []
    all_info = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(scan_df), batch_size), desc="Processing batches"):
            batch_rows = scan_df.iloc[i:i+batch_size]
            batch_volumes = []
            batch_info = []
            
            # Load volumes for this batch
            for _, row in batch_rows.iterrows():
                patid = str(row['patid'])
                date = str(row['scandate'])
                dirpth = row['dirpth']  # Image directory path from CSV
                
                filename = f"{patid}_{date}.nii.gz"
                filepath = os.path.join(dirpth, filename)
                
                if not os.path.exists(filepath):
                    print(f"\nWarning: File not found: {filepath}")
                    continue
                
                try:
                    # Load and preprocess using validation transform (same as CVAE saliency)
                    data = {"volume": filepath}  # Use "volume" key like in training
                    data = transform(data)
                    volume = data["volume"]  # Extract tensor from dict
                    batch_volumes.append(volume)
                    batch_info.append(row.to_dict())
                except Exception as e:
                    import traceback
                    print(f"\nError loading {filepath}:")
                    print(traceback.format_exc())
                    continue
            
            if len(batch_volumes) == 0:
                continue
            
            # Stack and move to device
            volumes_tensor = torch.stack(batch_volumes).to(device)  # [B, C, D, H, W]
            
            # Extract features
            features = vit_model(volumes_tensor)  # [B, 768]
            
            # Move to CPU and convert to numpy
            features_np = features.cpu().numpy()
            
            all_features.append(features_np)
            all_info.extend(batch_info)
    
    # Concatenate all features
    if len(all_features) == 0:
        print("\nError: No features extracted!")
        return
    
    all_features = np.vstack(all_features)  # [N, 768]
    print(f"\nExtracted features shape: {all_features.shape}")
    
    # Create output dataframe
    feature_cols = [f"feat_{i:03d}" for i in range(all_features.shape[1])]
    features_df = pd.DataFrame(all_features, columns=feature_cols)
    
    info_df = pd.DataFrame(all_info)
    output_df = pd.concat([info_df, features_df], axis=1)
    
    # Save to CSV
    output_df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(output_df)} scan features to: {output_csv}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total scans processed: {len(output_df)}")
    print(f"Unique patients: {output_df['patid'].nunique()}")
    print(f"Feature dimensions: {all_features.shape[1]}")
    print(f"\nFeature statistics:")
    print(f"  Mean: {all_features.mean():.4f}")
    print(f"  Std:  {all_features.std():.4f}")
    print(f"  Min:  {all_features.min():.4f}")
    print(f"  Max:  {all_features.max():.4f}")
    print("="*80)
    
    # Print sample rows
    print("\nSample rows from output:")
    if 'dataset' in output_df.columns and 'pathology' in output_df.columns:
        print(output_df[['patid', 'scandate', 'dataset', 'pathology']].head(10))
    else:
        print(output_df[['patid', 'scandate']].head(10))
    
    return output_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract ViT features from MRI volumes")
    parser.add_argument("--csv_path", type=str, required=True, help="Input CSV with scan paths")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV with features")
    parser.add_argument("--vit_checkpoint", type=str, default=None, help="Path to ViT checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    extract_features(
        csv_path=args.csv_path,
        output_csv=args.output_csv,
        vit_checkpoint=args.vit_checkpoint,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()

