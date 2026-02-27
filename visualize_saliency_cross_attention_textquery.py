import os
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from monai.visualize.gradient_based import VanillaGrad, SmoothGrad, GuidedBackpropSmoothGrad

from configs.config import config
from data.dataset import get_validation_transform
from models.fine_grain_cross_attention_textquery import FineGrainedMRITextCrossAttentionModel


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
class CrossAttentionScalarWrapper(nn.Module):
    """Wrap cross-attention model to produce scalar (L2 norm of pooled embedding)."""

    def __init__(self, model: nn.Module, text_query: str):
        super().__init__()
        self.model = model
        self.text_query = text_query

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape [B, C, D, H, W] or [B, T, C, D, H, W]
        Returns:
            Scalar tensor [B, 1] representing L2 norm of fused embedding.
        """
        if x.dim() == 5:  # [B, C, D, H, W] -> add temporal dimension
            x = x.unsqueeze(1)
        if x.dim() != 6:
            raise ValueError(f"Expected 5D or 6D tensor, got shape {tuple(x.shape)}")

        B, T, C, D, H, W = x.shape
        pad_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        texts = [self.text_query] * B

        _, pooled = self.model(x, pad_mask, texts)
        return torch.norm(pooled, dim=1, keepdim=True)  # [B, 1]


def clean_and_load_state_dict(model: nn.Module, checkpoint_path: str, device: str):
    """Load checkpoint, stripping common prefixes (model./module.)."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    raw_state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in raw_state.items():
        if k.startswith("model."):
            cleaned[k[len("model.") :]] = v
        elif k.startswith("module."):
            cleaned[k[len("module.") :]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)} (expected when loading partial ckpt)")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    print("Checkpoint loaded successfully!")


def build_preprocess(size_3d):
    """Build preprocessing pipeline using validation transforms."""
    if isinstance(size_3d, int):
        target_size = (size_3d, size_3d, size_3d)
    else:
        target_size = tuple(size_3d)
    return get_validation_transform(target_size)


def resolve_text_query(text_file_path, text_query):
    if text_file_path:
        if not os.path.exists(text_file_path):
            raise FileNotFoundError(f"Text file does not exist: {text_file_path}")
        text = Path(text_file_path).read_text().strip()
        if not text:
            raise ValueError(f"Text file {text_file_path} is empty.")
        return text
    if not text_query or not text_query.strip():
        raise ValueError("Text query is empty and no text file provided.")
    return text_query.strip()


def load_sequence_tensor(preprocess, device: str, image_paths):
    """Load, preprocess, and stack a sequence of volumes."""
    tensors = []
    for idx, path in enumerate(image_paths):
        data = preprocess({"volume": path})
        tensor = data["volume"]  # [C, D, H, W]
        tensors.append(tensor)
        print(
            f"  [{idx}] {path} -> shape {tuple(tensor.shape)}, "
            f"range [{tensor.min().item():.3f}, {tensor.max().item():.3f}]"
        )

    seq = torch.stack(tensors, dim=0)  # [T, C, D, H, W]
    seq = seq.unsqueeze(0).to(device)  # [1, T, C, D, H, W]
    seq.requires_grad_(True)
    return seq


def generate_saliency(image_paths, text_query, checkpoint_path, output_dir, methods):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if isinstance(image_paths, str):
        image_paths = [image_paths]
    image_paths = [str(p) for p in image_paths]

    if not image_paths:
        raise ValueError("No image paths provided")
    
    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Image paths do not exist: {missing}")

    text_query = resolve_text_query(None, text_query)
    print(f"Using text query ({len(text_query.split())} words)")

    if checkpoint_path and not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    os.makedirs(output_dir, exist_ok=True)

    model = FineGrainedMRITextCrossAttentionModel().to(device).eval()

    if checkpoint_path:
        clean_and_load_state_dict(model, checkpoint_path, device)

    wrapped = CrossAttentionScalarWrapper(model, text_query).to(device).eval()

    preprocess = build_preprocess(config.image_size)
    img_tensor = load_sequence_tensor(preprocess, device, image_paths)

    visualizers = {
        "vanilla": VanillaGrad(wrapped),
        "smooth": SmoothGrad(wrapped, stdev_spread=0.15, n_samples=25, magnitude=True),
        "guided_backprop": GuidedBackpropSmoothGrad(wrapped, stdev_spread=0.15, n_samples=25, magnitude=True),
    }
    pre_np = img_tensor.squeeze(0).detach().cpu().numpy()
    for idx, path in enumerate(image_paths):
        vol = pre_np[idx][0] if pre_np[idx].shape[0] == 1 else pre_np[idx][0]
        image_name = Path(path).stem.replace(".nii", "")
        input_nii = nib.Nifti1Image(vol, affine=np.eye(4))
        input_path = os.path.join(output_dir, f"{image_name}_preprocessed_input.nii.gz")
        nib.save(input_nii, input_path)

    for method in methods:
        if method not in visualizers:
            continue
        try:
            with torch.enable_grad():
                sal = visualizers[method](img_tensor)
            sal_np = sal.detach().cpu().numpy().squeeze(0)
            for idx, path in enumerate(image_paths):
                vol = sal_np[idx][0] if sal_np[idx].shape[0] == 1 else sal_np[idx][0]
                image_name = Path(path).stem.replace(".nii", "")
                out_nii = nib.Nifti1Image(vol, affine=np.eye(4))
                out_path = os.path.join(output_dir, f"{image_name}_saliency_{method}.nii.gz")
                nib.save(out_nii, out_path)
        except Exception as e:
            print(f"Error generating {method}: {e}")

    print(f"Saliency maps saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate saliency maps for cross-attention model")
    parser.add_argument("--image_paths", type=str, nargs="+", required=True, help="Paths to input NIfTI images")
    parser.add_argument("--text_query", type=str, required=True, help="Text query for the model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./saliency_output", help="Output directory")
    parser.add_argument("--methods", type=str, nargs="+", default=["vanilla", "smooth", "guided_backprop"],
                        help="Saliency methods to use")
    args = parser.parse_args()
    
    generate_saliency(args.image_paths, args.text_query, args.checkpoint_path, args.output_dir, args.methods)


