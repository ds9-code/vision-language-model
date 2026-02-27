from dataclasses import dataclass
from typing import Tuple, Union, Optional


@dataclass
class ModelConfig:
    # Vision
    vision_encoder: str = "vitb"
    image_size: Union[int, Tuple[int, int, int]] = (96, 96, 96)
    in_channels: int = 1
    resnet_type: str = "vitb"
    resnet_pretrained: bool = True
    resnet_dropout: float = 0.2

    # Temporal transformer
    temp_depth: int = 3
    temp_heads: int = 8
    temp_mlp_dim: int = 2048
    max_seq_len: int = 6

    # Cross-attention transformer
    cross_attn_depth: int = 2
    cross_attn_heads: int = 8
    cross_attn_mlp_dim: int = 2048
    cross_attn_dropout: float = 0.1

    # Text
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    text_max_length: int = 128

    # Embedding
    embedding_dim: int = 512

    # CVAE Configuration
    cvae_latent_dim: int = 512
    cvae_enc_heads: int = 4
    cvae_enc_layers: int = 2
    cvae_dec_heads: int = 4
    cvae_dec_layers: int = 2
    cvae_dropout: float = 0.1
    cvae_recon_weight: float = 0.4
    cvae_cls_weight: float = 1.0
    cvae_kl_weight: float = 0.1
    cvae_kl_weight_min: float = 0.01
    cvae_kl_warmup_epochs: int = 50
    cvae_recon_weight_schedule: bool = True
    cvae_recon_weight_min: float = 0.6
    cvae_recon_weight_max: float = 0.8
    cvae_recon_weight_final: float = 0.6
    cvae_recon_warmup_epochs: int = 20
    cvae_recon_peak_epochs: int = 50

    # Training
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    max_epochs: int = 100
    warmup_steps: int = 1000
    temperature: float = 0.07
    gpus: int = 1
    num_workers: int = 4
    seed: Optional[int] = 42
    backbone_learning_rate: float = 5e-5

    # Pathsto the datasets
    train_csv: str = "/media/sdb/divyanshu/divyanshu/tlv2/csvs/vlm_efs__trainval_2_withouterrorimages_train_oversampled_resectionstatus_with_timedeltas.csv"
    val_csv: str = "/media/sdb/divyanshu/divyanshu/tlv2/csvs/vlm_efs__trainval_2_withouterrorimages_val_oversampled_balanced_resectionstatus_with_timedeltas.csv"
    image_dir: str = "/media/sdb/divyanshu/divyanshu/tlv2/data/preprocessed_flair/nnunet/imagesTs_renamed"
    text_dir: str = "/media/sdb/divyanshu/divyanshu/tlv2/data/cross_attention_text/trainval_resection_ageatdiag"
    checkpoint_dir: str = "/media/sdb/divyanshu/divyanshu/tlv2/checkpoints_v2/crossattention_text"

    # Inference specific paths
    inference_checkpoint_path: Optional[str] = "./checkpoints/textquery_crossattention_resectionstatus_brainiacvit_FROZEN_noselfattn_meanpoolingtokenvit_clsqueriesimg_lr00005_epoch=21-val_f1=0.8584-val_auc=0.9263.ckpt"
    inference_test_csv_path: Optional[str] = "/media/sdb/divyanshu/divyanshu/tlv2/csvs/vlm_efs__test_2_withouterrorimages_realistic_undersampled_resectionstatus copy_with_timedeltas.csv"
    inference_threshold: float = 0.5
    testing_cohort: str = ""

    # Model type
    model_type: str = "cross_attention_textquery"

    # Fine-tuning configuration
    finetune_from_checkpoint: bool = False
    finetune_checkpoint_path: Optional[str] = None

    @property
    def data(self) -> dict:
        return {"size": self.image_size}

    # Logging
    log_every_n_steps: int = 10
    use_wandb: bool = True
    project_name: str = "VLM_EFS"
    run_name: str = ""


config = ModelConfig()
