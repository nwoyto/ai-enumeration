# config.py
# Centralized configuration for model architecture and training hyperparameters

MODEL_CONFIG = {
    "num_input_channels": 8,
    "cnn_model": "resnet50",
    "pretrained_cnn": True,
    "vit_embed_dim": 768,
    "vit_depth": 4,         # If you implement depth stacking
    "vit_heads": 12,
    "gat_heads": 4,
    "num_classes": 1,
    "dropout": 0.2,
}

TRAINING_CONFIG = {
    "batch_size": 16,
    "epochs": 40,
    "learning_rate": 1e-4,
    "gradient_clip_val": 1.0,
    "log_interval": 10,
    "lr_scheduler_patience": 5,
    "num_workers": 4,
    "seed": 42,
    # Add more as needed
}

# Example for experiment tracking
EXPERIMENT_CONFIG = {
    "experiment_name": "hybrid_vitgat_overfit_test",
    "save_dir": "outputs/exp1/",
}
