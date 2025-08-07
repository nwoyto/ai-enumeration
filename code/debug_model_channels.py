import torch
from model import HybridGeoNet
from config import MODEL_CONFIG

# ---- CONFIGURE THESE TO MATCH YOUR TRAINING JOB ----
BATCH_SIZE = 16
NUM_INPUT_CHANNELS = 8
IMG_SIZE = 224  # or whatever your model expects (e.g., 224, 256, etc.)
CNN_MODEL = 'resnet50'
PRETRAINED_CNN = True
VIT_EMBED_DIM = 768
VIT_DEPTH = 4
VIT_HEADS = 12
GAT_HEADS = 4

# ---- Instantiate the model ----
model = HybridGeoNet(**MODEL_CONFIG)

model.eval()

# ---- Create dummy input ----
dummy_images = torch.randn(BATCH_SIZE, NUM_INPUT_CHANNELS, IMG_SIZE, IMG_SIZE)

# ---- Forward pass with shape debugging ----
def debug_forward(model, x):
    print(f"Input shape: {x.shape}")
    try:
        # Optionally, you can add hooks or modify model code to print more shapes
        out = model(x)
        print(f"Output shape: {out.shape}")
    except Exception as e:
        print(f"Exception during forward pass: {e}")
        raise

if __name__ == "__main__":
    debug_forward(model, dummy_images)
