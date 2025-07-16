import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import sys
# Import your custom modules
from model import HybridBuildingDetector
from dataset import SpaceNetBuildingDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def train(args):
    logger.info(f"Starting training on {args.device}...")
    logger.info(f"Hyperparameters: {args}")

    device = torch.device(args.device)

    # Data paths from SageMaker environment variables
    train_image_dir = os.environ.get('SM_CHANNEL_TRAIN_IMAGES')
    train_mask_dir = os.environ.get('SM_CHANNEL_TRAIN_MASKS')
    val_image_dir = os.environ.get('SM_CHANNEL_VAL_IMAGES')
    val_mask_dir = os.environ.get('SM_CHANNEL_VAL_MASKS')

    # Data preparation
    num_input_channels = args.num_input_channels

    train_dataset = SpaceNetBuildingDataset(
        image_root_dir=train_image_dir,
        mask_root_dir=train_mask_dir,
        num_input_channels=num_input_channels
    )
    val_dataset = SpaceNetBuildingDataset(
        image_root_dir=val_image_dir,
        mask_root_dir=val_mask_dir,
        num_input_channels=num_input_channels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Model, Loss, Optimizer
    model = HybridBuildingDetector(
        cnn_model=args.cnn_model,
        num_input_channels=num_input_channels,
        vit_embed_dim=args.vit_embed_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        gat_hidden_channels=args.gat_hidden_channels,
        gat_heads=args.gat_heads,
        num_classes=1
    ).to(device)

    criterion = nn.BCEWithLogitsLoss() # For binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                logger.info(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.6f}")
        
        avg_train_loss = running_loss / len(train_loader)
        logger.info(f'Epoch {epoch} Training Loss: {avg_train_loss:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item() * data.size(0)

        val_loss /= len(val_loader.dataset)
        logger.info(f'Validation set: Average loss: {val_loss:.4f}')
        
        # Print metrics for SageMaker CloudWatch parsing
        print(f'sagemaker_metric: Train_Loss={avg_train_loss:.6f}')
        print(f'sagemaker_metric: Validation_Loss={val_loss:.6f}')

    # Save the trained model
    logger.info(f"Saving model to {args.model_dir}/model.pth")
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.cpu().state_dict(), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Define all your argparse arguments here, as shown in previous responses.
    # Include SM_CHANNEL_* and SM_MODEL_DIR arguments.
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cnn-model', type=str, default='resnet50')
    parser.add_argument('--num-input-channels', type=int, default=3)
    parser.add_argument('--vit-embed-dim', type=int, default=768)
    parser.add_argument('--vit-depth', type=int, default=6)
    parser.add_argument('--vit-heads', type=int, default=12)
    parser.add_argument('--gat-hidden-channels', type=int, default=256)
    parser.add_argument('--gat-heads', type=int, default=4)

    # SageMaker specific arguments
    parser.add_argument('--train-images-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_IMAGES'))
    parser.add_argument('--train-masks-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_MASKS'))
    parser.add_argument('--val-images-dir', type=str, default=os.environ.get('SM_CHANNEL_VAL_IMAGES'))
    parser.add_argument('--val-masks-dir', type=str, default=os.environ.get('SM_CHANNEL_VAL_MASKS'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    # Default device handling for SageMaker
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train(args)