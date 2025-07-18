import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler # For Mixed Precision Training
import logging
import sys
import numpy as np # For random seed and image visualization
import matplotlib.pyplot as plt
import pandas as pd # For smoothing curves
from datetime import datetime # For timestamp in summary report

# Import your custom modules
from model import HybridGeoNet # Assuming model.py defines HybridGeoNet

# --- Placeholder for SpaceNetBuildingDataset ---
# YOU MUST REPLACE THIS WITH YOUR ACTUAL SpaceNetBuildingDataset IMPLEMENTATION.
# This placeholder generates dummy data for the script to run without errors.
class SpaceNetBuildingDataset(torch.utils.data.Dataset):
    def __init__(self, image_root_dir, mask_root_dir, num_input_channels, img_size=(224, 224)):
        self.image_root_dir = image_root_dir
        self.mask_root_dir = mask_root_dir
        self.num_input_channels = num_input_channels
        self.img_size = img_size
        
        # In a real scenario, you'd list and load actual image and mask files.
        # Example: self.image_files = sorted([f for f in os.listdir(image_root_dir) if f.endswith('.tif')])
        # For this placeholder, we'll just simulate a dataset size.
        self.num_samples = 1000 # Example number of samples
        logger.info(f"Initialized dummy SpaceNetBuildingDataset with {self.num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate dummy data for demonstration
        image = torch.rand(self.num_input_channels, self.img_size[0], self.img_size[1]).float() * 0.8 + 0.1 # Values between 0.1 and 0.9
        mask = torch.randint(0, 2, (1, self.img_size[0], self.img_size[1])).float() # Binary mask (0 or 1)

        # IMPORTANT: In your actual implementation, ensure:
        # 1. Images are loaded as float tensors (e.g., torch.float32)
        # 2. Masks are binary (0 or 1) float tensors (e.g., torch.float32)
        # 3. Proper image normalization (e.g., to [0,1] or [-1,1]) if not already handled by input data
        return image, mask
# --- END Placeholder ---


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# --- Loss Functions ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Apply sigmoid to logits for Dice Loss
        pred = torch.sigmoid(pred)

        # Flatten label and prediction tensors
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

# --- Metrics ---
def calculate_metrics(preds, targets, smooth=1e-6):
    # preds are logits, convert to probabilities and then binary
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float() # Thresholding at 0.5

    intersection = (preds * targets).sum(dim=[1, 2, 3]) # Sum over C, H, W
    union = (preds + targets).sum(dim=[1, 2, 3]) - intersection

    # Avoid division by zero for empty masks
    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (preds.sum(dim=[1, 2, 3]) + targets.sum(dim=[1, 2, 3]) + smooth)

    return iou.mean(), dice.mean() # Return mean over batch

# --- Plotting Functions ---
def plot_curves(train_losses, val_losses, val_ious, val_dices, epoch, best_val_iou, best_iou_epoch_num, output_img_dir):
    epochs_range = range(len(train_losses))

    # Smoothing (optional, but helpful for noisy curves)
    if len(train_losses) > 5: # Only smooth if enough data
        smoothed_train_losses = pd.Series(train_losses).rolling(window=3, min_periods=1).mean()
        smoothed_val_losses = pd.Series(val_losses).rolling(window=3, min_periods=1).mean()
        smoothed_val_ious = pd.Series(val_ious).rolling(window=3, min_periods=1).mean()
        smoothed_val_dices = pd.Series(val_dices).rolling(window=3, min_periods=1).mean()
    else: # If not enough data points, no smoothing
        smoothed_train_losses = train_losses
        smoothed_val_losses = val_losses
        smoothed_val_ious = val_ious
        smoothed_val_dices = val_dices

    # Plot Loss Curve
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, smoothed_train_losses, label='Train Loss', color='blue', linestyle='-')
    plt.plot(epochs_range, smoothed_val_losses, label='Validation Loss', color='red', linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training & Validation Loss Over Epochs (Epoch {epoch + 1})', fontsize=14) # epoch + 1 for display
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, f'loss_curve_epoch{epoch}.png'))
    plt.close()

    # Plot Metric Curves (IoU & Dice)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, smoothed_val_ious, label='Validation IoU', color='green', linestyle='-')
    plt.plot(epochs_range, smoothed_val_dices, label='Validation Dice', color='purple', linestyle='--')
    
    # Highlight the best IoU point
    if best_val_iou > -1.0: # If a best IoU has been recorded
        plt.scatter(best_iou_epoch_num, best_val_iou, color='darkgreen', marker='*', s=200, zorder=5, label=f'Best IoU: {best_val_iou:.4f}')
        plt.annotate(f'{best_val_iou:.4f}', (best_iou_epoch_num, best_val_iou),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='darkgreen')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title(f'Validation Metrics (IoU & Dice) Over Epochs (Epoch {epoch + 1})', fontsize=14) # epoch + 1 for display
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.ylim(0, 1.05) # Metrics are usually between 0 and 1
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, f'metrics_curve_epoch{epoch}.png'))
    plt.close()

# --- Summary Report Generation ---
def generate_summary_report(args, best_val_iou, best_iou_epoch_num, train_losses, val_losses, val_dices, output_img_dir):
    report_path = os.path.join(args.output_data_dir, 'training_summary_report.md')
    with open(report_path, 'w') as f:
        f.write(f"# Training Summary Report for Building Detector 🏗️\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model:** HybridGeoNet\n")
        f.write(f"**Dataset:** SpaceNet Las Vegas Building Detection (Sampled)\n\n") # Adjusted for placeholder dataset
        
        f.write("## Key Performance Metrics\n")
        f.write(f"- **Best Validation IoU**: {best_val_iou:.4f} (achieved at Epoch {best_iou_epoch_num})\n")
        
        # Ensure lists are not empty before accessing last element
        final_val_loss = val_losses[-1] if val_losses else "N/A"
        final_val_dice = val_dices[-1] if val_dices else "N/A"

        f.write(f"- **Final Validation Loss**: {final_val_loss:.4f}\n")
        f.write(f"- **Final Validation Dice**: {final_val_dice:.4f}\n\n")
        
        f.write("## Hyperparameters Used\n")
        for arg, value in vars(args).items():
            f.write(f"- `{arg}`: `{value}`\n")
        f.write("\n")

        f.write("## Visualizations\n")
        f.write("Below are links to key plots and example predictions. All images are saved in the `images/` subdirectory of the output data path.\n\n")
        
        # Link to the final loss/metric curves
        f.write(f"### 📈 Performance Curves\n")
        f.write(f"![Loss Curve](images/loss_curve_epoch{args.epochs-1}.png)\n")
        f.write(f"![Metrics Curve](images/metrics_curve_epoch{args.epochs-1}.png)\n\n")

        # Link to sample predictions (e.g., from the best epoch or final epoch)
        f.write(f"### 👁️‍🗨️ Example Building Detections (Last Epoch: {args.epochs})\n")
        for i in range(min(args.num_vis_samples, 3)): # Show a few examples
            f.write(f"#### Sample {i+1}\n")
            f.write(f"![Sample {i+1} Prediction Overlay](images/val_pred_overlay_epoch{args.epochs-1}_sample{i}.png)\n\n")
        
        f.write("\n")
        f.write("---")
    logger.info(f"Summary report generated at {report_path}")


def train(args):
    logger.info(f"Starting training on {args.device}...")
    logger.info(f"Hyperparameters: {args}")

    device = torch.device(args.device)

    # Data paths from SageMaker environment variables or default args
    train_image_dir = args.train_images_dir
    train_mask_dir = args.train_masks_dir
    val_image_dir = args.val_images_dir
    val_mask_dir = args.val_masks_dir

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
    model = HybridGeoNet( # Using HybridGeoNet
        cnn_model=args.cnn_model,
        num_input_channels=num_input_channels,
        vit_embed_dim=args.vit_embed_dim,
        # vit_depth is currently not used directly in HybridGeoNet as only one HybridViTGATBlock is defined.
        # If you stack multiple, you would use this arg.
        vit_depth=args.vit_depth, 
        vit_heads=args.vit_heads,
        gat_heads=args.gat_heads,
        num_classes=1, # For binary building detection
        pretrained_cnn=args.pretrained_cnn
    ).to(device)

    # Combined Loss Function
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate) # AdamW is often better for Transformers

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_scheduler_patience, verbose=True, min_lr=1e-6)
    # Using 'max' mode for scheduler to track increasing IoU/Dice

    # GradScaler for Mixed Precision Training
    scaler = GradScaler()

    # Training loop variables
    train_losses = []
    val_losses = []
    val_ious = []
    val_dices = []
    best_val_iou = -1.0 # Initialize with a low value for tracking best model
    best_iou_epoch_num = -1 # To store epoch number of best IoU
    
    output_img_dir = os.path.join(args.output_data_dir, "images")
    os.makedirs(output_img_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            
            with autocast(): # Enable mixed precision
                output = model(data)
                loss_bce = criterion_bce(output, target)
                loss_dice = criterion_dice(output, target)
                loss = loss_bce + loss_dice # Simple sum, can be weighted (e.g., 0.5*bce + 0.5*dice)
            
            scaler.scale(loss).backward() # Scale loss and call backward
            
            # Gradient Clipping
            scaler.unscale_(optimizer) # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_val)
            
            scaler.step(optimizer)        # Update optimizer
            scaler.update()               # Update scaler for next iteration
            running_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                logger.info(f"Train Epoch: {epoch} [{batch_idx * args.batch_size}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.6f}")
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logger.info(f'Epoch {epoch} Training Loss: {avg_train_loss:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0
        val_total_iou = 0
        val_total_dice = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                with autocast(): # Also use mixed precision for validation
                    output = model(data)
                    loss_bce = criterion_bce(output, target)
                    loss_dice = criterion_dice(output, target)
                    current_loss = loss_bce + loss_dice
                
                val_loss += current_loss.item() * data.size(0)

                # Calculate metrics
                batch_iou, batch_dice = calculate_metrics(output, target)
                val_total_iou += batch_iou.item() * data.size(0)
                val_total_dice += batch_dice.item() * data.size(0)

        val_loss /= len(val_loader.dataset)
        avg_val_iou = val_total_iou / len(val_loader.dataset)
        avg_val_dice = val_total_dice / len(val_loader.dataset)

        val_losses.append(val_loss)
        val_ious.append(avg_val_iou)
        val_dices.append(avg_val_dice)

        logger.info(f'Validation set: Average loss: {val_loss:.4f}, IoU: {avg_val_iou:.4f}, Dice: {avg_val_dice:.4f}')
        
        # Print metrics for SageMaker CloudWatch parsing
        print(f'sagemaker_metric: Train_Loss={avg_train_loss:.6f}')
        print(f'sagemaker_metric: Validation_Loss={val_loss:.6f}')
        print(f'sagemaker_metric: Validation_IoU={avg_val_iou:.6f}')
        print(f'sagemaker_metric: Validation_Dice={avg_val_dice:.6f}')

        # Learning rate scheduler step
        scheduler.step(avg_val_iou) # Step based on validation IoU (mode='max')

        # Save model checkpoint if it's the best so far based on IoU
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            best_iou_epoch_num = epoch # Store the epoch number
            best_model_path = os.path.join(args.model_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with IoU: {best_val_iou:.4f} at epoch {best_iou_epoch_num}")
        
        # Save loss and metric curves and prediction visualizations
        if (epoch % args.plot_interval == 0 or epoch == args.epochs - 1):
            plot_curves(train_losses, val_losses, val_ious, val_dices, epoch, best_val_iou, best_iou_epoch_num, output_img_dir)

            # Save prediction visualization on first batch of val set
            model.eval()
            with torch.no_grad():
                # To ensure we get a consistent batch for visualization:
                # Store the first batch from val_loader or load specific examples
                val_data_iter = iter(val_loader)
                try:
                    val_images, val_masks = next(val_data_iter)
                except StopIteration:
                    # If val_loader is exhausted, reset it
                    val_loader_reset = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
                    val_images, val_masks = next(iter(val_loader_reset))

                val_images = val_images.to(device)
                with autocast():
                    preds = model(val_images)
                preds = torch.sigmoid(preds)
                preds = (preds > 0.5).float()
                
                # Plot first N samples
                for i in range(min(args.num_vis_samples, val_images.shape[0])):
                    fig, axs = plt.subplots(1, 4, figsize=(16, 4)) # Adjusted for 4 plots

                    # --- Plot 1: Input Image (RGB) ---
                    img_np = val_images[i].detach().cpu().numpy()
                    # Transpose to (H, W, C) for matplotlib imshow
                    if img_np.shape[0] > 3: # If multi-channel, take first 3 for RGB display
                        img_np_display = np.transpose(img_np[:3], (1, 2, 0))
                    else: # Assuming 1 or 3 channels already (C, H, W)
                        img_np_display = np.transpose(img_np, (1, 2, 0))
                    
                    # Normalize image for display if it's not already in [0,1]
                    # This depends on your SpaceNetBuildingDataset's normalization strategy.
                    # Example for normalization from [-1, 1] to [0, 1] for display:
                    # img_np_display = (img_np_display + 1) / 2.0
                    # For a general case, clamp to [0,1] if it's float:
                    img_np_display = np.clip(img_np_display, 0, 1)

                    axs[0].imshow(img_np_display)
                    axs[0].set_title('Input Image (RGB)', fontsize=10)

                    # --- Plot 2: Ground Truth Mask ---
                    gt_mask_np = val_masks[i][0].detach().cpu().numpy() # [0] because mask is 1xHxW
                    axs[1].imshow(gt_mask_np, cmap='gray', vmin=0, vmax=1)
                    axs[1].set_title('Ground Truth Mask', fontsize=10)

                    # --- Plot 3: Predicted Mask (Binary) ---
                    pred_mask_np = preds[i][0].detach().cpu().numpy() # [0] because prediction is 1xHxW
                    axs[2].imshow(pred_mask_np, cmap='gray', vmin=0, vmax=1)
                    axs[2].set_title('Predicted Mask (Binary)', fontsize=10)

                    # --- Plot 4: Overlayed Prediction ---
                    axs[3].imshow(img_np_display) # Start with the image background
                    # Create a red overlay for buildings (predicted=1)
                    overlay = np.zeros(img_np_display.shape)
                    overlay[:, :, 0] = pred_mask_np * 1.0 # Set red channel for predicted buildings
                    axs[3].imshow(overlay, alpha=0.4) # Adjust alpha for transparency
                    axs[3].set_title('Prediction Overlay', fontsize=10)

                    for ax in axs:
                        ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_img_dir, f'val_pred_overlay_epoch{epoch}_sample{i}.png'))
                    plt.close()

    # Save the final trained model (can also just rely on best_model.pth)
    logger.info(f"Saving final model to {args.model_dir}/final_model.pth")
    with open(os.path.join(args.model_dir, 'final_model.pth'), 'wb') as f:
        torch.save(model.cpu().state_dict(), f)

    # Generate summary report at the end
    generate_summary_report(args, best_val_iou, best_iou_epoch_num, train_losses, val_losses, val_dices, output_img_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General training arguments
    parser.add_argument('--batch-size', type=int, default=16, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate for optimizer.')
    parser.add_argument('--log-interval', type=int, default=10, help='How many batches to wait before logging training status.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--gradient-clip-val', type=float, default=1.0, help='Gradient clipping value.')
    parser.add_argument('--lr-scheduler-patience', type=int, default=5, help='Patience for ReduceLROnPlateau scheduler (epochs).')
    parser.add_argument('--plot-interval', type=int, default=5, help='Interval (epochs) for saving plots and visualizations.')
    parser.add_argument('--num-vis-samples', type=int, default=3, help='Number of samples to visualize predictions for.')


    # Model specific arguments for HybridGeoNet
    parser.add_argument('--cnn-model', type=str, default='resnet50', help='CNN backbone model (e.g., resnet50, resnet101).')
    parser.add_argument('--num-input-channels', type=int, default=8, help='Number of input channels for satellite imagery (e.g., 3 for RGB, 8 for SpaceNet).')
    parser.add_argument('--pretrained-cnn', type=bool, default=True, help='Use pretrained weights for CNN backbone.')
    parser.add_argument('--vit-embed-dim', type=int, default=768, help='Embedding dimension for Vision Transformer.')
    parser.add_argument('--vit-depth', type=int, default=4, help='Number of Transformer encoder layers (depth).')
    parser.add_argument('--vit-heads', type=int, default=12, help='Number of attention heads in ViT.')
    parser.add_argument('--gat-heads', type=int, default=4, help='Number of attention heads in GAT.')


    # SageMaker specific arguments (default values from environment variables)
    parser.add_argument('--train-images-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_IMAGES', 'data/train_images'), help='Path to training images.')
    parser.add_argument('--train-masks-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_MASKS', 'data/train_masks'), help='Path to training masks.')
    parser.add_argument('--val-images-dir', type=str, default=os.environ.get('SM_CHANNEL_VAL_IMAGES', 'data/val_images'), help='Path to validation images.')
    parser.add_argument('--val-masks-dir', type=str, default=os.environ.get('SM_CHANNEL_VAL_MASKS', 'data/val_masks'), help='Path to validation masks.')
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'model'), help='Path to save the trained model artifacts.')
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', 'output'), help='Path to save output data (e.g., plots).')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cuda or cpu).')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train(args)