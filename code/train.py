# import argparse
# import sys
# import subprocess

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# for pkg in ["rasterio", "geopandas", "shapely", "tqdm", "scikit-learn", "pandas"]:
#     try:
#         __import__(pkg)
#     except ImportError:
#         install(pkg)

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler # For Mixed Precision Training
# import logging
# import sys
# import numpy as np # For random seed and image visualization
# import matplotlib.pyplot as plt
# import pandas as pd # For smoothing curves
# from datetime import datetime # For timestamp in summary report

# # Import your custom modules
# from model import HybridGeoNet # Assuming model.py defines HybridGeoNet

# # --- Import the real SpaceNetBuildingDataset implementation ---
# from dataset import SpaceNetBuildingDataset
# # The actual implementation is now used for real image and mask loading.

# # Set a robust default for S3 bucket
# DEFAULT_S3_BUCKET = "sagemaker-us-east-1-040571275415"


# # Setup logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler(sys.stdout))

# # --- Loss Functions ---
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, pred, target):
#         # Apply sigmoid to logits for Dice Loss
#         pred = torch.sigmoid(pred)

#         # Flatten label and prediction tensors
#         pred = pred.contiguous().view(-1)
#         target = target.contiguous().view(-1)

#         intersection = (pred * target).sum()
#         dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
#         return 1 - dice

# # --- Metrics ---
# def calculate_metrics(preds, targets, smooth=1e-6):
#     # preds are logits, convert to probabilities and then binary
#     preds = torch.sigmoid(preds)
#     preds = (preds > 0.5).float() # Thresholding at 0.5

#     intersection = (preds * targets).sum(dim=[1, 2, 3]) # Sum over C, H, W
#     union = (preds + targets).sum(dim=[1, 2, 3]) - intersection

#     # Avoid division by zero for empty masks
#     iou = (intersection + smooth) / (union + smooth)
#     dice = (2. * intersection + smooth) / (preds.sum(dim=[1, 2, 3]) + targets.sum(dim=[1, 2, 3]) + smooth)

#     return iou.mean(), dice.mean() # Return mean over batch

# # --- Plotting Functions ---
# def plot_curves(train_losses, val_losses, val_ious, val_dices, epoch, best_val_iou, best_iou_epoch_num, output_img_dir):
#     epochs_range = range(len(train_losses))

#     # Smoothing (optional, but helpful for noisy curves)
#     if len(train_losses) > 5: # Only smooth if enough data
#         smoothed_train_losses = pd.Series(train_losses).rolling(window=3, min_periods=1).mean()
#         smoothed_val_losses = pd.Series(val_losses).rolling(window=3, min_periods=1).mean()
#         smoothed_val_ious = pd.Series(val_ious).rolling(window=3, min_periods=1).mean()
#         smoothed_val_dices = pd.Series(val_dices).rolling(window=3, min_periods=1).mean()
#     else: # If not enough data points, no smoothing
#         smoothed_train_losses = train_losses
#         smoothed_val_losses = val_losses
#         smoothed_val_ious = val_ious
#         smoothed_val_dices = val_dices

#     # Plot Loss Curve
#     plt.figure(figsize=(12, 6))
#     plt.plot(epochs_range, smoothed_train_losses, label='Train Loss', color='blue', linestyle='-')
#     plt.plot(epochs_range, smoothed_val_losses, label='Validation Loss', color='red', linestyle='--')
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Loss', fontsize=12)
#     plt.title(f'Training & Validation Loss Over Epochs (Epoch {epoch + 1})', fontsize=14) # epoch + 1 for display
#     plt.legend(fontsize=10)
#     plt.grid(True, linestyle=':', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_img_dir, f'loss_curve_epoch{epoch}.png'))
#     plt.close()

#     # Plot Metric Curves (IoU & Dice)
#     plt.figure(figsize=(12, 6))
#     plt.plot(epochs_range, smoothed_val_ious, label='Validation IoU', color='green', linestyle='-')
#     plt.plot(epochs_range, smoothed_val_dices, label='Validation Dice', color='purple', linestyle='--')
    
#     # Highlight the best IoU point
#     if best_val_iou > -1.0: # If a best IoU has been recorded
#         plt.scatter(best_iou_epoch_num, best_val_iou, color='darkgreen', marker='*', s=200, zorder=5, label=f'Best IoU: {best_val_iou:.4f}')
#         plt.annotate(f'{best_val_iou:.4f}', (best_iou_epoch_num, best_val_iou),
#                      textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='darkgreen')

#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Metric Value', fontsize=12)
#     plt.title(f'Validation Metrics (IoU & Dice) Over Epochs (Epoch {epoch + 1})', fontsize=14) # epoch + 1 for display
#     plt.legend(fontsize=10)
#     plt.grid(True, linestyle=':', alpha=0.7)
#     plt.ylim(0, 1.05) # Metrics are usually between 0 and 1
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_img_dir, f'metrics_curve_epoch{epoch}.png'))
#     plt.close()

# # --- Summary Report Generation ---
# def generate_summary_report(args, best_val_iou, best_iou_epoch_num, train_losses, val_losses, val_dices, output_img_dir, metrics_table):
#     report_path = os.path.join(args.output_data_dir, 'training_summary_report.md')
#     with open(report_path, 'w') as f:
#         f.write(f"# Training Summary Report for Building Detector 🏗️\n\n")
#         f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"**Model:** HybridGeoNet\n")
#         f.write(f"**Dataset:** SpaceNet Las Vegas Building Detection (Sampled)\n\n") # Adjusted for placeholder dataset
        
#         f.write("## Key Performance Metrics\n")
#         f.write(f"- **Best Validation IoU**: {best_val_iou:.4f} (achieved at Epoch {best_iou_epoch_num})\n")
        
#         # Ensure lists are not empty before accessing last element
#         final_val_loss = val_losses[-1] if val_losses else "N/A"
#         final_val_dice = val_dices[-1] if val_dices else "N/A"

#         f.write(f"- **Final Validation Loss**: {final_val_loss:.4f}\n")
#         f.write(f"- **Final Validation Dice**: {final_val_dice:.4f}\n\n")
        
#         f.write("## Hyperparameters Used\n")
#         for arg, value in vars(args).items():
#             f.write(f"- `{arg}`: `{value}`\n")
#         f.write("\n")

#         f.write("## Visualizations\n")
#         f.write("Below are links to key plots and example predictions. All images are saved in the `images/` subdirectory of the output data path.\n\n")
        
#         # Link to the final loss/metric curves
#         f.write(f"### 📈 Performance Curves\n")
#         f.write(f"![Loss Curve](images/loss_curve_epoch{args.epochs-1}.png)\n")
#         f.write(f"![Metrics Curve](images/metrics_curve_epoch{args.epochs-1}.png)\n\n")

#         # Link to sample predictions (e.g., from the best epoch or final epoch)
#         f.write(f"### 👁️‍🗨️ Example Building Detections (Last Epoch: {args.epochs})\n")
#         for i in range(min(args.num_vis_samples, 3)): # Show a few examples
#             f.write(f"#### Sample {i+1}\n")
#             f.write(f"![Sample {i+1} Prediction Overlay](images/val_pred_overlay_epoch{args.epochs-1}_sample{i}.png)\n\n")
        
#         f.write("\n")
#         f.write("---\n")
#         f.write("## Summary Table of Epoch Metrics\n")
#         f.write("| Epoch | Train Loss | Validation Loss | Validation IoU | Validation Dice |\n")
#         f.write("| --- | --- | --- | --- | --- |\n")
#         for row in metrics_table:
#             f.write(f"| {row['epoch']} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | {row['val_iou']:.4f} | {row['val_dice']:.4f} |\n")
#     logger.info(f"Summary report generated at {report_path}")


# def train(args):
#     logger.info(f"Starting training on {args.device}...")
#     logger.info(f"Hyperparameters: {args}")

#     device = torch.device(args.device)

#     # Data paths from SageMaker environment variables or default args
#     train_image_dir = args.train_images_dir
#     train_mask_dir = args.train_masks_dir
#     val_image_dir = args.val_images_dir
#     val_mask_dir = args.val_masks_dir

#     # Data preparation
#     num_input_channels = args.num_input_channels

#     # Gracefully handle s3_bucket for SageMaker
    
#     S3_BUCKET = "sagemaker-us-east-1-040571275415"
#     s3_bucket = S3_BUCKET

#     #s3_bucket = getattr(args, 's3_bucket', os.environ.get('S3_BUCKET', DEFAULT_S3_BUCKET))
#     # Debug: List files in image/mask directories
#     import os
#     logger.info(f"Files in train_images_dir ({train_image_dir}): {os.listdir(train_image_dir) if os.path.exists(train_image_dir) else 'NOT FOUND'}")
#     logger.info(f"Files in train_masks_dir ({train_mask_dir}): {os.listdir(train_mask_dir) if os.path.exists(train_mask_dir) else 'NOT FOUND'}")
#     logger.info(f"Files in val_images_dir ({val_image_dir}): {os.listdir(val_image_dir) if os.path.exists(val_image_dir) else 'NOT FOUND'}")
#     logger.info(f"Files in val_masks_dir ({val_mask_dir}): {os.listdir(val_mask_dir) if os.path.exists(val_mask_dir) else 'NOT FOUND'}")
#     train_dataset = SpaceNetBuildingDataset(
#         s3_bucket=s3_bucket,
#         image_s3_path_prefix=train_image_dir,
#         mask_s3_path_prefix=train_mask_dir,
#         augment_mode='train',
#         num_input_channels=num_input_channels
#     )
#     logger.info(f"[PAIRING CHECK] Train dataset: {len(train_dataset)} valid image-mask pairs found.")
#     val_dataset = SpaceNetBuildingDataset(
#         s3_bucket=s3_bucket,
#         image_s3_path_prefix=val_image_dir,
#         mask_s3_path_prefix=val_mask_dir,
#         augment_mode='val',
#         num_input_channels=num_input_channels
#     )
#     logger.info(f"[PAIRING CHECK] Val dataset: {len(val_dataset)} valid image-mask pairs found.")

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=True
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True
#     )
    
#     logger.info(f"Train dataset size: {len(train_dataset)}")
#     logger.info(f"Validation dataset size: {len(val_dataset)}")

#     # Model, Loss, Optimizer
#     model = HybridGeoNet( # Using HybridGeoNet
#         cnn_model=args.cnn_model,
#         num_input_channels=num_input_channels,
#         vit_embed_dim=args.vit_embed_dim,
#         # vit_depth is currently not used directly in HybridGeoNet as only one HybridViTGATBlock is defined.
#         # If you stack multiple, you would use this arg.
#         vit_depth=args.vit_depth, 
#         vit_heads=args.vit_heads,
#         gat_heads=args.gat_heads,
#         num_classes=1, # For binary building detection
#         pretrained_cnn=args.pretrained_cnn
#     ).to(device)

#     # Combined Loss Function
#     criterion_bce = nn.BCEWithLogitsLoss()
#     criterion_dice = DiceLoss()
    
#     # Optimizer
#     optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate) # AdamW is often better for Transformers

#     # Learning Rate Scheduler
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_scheduler_patience, verbose=True, min_lr=1e-6)
#     # Using 'max' mode for scheduler to track increasing IoU/Dice

#     # GradScaler for Mixed Precision Training
#     scaler = GradScaler()

#     # Training loop variables
#     train_losses = []
#     val_losses = []
#     val_ious = []
#     val_dices = []
#     best_val_iou = -1.0 # Initialize with a low value for tracking best model
#     best_iou_epoch_num = -1 # To store epoch number of best IoU
    
#     output_img_dir = os.path.join(args.output_data_dir, "images")
#     os.makedirs(output_img_dir, exist_ok=True)

#     # --- Initialize metrics table for CSV output ---
#     metrics_table = []

#     for epoch in range(args.epochs):
#         model.train()
#         running_loss = 0.0
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.to(device), target.to(device)

#             optimizer.zero_grad()
            
#             with autocast(): # Enable mixed precision
#                 output = model(data)
#                 loss_bce = criterion_bce(output, target)
#                 loss_dice = criterion_dice(output, target)
#                 loss = loss_bce + loss_dice # Simple sum, can be weighted (e.g., 0.5*bce + 0.5*dice)
            
#             scaler.scale(loss).backward() # Scale loss and call backward
            
#             # Gradient Clipping
#             scaler.unscale_(optimizer) # Unscale gradients before clipping
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_val)
            
#             scaler.step(optimizer)        # Update optimizer
#             scaler.update()               # Update scaler for next iteration
#             running_loss += loss.item()

#             if batch_idx % args.log_interval == 0:
#                 logger.info(f"Train Epoch: {epoch} [{batch_idx * args.batch_size}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.6f}")
        
#         avg_train_loss = running_loss / len(train_loader)
#         train_losses.append(avg_train_loss)
#         logger.info(f'Epoch {epoch} Training Loss: {avg_train_loss:.4f}')

#         # Validation loop
#         model.eval()
#         val_loss = 0
#         val_total_iou = 0
#         val_total_dice = 0
#         with torch.no_grad():
#             for data, target in val_loader:
#                 data, target = data.to(device), target.to(device)
#                 with autocast(): # Also use mixed precision for validation
#                     output = model(data)
#                     loss_bce = criterion_bce(output, target)
#                     loss_dice = criterion_dice(output, target)
#                     current_loss = loss_bce + loss_dice
                
#                 val_loss += current_loss.item() * data.size(0)

#                 # Calculate metrics
#                 batch_iou, batch_dice = calculate_metrics(output, target)
#                 val_total_iou += batch_iou.item() * data.size(0)
#                 val_total_dice += batch_dice.item() * data.size(0)

#         val_loss /= len(val_loader.dataset)
#         avg_val_iou = val_total_iou / len(val_loader.dataset)
#         avg_val_dice = val_total_dice / len(val_loader.dataset)

#         val_losses.append(val_loss)
#         val_ious.append(avg_val_iou)
#         val_dices.append(avg_val_dice)

#         logger.info(f'Validation set: Average loss: {val_loss:.4f}, IoU: {avg_val_iou:.4f}, Dice: {avg_val_dice:.4f}')

#         # --- Append metrics to table for CSV ---
#         metrics_table.append({
#             'epoch': epoch,
#             'train_loss': avg_train_loss,
#             'val_loss': val_loss,
#             'val_iou': avg_val_iou,
#             'val_dice': avg_val_dice,
#         })
        
#         # Print metrics for SageMaker CloudWatch parsing
#         print(f'sagemaker_metric: Train_Loss={avg_train_loss:.6f}')
#         print(f'sagemaker_metric: Validation_Loss={val_loss:.6f}')
#         print(f'sagemaker_metric: Validation_IoU={avg_val_iou:.6f}')
#         print(f'sagemaker_metric: Validation_Dice={avg_val_dice:.6f}')

#         # Learning rate scheduler step
#         scheduler.step(avg_val_iou) # Step based on validation IoU (mode='max')

#         # Save model checkpoint if it's the best so far based on IoU
#         if avg_val_iou > best_val_iou:
#             best_val_iou = avg_val_iou
#             best_iou_epoch_num = epoch # Store the epoch number
#             best_model_path = os.path.join(args.model_dir, 'best_model.pth')
#             torch.save(model.state_dict(), best_model_path)
#             logger.info(f"New best model saved with IoU: {best_val_iou:.4f} at epoch {best_iou_epoch_num}")
        
#         # Save loss and metric curves and prediction visualizations
#         if (epoch % args.plot_interval == 0 or epoch == args.epochs - 1):
#             plot_curves(train_losses, val_losses, val_ious, val_dices, epoch, best_val_iou, best_iou_epoch_num, output_img_dir)

#             # Save prediction visualization on first batch of val set
#             model.eval()
#             with torch.no_grad():
#                 # To ensure we get a consistent batch for visualization:
#                 # Store the first batch from val_loader or load specific examples
#                 val_data_iter = iter(val_loader)
#                 try:
#                     val_images, val_masks = next(val_data_iter)
#                 except StopIteration:
#                     # If val_loader is exhausted, reset it
#                     val_loader_reset = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
#                     val_images, val_masks = next(iter(val_loader_reset))

#                 val_images = val_images.to(device)
#                 with autocast():
#                     preds = model(val_images)
#                 preds = torch.sigmoid(preds)
#                 preds = (preds > 0.5).float()
                
#                 # Plot first N samples with improved overlays and titles
#                 for i in range(min(args.num_vis_samples, val_images.shape[0])):
#                     fig, axs = plt.subplots(1, 4, figsize=(18, 4))

#                     # --- Plot 1: Input Image (RGB) ---
#                     img_np = val_images[i].detach().cpu().numpy()
#                     if img_np.shape[0] > 3:
#                         img_np_display = np.transpose(img_np[:3], (1, 2, 0))
#                     else:
#                         img_np_display = np.transpose(img_np, (1, 2, 0))
#                     img_np_display = np.clip(img_np_display, 0, 1)

#                     axs[0].imshow(img_np_display)
#                     axs[0].set_title('Input Image (RGB)', fontsize=12, wrap=True)

#                     # --- Plot 2: Ground Truth Mask ---
#                     gt_mask_np = val_masks[i][0].detach().cpu().numpy()
#                     axs[1].imshow(img_np_display, alpha=0.7)
#                     axs[1].imshow(gt_mask_np, cmap='spring', alpha=0.4, vmin=0, vmax=1)
#                     axs[1].set_title('GT Mask Overlay', fontsize=12, wrap=True)

#                     # --- Plot 3: Predicted Mask (Binary) ---
#                     pred_mask_np = preds[i][0].detach().cpu().numpy()
#                     axs[2].imshow(img_np_display, alpha=0.7)
#                     axs[2].imshow(pred_mask_np, cmap='autumn', alpha=0.4, vmin=0, vmax=1)
#                     axs[2].set_title('Pred Mask Overlay', fontsize=12, wrap=True)

#                     # --- Plot 4: Overlayed Prediction vs GT ---
#                     axs[3].imshow(img_np_display, alpha=0.7)
#                     axs[3].imshow(gt_mask_np, cmap='spring', alpha=0.3, vmin=0, vmax=1)
#                     axs[3].imshow(pred_mask_np, cmap='autumn', alpha=0.3, vmin=0, vmax=1)
#                     axs[3].set_title(f'Overlay GT (green) & Pred (orange)\nEpoch {epoch} Sample {i}', fontsize=11, wrap=True)

#                     for ax in axs:
#                         ax.axis('off')
#                     plt.tight_layout(rect=[0, 0, 1, 0.95])
#                     plt.subplots_adjust(top=0.85)
#                     fig.suptitle(f'Validation Visualization\nEpoch {epoch} Sample {i}', fontsize=14)
#                     plt.savefig(os.path.join(output_img_dir, f'val_pred_overlay_epoch{epoch}_sample{i}.png'), dpi=120)
#                     plt.close()

#     # --- Save metrics table as CSV for review ---
#     metrics_df = pd.DataFrame(metrics_table)
#     metrics_csv_path = os.path.join(args.output_data_dir, 'metrics_per_epoch.csv')
#     metrics_df.to_csv(metrics_csv_path, index=False)
#     logger.info(f"Saved per-epoch metrics table at {metrics_csv_path}")

#     # Save the final trained model (can also just rely on best_model.pth)
#     logger.info(f"Saving final model to {args.model_dir}/final_model.pth")
#     with open(os.path.join(args.model_dir, 'final_model.pth'), 'wb') as f:
#         torch.save(model.cpu().state_dict(), f)

#     # Generate summary report at the end
#     generate_summary_report(args, best_val_iou, best_iou_epoch_num, train_losses, val_losses, val_dices, output_img_dir)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # General training arguments
#     parser.add_argument('--batch-size', type=int, default=16, help='Input batch size for training.')
#     parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
#     parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate for optimizer.')
#     parser.add_argument('--log-interval', type=int, default=10, help='How many batches to wait before logging training status.')
#     parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers.')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
#     parser.add_argument('--gradient-clip-val', type=float, default=1.0, help='Gradient clipping value.')
#     parser.add_argument('--lr-scheduler-patience', type=int, default=5, help='Patience for ReduceLROnPlateau scheduler (epochs).')
#     parser.add_argument('--plot-interval', type=int, default=5, help='Interval (epochs) for saving plots and visualizations.')
#     parser.add_argument('--num-vis-samples', type=int, default=3, help='Number of samples to visualize predictions for.')


#     # Model specific arguments for HybridGeoNet
#     parser.add_argument('--cnn-model', type=str, default='resnet50', help='CNN backbone model (e.g., resnet50, resnet101).')
#     parser.add_argument('--num-input-channels', type=int, default=8, help='Number of input channels for satellite imagery (e.g., 3 for RGB, 8 for SpaceNet).')
#     parser.add_argument('--pretrained-cnn', type=bool, default=True, help='Use pretrained weights for CNN backbone.')
#     parser.add_argument('--vit-embed-dim', type=int, default=768, help='Embedding dimension for Vision Transformer.')
#     parser.add_argument('--vit-depth', type=int, default=4, help='Number of Transformer encoder layers (depth).')
#     parser.add_argument('--vit-heads', type=int, default=12, help='Number of attention heads in ViT.')
#     parser.add_argument('--gat-heads', type=int, default=4, help='Number of attention heads in GAT.')


#     # SageMaker specific arguments (default values from environment variables)
#     parser.add_argument('--train-images-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_IMAGES', 'data/train_images'), help='Path to training images.')
#     parser.add_argument('--train-masks-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_MASKS', 'data/train_masks'), help='Path to training masks.')
#     parser.add_argument('--val-images-dir', type=str, default=os.environ.get('SM_CHANNEL_VAL_IMAGES', 'data/val_images'), help='Path to validation images.')
#     parser.add_argument('--val-masks-dir', type=str, default=os.environ.get('SM_CHANNEL_VAL_MASKS', 'data/val_masks'), help='Path to validation masks.')
#     parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'model'), help='Path to save the trained model artifacts.')
#     parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', 'output'), help='Path to save output data (e.g., plots).')
#     parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cuda or cpu).')

#     args = parser.parse_args()

#     # Set random seeds for reproducibility
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     if torch.cuda.is_available() and args.device == 'cuda':
#         torch.cuda.manual_seed(args.seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

#     train(args)



###########################################################################
#Gemini Takeover
###########################################################################


# In code/train.py

# In code/train.py

# import argparse
# import sys
# import subprocess
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler
# import logging
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from datetime import datetime
# from tqdm import tqdm

# # Import your custom modules
# from model import HybridGeoNet
# from dataset import SpaceNetBuildingDataset, calculate_stats # Import the calculator

# # --- Logger Setup ---
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))

# # --- Loss & Metrics ---
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, pred, target):
#         pred = torch.sigmoid(pred)
#         pred = pred.contiguous().view(-1)
#         target = target.contiguous().view(-1)
#         intersection = (pred * target).sum()
#         dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
#         return 1 - dice

# def calculate_iou(preds, targets, smooth=1e-6):
#     preds = torch.sigmoid(preds) > 0.5
#     targets = targets > 0.5
#     intersection = (preds & targets).float().sum((1, 2, 3))
#     union = (preds | targets).float().sum((1, 2, 3))
#     iou = (intersection + smooth) / (union + smooth)
#     return iou.mean()

# # --- Main Training Function ---
# def train(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")
    
#     # Calculate normalization stats from the downloaded training data
#     mean, std = calculate_stats(args.train_images_dir, args.num_input_channels)
#     logger.info(f"Calculated Mean: {mean}")
#     logger.info(f"Calculated Std Dev: {std}")

#     # Initialize datasets using local paths
#     train_dataset = SpaceNetBuildingDataset(args.train_images_dir, args.train_masks_dir, 'train', args.num_input_channels, mean, std)
#     val_dataset = SpaceNetBuildingDataset(args.val_images_dir, args.val_masks_dir, 'val', args.num_input_channels, mean, std)

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

#     # Initialize model, loss, and optimizer
#     model = HybridGeoNet(
#         cnn_model=args.cnn_model, num_input_channels=args.num_input_channels,
#         vit_embed_dim=args.vit_embed_dim, vit_depth=args.vit_depth,
#         vit_heads=args.vit_heads, gat_heads=args.gat_heads,
#         num_classes=1, pretrained_cnn=args.pretrained_cnn
#     ).to(device)

#     criterion_bce = nn.BCEWithLogitsLoss()
#     criterion_dice = DiceLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_scheduler_patience, verbose=True)
#     scaler = GradScaler()
    
#     best_val_iou = 0.0

#     # Training loop
#     for epoch in range(args.epochs):
#         model.train()
#         train_loss = 0.0
#         for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
#             images, masks = images.to(device), masks.to(device)
            
#             optimizer.zero_grad()
#             with autocast():
#                 outputs = model(images)
#                 loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
            
#             scaler.scale(loss).backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_val)
#             scaler.step(optimizer)
#             scaler.update()
            
#             train_loss += loss.item()
        
#         avg_train_loss = train_loss / len(train_loader)
#         logger.info(f"Epoch {epoch+1}: Avg Training Loss: {avg_train_loss:.4f}")

#         # Validation loop
#         model.eval()
#         val_loss, val_iou = 0.0, 0.0
#         with torch.no_grad():
#             for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
#                 images, masks = images.to(device), masks.to(device)
#                 with autocast():
#                     outputs = model(images)
#                     loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
#                 val_loss += loss.item()
#                 val_iou += calculate_iou(outputs, masks).item()

#         avg_val_loss = val_loss / len(val_loader)
#         avg_val_iou = val_iou / len(val_loader)
#         logger.info(f"Epoch {epoch+1}: Avg Validation Loss: {avg_val_loss:.4f}, Avg IoU: {avg_val_iou:.4f}")

#         scheduler.step(avg_val_iou)

#         # Save best model
#         if avg_val_iou > best_val_iou:
#             best_val_iou = avg_val_iou
#             torch.save(model.state_dict(), os.path.join(args.model_dir, "best_model.pth"))
#             logger.info(f"New best model saved with IoU: {best_val_iou:.4f}")

#     # Save final model
#     torch.save(model.state_dict(), os.path.join(args.model_dir, "final_model.pth"))
#     logger.info("Training complete. Final model saved.")

#     # --- Robust Output Saving: Real Training Outputs ---
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime

# def train(args):
#     # Initialize metrics lists and best-tracking variables
#     train_losses = []
#     val_losses = []
#     val_ious = []
#     val_dices = []
#     best_val_iou = -1.0
#     best_iou_epoch_num = -1
#     # --- Robust Output Saving: Real Training Outputs ---
#     output_artifacts_dir = os.path.join('/opt/ml/output', 'artifacts')
#     images_dir = os.path.join(output_artifacts_dir, 'images')
#     os.makedirs(images_dir, exist_ok=True)

#     # === Collect metrics during training ===
#     # These lists must be initialized at the top of train() and updated in the training loop:
#     # train_losses, val_losses, val_ious, val_dices, best_val_iou, best_iou_epoch_num
#     # (rest of your output-saving logic...)

#     # We'll re-run the loop below to collect them if not already present

#     # (If not already present, define these at the top of train):
#     # train_losses, val_losses, val_ious, val_dices = [], [], [], []
#     # best_val_iou = -1.0
#     # best_iou_epoch_num = -1
#     # metrics_table = []

#     # Save metrics CSV (real values)
#     metrics_table = []
#     for epoch in range(args.epochs):
#         metrics_table.append({
#             'epoch': epoch+1,
#             'train_loss': train_losses[epoch] if len(train_losses) > epoch else None,
#             'val_loss': val_losses[epoch] if len(val_losses) > epoch else None,
#             'val_iou': val_ious[epoch] if len(val_ious) > epoch else None,
#             'val_dice': val_dices[epoch] if len(val_dices) > epoch else None,
#         })
#     metrics_df = pd.DataFrame(metrics_table)
#     metrics_csv_path = os.path.join(output_artifacts_dir, 'metrics_per_epoch.csv')
#     metrics_df.to_csv(metrics_csv_path, index=False)
#     logger.info(f"Saved metrics CSV at {metrics_csv_path}")

#     # Save loss curve (last epoch)
#     plt.figure(figsize=(12, 6))
#     plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', color='blue', linestyle='-')
#     plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', color='red', linestyle='--')
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Loss', fontsize=12)
#     plt.title(f'Training & Validation Loss Over Epochs', fontsize=14)
#     plt.legend(fontsize=10)
#     plt.grid(True, linestyle=':', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig(os.path.join(images_dir, f'loss_curve_epoch{args.epochs-1}.png'))
#     plt.close()

#     # Save metric curves (IoU & Dice)
#     plt.figure(figsize=(12, 6))
#     plt.plot(range(1, len(val_ious)+1), val_ious, label='Validation IoU', color='green', linestyle='-')
#     plt.plot(range(1, len(val_dices)+1), val_dices, label='Validation Dice', color='purple', linestyle='--')
#     if best_val_iou > -1.0:
#         plt.scatter(best_iou_epoch_num, best_val_iou, color='darkgreen', marker='*', s=200, zorder=5, label=f'Best IoU: {best_val_iou:.4f}')
#         plt.annotate(f'{best_val_iou:.4f}', (best_iou_epoch_num, best_val_iou), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='darkgreen')
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Metric Value', fontsize=12)
#     plt.title(f'Validation Metrics (IoU & Dice) Over Epochs', fontsize=14)
#     plt.legend(fontsize=10)
#     plt.grid(True, linestyle=':', alpha=0.7)
#     plt.ylim(0, 1.05)
#     plt.tight_layout()
#     plt.savefig(os.path.join(images_dir, f'metrics_curve_epoch{args.epochs-1}.png'))
#     plt.close()

#     # Save prediction overlays for last epoch (or best epoch)
#     # Use a batch from val_loader
#     model.eval()
#     with torch.no_grad():
#         val_data_iter = iter(val_loader)
#         try:
#             val_images, val_masks = next(val_data_iter)
#         except StopIteration:
#             val_loader_reset = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
#             val_images, val_masks = next(iter(val_loader_reset))
#         val_images = val_images.to(device)
#         with autocast():
#             preds = model(val_images)
#         preds = torch.sigmoid(preds)
#         preds = (preds > 0.5).float()
#         for i in range(min(args.num_vis_samples, val_images.shape[0])):
#             fig, axs = plt.subplots(1, 4, figsize=(18, 4))
#             img_np = val_images[i].detach().cpu().numpy()
#             if img_np.shape[0] > 3:
#                 img_np_display = np.transpose(img_np[:3], (1, 2, 0))
#             else:
#                 img_np_display = np.transpose(img_np, (1, 2, 0))
#             img_np_display = np.clip(img_np_display, 0, 1)
#             axs[0].imshow(img_np_display)
#             axs[0].set_title('Input Image (RGB)', fontsize=12, wrap=True)
#             gt_mask_np = val_masks[i][0].detach().cpu().numpy()
#             axs[1].imshow(img_np_display, alpha=0.7)
#             axs[1].imshow(gt_mask_np, cmap='spring', alpha=0.4, vmin=0, vmax=1)
#             axs[1].set_title('GT Mask Overlay', fontsize=12, wrap=True)
#             pred_mask_np = preds[i][0].detach().cpu().numpy()
#             axs[2].imshow(img_np_display, alpha=0.7)
#             axs[2].imshow(pred_mask_np, cmap='autumn', alpha=0.4, vmin=0, vmax=1)
#             axs[2].set_title('Pred Mask Overlay', fontsize=12, wrap=True)
#             axs[3].imshow(img_np_display, alpha=0.7)
#             axs[3].imshow(gt_mask_np, cmap='spring', alpha=0.3, vmin=0, vmax=1)
#             axs[3].imshow(pred_mask_np, cmap='autumn', alpha=0.3, vmin=0, vmax=1)
#             axs[3].set_title(f'Overlay GT (green) & Pred (orange)\nSample {i}', fontsize=11, wrap=True)
#             for ax in axs:
#                 ax.axis('off')
#             plt.tight_layout(rect=[0, 0, 1, 0.95])
#             plt.subplots_adjust(top=0.85)
#             fig.suptitle(f'Validation Visualization\nSample {i}', fontsize=14)
#             plt.savefig(os.path.join(images_dir, f'val_pred_overlay_epoch{args.epochs-1}_sample{i}.png'), dpi=120)
#             plt.close()

#     # Save summary report
#     summary_path = os.path.join(output_artifacts_dir, 'training_summary_report.md')
#     with open(summary_path, 'w') as f:
#         f.write(f"# Training Summary Report for Building Detector 🏗️\n\n")
#         f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"**Model:** HybridGeoNet\n")
#         f.write(f"**Epochs:** {args.epochs}\n")
#         f.write(f"**Batch Size:** {args.batch_size}\n")
#         f.write(f"**Best Validation IoU:** {best_val_iou:.4f} (achieved at Epoch {best_iou_epoch_num})\n")
#         f.write(f"**Input Channels:** {args.num_input_channels}\n\n")
#         f.write("## Visualizations\n")
#         f.write("Below are links to key plots and example predictions. All images are saved in the `images/` subdirectory of the output data path.\n\n")
#         f.write(f"### 📈 Performance Curves\n")
#         f.write(f"![Loss Curve](images/loss_curve_epoch{args.epochs-1}.png)\n")
#         f.write(f"![Metrics Curve](images/metrics_curve_epoch{args.epochs-1}.png)\n\n")
#         f.write(f"### 👁️‍🗨️ Example Building Detections (Last Epoch: {args.epochs})\n")
#         for i in range(min(args.num_vis_samples, len(val_images))):
#             f.write(f"#### Sample {i+1}\n")
#             f.write(f"![Sample {i+1} Prediction Overlay](images/val_pred_overlay_epoch{args.epochs-1}_sample{i}.png)\n\n")
#         f.write("\n---\n")
#         f.write("## Summary Table of Epoch Metrics\n")
#         f.write("| Epoch | Train Loss | Validation Loss | Validation IoU | Validation Dice |\n")
#         f.write("| --- | --- | --- | --- | --- |\n")
#         for row in metrics_table:
#             f.write(f"| {row['epoch']} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | {row['val_iou']:.4f} | {row['val_dice']:.4f} |\n")
#     logger.info(f"Saved summary report at {summary_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
    
#     # --- Data and Model Directories ---
#     parser.add_argument('--train-images-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_IMAGES'))
#     parser.add_argument('--train-masks-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_MASKS'))
#     parser.add_argument('--val-images-dir', type=str, default=os.environ.get('SM_CHANNEL_VAL_IMAGES'))
#     parser.add_argument('--val-masks-dir', type=str, default=os.environ.get('SM_CHANNEL_VAL_MASKS'))
#     parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
#     # --- All Hyperparameters ---
#     parser.add_argument('--batch-size', type=int, default=16)
#     parser.add_argument('--epochs', type=int, default=40)
#     parser.add_argument('--learning-rate', type=float, default=0.0001)
#     parser.add_argument('--num-workers', type=int, default=4)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--gradient-clip-val', type=float, default=1.0)
#     parser.add_argument('--lr-scheduler-patience', type=int, default=5)
#     parser.add_argument('--log-interval', type=int, default=10) # Used in more detailed logging
#     parser.add_argument('--plot-interval', type=int, default=5) # For visualization
#     parser.add_argument('--num-vis-samples', type=int, default=3) # For visualization

#     # Model specific
#     parser.add_argument('--num-input-channels', type=int, default=8)
#     parser.add_argument('--cnn-model', type=str, default='resnet50')
#     parser.add_argument('--pretrained-cnn', type=lambda x: (str(x).lower() == 'true'), default=True)
#     parser.add_argument('--vit-embed-dim', type=int, default=768)
#     parser.add_argument('--vit-depth', type=int, default=4)
#     parser.add_argument('--vit-heads', type=int, default=12)
#     parser.add_argument('--gat-heads', type=int, default=4)

#     args = parser.parse_args()
    
#     train(args)

###############
# Gemini 2.0
###############



# # In code/train.py

# import argparse
# import os
# import logging
# import sys
# import random
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler # Reverted to older import for compatibility
# import rasterio

# # Import your custom modules
# from dataset import SpaceNetBuildingDataset, calculate_stats
# from model import HybridGeoNet

# # --- Logger Setup ---
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))

# # --- Loss & Metrics ---
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, pred, target):
#         pred = torch.sigmoid(pred)
#         pred, target = pred.contiguous().view(-1), target.contiguous().view(-1)
#         intersection = (pred * target).sum()
#         dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
#         return 1 - dice

# def calculate_metrics(preds, targets, smooth=1e-6):
#     """Calculates both IoU and Dice score for a batch."""
#     preds = torch.sigmoid(preds) > 0.5
#     targets = targets > 0.5
    
#     # IoU calculation
#     intersection = (preds & targets).float().sum((1, 2, 3))
#     union = (preds | targets).float().sum((1, 2, 3))
#     iou = (intersection + smooth) / (union + smooth)

#     # Dice score calculation
#     preds_flat = preds.contiguous().view(-1)
#     targets_flat = targets.contiguous().view(-1)
#     intersection_dice = (preds_flat * targets_flat).sum()
#     dice_score = (2. * intersection_dice + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
    
#     return iou.mean(), dice_score.mean()


# # --- Main Training Function ---
# def train(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")

#     mean, std = calculate_stats(args.train_images_dir, args.num_input_channels)
#     logger.info(f"Calculated Mean: {mean}")
#     logger.info(f"Calculated Std Dev: {std}")

#     train_dataset = SpaceNetBuildingDataset(args.train_images_dir, args.train_masks_dir, 'train', args.num_input_channels, mean, std)
#     val_dataset = SpaceNetBuildingDataset(args.val_images_dir, args.val_masks_dir, 'val', args.num_input_channels, mean, std)

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

#     model = HybridGeoNet(
#         cnn_model=args.cnn_model, num_input_channels=args.num_input_channels,
#         vit_embed_dim=args.vit_embed_dim, vit_depth=args.vit_depth,
#         vit_heads=args.vit_heads, gat_heads=args.gat_heads,
#         num_classes=1, pretrained_cnn=args.pretrained_cnn
#     ).to(device)

#     criterion_bce = nn.BCEWithLogitsLoss()
#     criterion_dice = DiceLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_scheduler_patience, verbose=True)
#     scaler = GradScaler() # Corrected: Removed device_type argument
    
#     best_val_iou = 0.0

#     for epoch in range(args.epochs):
#         model.train()
#         train_loss = 0.0
#         for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
#             images, masks = images.to(device), masks.to(device)
#             optimizer.zero_grad(set_to_none=True)
#             with autocast(): # Corrected: Removed device_type and dtype arguments
#                 outputs = model(images)
#                 loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_val)
#             scaler.step(optimizer)
#             scaler.update()
#             train_loss += loss.item()
        
#         avg_train_loss = train_loss / len(train_loader)
#         logger.info(f"Epoch {epoch+1}: Avg Training Loss: {avg_train_loss:.4f}")

#         model.eval()
#         val_loss, val_iou, val_dice = 0.0, 0.0, 0.0
#         with torch.no_grad():
#             for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
#                 images, masks = images.to(device), masks.to(device)
#                 with autocast():
#                     outputs = model(images)
#                     loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
#                 val_loss += loss.item()
#                 iou, dice = calculate_metrics(outputs, masks)
#                 val_iou += iou.item()
#                 val_dice += dice.item()

#         avg_val_loss = val_loss / len(val_loader)
#         avg_val_iou = val_iou / len(val_loader)
#         avg_val_dice = val_dice / len(val_loader)
        
#         logger.info(f"Epoch {epoch+1}: Avg Val Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}, Dice: {avg_val_dice:.4f}")
#         print(f'sagemaker_metric: Validation_IoU={avg_val_iou:.6f}')
        
#         scheduler.step(avg_val_iou)

#         if avg_val_iou > best_val_iou:
#             best_val_iou = avg_val_iou
#             torch.save(model.state_dict(), os.path.join(args.model_dir, "best_model.pth"))
#             logger.info(f"New best model saved with IoU: {best_val_iou:.4f}")

#     torch.save(model.state_dict(), os.path.join(args.model_dir, "final_model.pth"))
#     logger.info("Training complete.")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
    
#     # SageMaker paths
#     parser.add_argument('--train-images-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_IMAGES'))
#     parser.add_argument('--train-masks-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_MASKS'))
#     parser.add_argument('--val-images-dir', type=str, default=os.environ.get('SM_CHANNEL_VAL_IMAGES'))
#     parser.add_argument('--val-masks-dir', type=str, default=os.environ.get('SM_CHANNEL_VAL_MASKS'))
#     parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
#     # All hyperparameters
#     parser.add_argument('--batch-size', type=int, default=16)
#     parser.add_argument('--epochs', type=int, default=40)
#     parser.add_argument('--learning-rate', type=float, default=0.0001)
#     parser.add_argument('--num-workers', type=int, default=4)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--gradient-clip-val', type=float, default=1.0)
#     parser.add_argument('--lr-scheduler-patience', type=int, default=5)
#     parser.add_argument('--log-interval', type=int, default=10)
#     parser.add_argument('--plot-interval', type=int, default=5)
#     parser.add_argument('--num-vis-samples', type=int, default=3)
#     parser.add_argument('--num-input-channels', type=int, default=8)
#     parser.add_argument('--cnn-model', type=str, default='resnet50')
#     parser.add_argument('--pretrained-cnn', type=lambda x: (str(x).lower() == 'true'), default=True)
#     parser.add_argument('--vit-embed-dim', type=int, default=768)
#     parser.add_argument('--vit-depth', type=int, default=4)
#     parser.add_argument('--vit-heads', type=int, default=12)
#     parser.add_argument('--gat-heads', type=int, default=4)

#     args, _ = parser.parse_known_args()
    
#     train(args)



###############
# Gemini 3.0
###############







# # In code/train.py

# import argparse
# import os
# import logging
# import sys
# import random
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler
# import rasterio
# import matplotlib.pyplot as plt
# import pandas as pd
# from datetime import datetime

# # Import your custom modules
# from dataset import SpaceNetBuildingDataset, calculate_stats
# from model import HybridGeoNet

# # --- Logger Setup ---
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))

# # --- Loss & Metrics ---
# class TverskyLoss(nn.Module):
#     def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
#         super(TverskyLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.smooth = smooth

#     def forward(self, pred, target):
#         pred = torch.sigmoid(pred)
#         pred, target = pred.contiguous().view(-1), target.contiguous().view(-1)
#         tp = (pred * target).sum()
#         fn = (target * (1 - pred)).sum()
#         fp = (pred * (1 - target)).sum()
#         tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
#         return 1 - tversky_index

# def calculate_metrics(preds, targets, smooth=1e-6):
#     preds = torch.sigmoid(preds) > 0.5
#     targets = targets > 0.5
#     intersection = (preds & targets).float().sum((1, 2, 3))
#     union = (preds | targets).float().sum((1, 2, 3))
#     iou = (intersection + smooth) / (union + smooth)
#     dice_score = (2. * intersection + smooth) / (preds.float().sum((1, 2, 3)) + targets.float().sum((1, 2, 3)) + smooth)
#     return iou.mean(), dice_score.mean()

# # --- Main Training Function ---
# def train(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")

#     # --- Set random seeds ---
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     random.seed(args.seed)

#     # --- Data Loading ---
#     mean, std = calculate_stats(args.train_images_dir, args.num_input_channels)
#     logger.info(f"Calculated Stats -> Mean: {mean}, Std: {std}")

#     train_dataset = SpaceNetBuildingDataset(args.train_images_dir, args.train_masks_dir, 'train', args.num_input_channels, mean, std)
#     val_dataset = SpaceNetBuildingDataset(args.val_images_dir, args.val_masks_dir, 'val', args.num_input_channels, mean, std)

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

#     # --- Model Initialization (Corrected) ---
#     model_hyperparams = {
#         'num_input_channels': args.num_input_channels,
#         'cnn_model': args.cnn_model,
#         'pretrained_cnn': args.pretrained_cnn,
#         'vit_embed_dim': args.vit_embed_dim,
#         'vit_depth': args.vit_depth,
#         'vit_heads': args.vit_heads,
#         'gat_heads': args.gat_heads,
#         'num_classes': 1
#     }
#     model = HybridGeoNet(**model_hyperparams).to(device)

#     # --- Loss, Optimizer, Scaler ---
#     criterion_bce = nn.BCEWithLogitsLoss()
#     criterion_tversky = TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta)
#     optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_scheduler_patience, verbose=True)
#     scaler = GradScaler()
    
#     best_val_iou = 0.0

#     # --- Training Loop ---
#     for epoch in range(args.epochs):
#         model.train()
#         train_loss = 0.0
#         for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
#             images, masks = images.to(device), masks.to(device)
#             optimizer.zero_grad(set_to_none=True)
#             with autocast():
#                 outputs = model(images)
#                 loss = criterion_bce(outputs, masks) + criterion_tversky(outputs, masks)
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_val)
#             scaler.step(optimizer)
#             scaler.update()
#             train_loss += loss.item()
        
#         avg_train_loss = train_loss / len(train_loader)
#         logger.info(f"Epoch {epoch+1}: Avg Training Loss: {avg_train_loss:.4f}")

#         # --- Validation Loop ---
#         model.eval()
#         val_loss, val_iou = 0.0, 0.0
#         with torch.no_grad():
#             for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
#                 images, masks = images.to(device), masks.to(device)
#                 with autocast():
#                     outputs = model(images)
#                     loss = criterion_bce(outputs, masks) + criterion_tversky(outputs, masks)
#                 val_loss += loss.item()
#                 iou, _ = calculate_metrics(outputs, masks)
#                 val_iou += iou.item()

#         avg_val_loss = val_loss / len(val_loader)
#         avg_val_iou = val_iou / len(val_loader)
        
#         logger.info(f"Epoch {epoch+1}: Avg Val Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}")
#         print(f'sagemaker_metric: Validation_IoU={avg_val_iou:.6f}')
        
#         scheduler.step(avg_val_iou)

#         if avg_val_iou > best_val_iou:
#             best_val_iou = avg_val_iou
#             torch.save(model.state_dict(), os.path.join(args.model_dir, "best_model.pth"))
#             logger.info(f"New best model saved with IoU: {best_val_iou:.4f}")

#     torch.save(model.state_dict(), os.path.join(args.model_dir, "final_model.pth"))
#     logger.info("Training complete.")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
    
#     # SageMaker paths
#     parser.add_argument('--train-images-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_IMAGES'))
#     parser.add_argument('--train-masks-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_MASKS'))
#     parser.add_argument('--val-images-dir', type=str, default=os.environ.get('SM_CHANNEL_VAL_IMAGES'))
#     parser.add_argument('--val-masks-dir', type=str, default=os.environ.get('SM_CHANNEL_VAL_MASKS'))
#     parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
#     # Hyperparameters
#     parser.add_argument('--batch-size', type=int, default=16)
#     parser.add_argument('--epochs', type=int, default=40)
#     parser.add_argument('--learning-rate', type=float, default=0.0001)
#     parser.add_argument('--num-workers', type=int, default=4)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--gradient-clip-val', type=float, default=1.0)
#     parser.add_argument('--lr-scheduler-patience', type=int, default=5)
#     parser.add_argument('--tversky-alpha', type=float, default=0.3)
#     parser.add_argument('--tversky-beta', type=float, default=0.7)
    
#     # Model specific
#     parser.add_argument('--num-input-channels', type=int, default=8)
#     parser.add_argument('--cnn-model', type=str, default='resnet50')
#     parser.add_argument('--pretrained-cnn', type=lambda x: (str(x).lower() == 'true'), default=True)
#     parser.add_argument('--vit-embed-dim', type=int, default=768)
#     parser.add_argument('--vit-depth', type=int, default=4)
#     parser.add_argument('--vit-heads', type=int, default=12)
#     parser.add_argument('--gat-heads', type=int, default=4)
    
#     args, _ = parser.parse_known_args()
    
#     train(args)











# In code/train.py

import argparse
import os
import logging
import sys
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import rasterio
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Import your custom modules
from dataset import SpaceNetBuildingDataset, calculate_stats
from model import HybridGeoNet

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# --- Loss & Metrics ---
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred, target = pred.contiguous().view(-1), target.contiguous().view(-1)
        tp = (pred * target).sum()
        fn = (target * (1 - pred)).sum()
        fp = (pred * (1 - target)).sum()
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky_index

def calculate_metrics(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds) > 0.5
    targets = targets > 0.5
    intersection = (preds & targets).float().sum((1, 2, 3))
    union = (preds | targets).float().sum((1, 2, 3))
    iou = (intersection + smooth) / (union + smooth)
    dice_score = (2. * intersection + smooth) / (preds.float().sum((1, 2, 3)) + targets.float().sum((1, 2, 3)) + smooth)
    return iou.mean(), dice_score.mean()

def save_mask_images(images, true_masks, pred_masks, output_dir, epoch, num_images=4):
    """Saves a grid of original images, true masks, and predicted masks."""
    output_dir = Path(output_dir) / "predicted_masks"
    output_dir.mkdir(exist_ok=True)
    
    # Ensure we don't try to save more images than are in the batch
    num_images = min(num_images, images.shape[0])

    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    fig.suptitle(f"Epoch {epoch+1} Predictions", fontsize=16)

    # Denormalize image for visualization if necessary (assuming it's just a single channel for plotting)
    # This part may need adjustment based on your specific normalization and number of channels
    images_to_plot = images.cpu().numpy()
    true_masks_to_plot = true_masks.cpu().numpy().squeeze(1)
    pred_masks_to_plot = (torch.sigmoid(pred_masks).cpu().numpy() > 0.5).squeeze(1)

    for i in range(num_images):
        # Plot original image (displaying the first channel)
        ax = axes[i, 0]
        ax.imshow(images_to_plot[i][0], cmap='gray')
        ax.set_title(f"Image {i+1}")
        ax.axis('off')

        # Plot true mask
        ax = axes[i, 1]
        ax.imshow(true_masks_to_plot[i], cmap='gray')
        ax.set_title(f"True Mask {i+1}")
        ax.axis('off')

        # Plot predicted mask
        ax = axes[i, 2]
        ax.imshow(pred_masks_to_plot[i], cmap='gray')
        ax.set_title(f"Predicted Mask {i+1}")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / f"epoch_{epoch+1}_predictions.png")
    plt.close()


# --- Main Training Function ---
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Set random seeds ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # --- Data Loading ---
    mean, std = calculate_stats(args.train_images_dir, args.num_input_channels)
    logger.info(f"Calculated Stats -> Mean: {mean}, Std: {std}")

    train_dataset = SpaceNetBuildingDataset(args.train_images_dir, args.train_masks_dir, 'train', args.num_input_channels, mean, std)
    val_dataset = SpaceNetBuildingDataset(args.val_images_dir, args.val_masks_dir, 'val', args.num_input_channels, mean, std)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Model Initialization ---
    model_hyperparams = {
        'num_input_channels': args.num_input_channels,
        'cnn_model': args.cnn_model,
        'pretrained_cnn': args.pretrained_cnn,
        'vit_embed_dim': args.vit_embed_dim,
        'vit_depth': args.vit_depth,
        'vit_heads': args.vit_heads,
        'gat_heads': args.gat_heads,
        'num_classes': 1
    }
    model = HybridGeoNet(**model_hyperparams).to(device)

    # --- Loss, Optimizer, Scaler ---
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_tversky = TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_scheduler_patience, verbose=True)
    scaler = GradScaler()
    
    best_val_iou = 0.0
    
    # --- History Tracking ---
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': []
    }

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(images)
                loss = criterion_bce(outputs, masks) + criterion_tversky(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        logger.info(f"Epoch {epoch+1}: Avg Training Loss: {avg_train_loss:.4f}")

        # --- Validation Loop ---
        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")):
                images, masks = images.to(device), masks.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion_bce(outputs, masks) + criterion_tversky(outputs, masks)
                val_loss += loss.item()
                iou, _ = calculate_metrics(outputs, masks)
                val_iou += iou.item()
                
                # Save mask images on the first batch of the last epoch
                if epoch == args.epochs - 1 and i == 0:
                    logger.info("Saving predicted mask images...")
                    save_mask_images(images.cpu(), masks.cpu(), outputs.cpu(), args.output_dir, epoch)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(avg_val_iou)
        
        logger.info(f"Epoch {epoch+1}: Avg Val Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f}")
        print(f'sagemaker_metric: Validation_IoU={avg_val_iou:.6f}')
        
        scheduler.step(avg_val_iou)

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), os.path.join(args.model_dir, "best_model.pth"))
            logger.info(f"New best model saved with IoU: {best_val_iou:.4f}")

    torch.save(model.state_dict(), os.path.join(args.model_dir, "final_model.pth"))
    logger.info("Training complete.")

    # --- Save Training Summary and Plots ---
    logger.info(f"Saving training summary and plots to {args.output_dir}")
    
    # Save metrics to CSV
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(args.output_dir, "training_summary.csv"), index_label="Epoch")

    # Save loss curve plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker paths
    parser.add_argument('--train-images-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_IMAGES'))
    parser.add_argument('--train-masks-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN_MASKS'))
    parser.add_argument('--val-images-dir', type=str, default=os.environ.get('SM_CHANNEL_VAL_IMAGES'))
    parser.add_argument('--val-masks-dir', type=str, default=os.environ.get('SM_CHANNEL_VAL_MASKS'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    
    # Hyperparameters
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-clip-val', type=float, default=1.0)
    parser.add_argument('--lr-scheduler-patience', type=int, default=5)
    parser.add_argument('--tversky-alpha', type=float, default=0.3)
    parser.add_argument('--tversky-beta', type=float, default=0.7)
    
    # Model specific
    parser.add_argument('--num-input-channels', type=int, default=8)
    parser.add_argument('--cnn-model', type=str, default='resnet50')
    parser.add_argument('--pretrained-cnn', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--vit-embed-dim', type=int, default=768)
    parser.add_argument('--vit-depth', type=int, default=4)
    parser.add_argument('--vit-heads', type=int, default=12)
    parser.add_argument('--gat-heads', type=int, default=4)
    
    args, _ = parser.parse_known_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train(args)