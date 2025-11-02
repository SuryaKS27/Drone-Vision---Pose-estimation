#
# train.py
#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path

# Import from our other project files
from model import PKA_UNet
from data import get_data_loader
from utils import save_checkpoint, save_validation_results

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    # Create separate loaders for train and val
    train_loader = get_data_loader(
        img_dir=args.train_img_dir,
        ann_file=args.train_ann_file,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    val_loader = get_data_loader(
        img_dir=args.val_img_dir,
        ann_file=args.val_ann_file,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print("Data loaded.")
    
    # --- 2. Initialize Model, Loss, Optimizer ---
    model = PKA_UNet(
        n_channels_in=3,
        n_channels_out=3,
        num_keypoints=args.num_keypoints,
        img_size=args.img_size
    ).to(device)

    reconstruction_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # --- 3. Training Loop ---
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for batch_idx, cropped_images in enumerate(loop):
            cropped_images = cropped_images.to(device)
            
            # --- Forward pass ---
            reconstructed_image, _, _ = model(cropped_images)
            
            # --- Loss calculation ---
            loss = reconstruction_loss_fn(reconstructed_image, cropped_images)
            
            # --- Backward pass ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Avg. Train Loss: {avg_train_loss:.4f}")

        # --- 4. Validation & Visualization Loop ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, cropped_images in enumerate(val_loader):
                cropped_images = cropped_images.to(device)
                reconstructed_image, keypoints_xy, gaussian_maps = model(cropped_images)
                
                loss = reconstruction_loss_fn(reconstructed_image, cropped_images)
                val_loss += loss.item()
                
                # Save visualization for the first batch of validation
                if batch_idx == 0:
                    save_validation_results(
                        cropped_images, 
                        keypoints_xy, 
                        gaussian_maps, 
                        reconstructed_image, 
                        epoch=epoch+1,
                        save_dir=args.val_results_dir
                    )

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Avg. Val Loss: {avg_val_loss:.4f}")

        # --- 5. Save Checkpoint ---
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        save_checkpoint(checkpoint, filename=f"{args.checkpoint_dir}/epoch_{epoch+1}.pth.tar")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised Keypoint Autoencoder Training")
    
    # --- Data Paths ---
    parser.add_argument("--train_img_dir", type=str, required=True, help="Path to training images")
    parser.add_argument("--train_ann_file", type=str, required=True, help="Path to training COCO annotation file")
    parser.add_argument("--val_img_dir", type=str, required=True, help="Path to validation images")
    parser.add_argument("--val_ann_file", type=str, required=True, help="Path to validation COCO annotation file")
    
    # --- Model Hyperparameters ---
    parser.add_argument("--img_size", type=int, default=128, help="Image size to resize all crops to")
    parser.add_argument("--num_keypoints", type=int, default=10, help="Number of keypoints to discover")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # --- Logging/Saving ---
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--val_results_dir", type=str, default="val_results", help="Directory to save validation images")

    args = parser.parse_args()
    main(args)
