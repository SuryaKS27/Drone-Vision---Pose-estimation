#
# utils.py
#
import torch
import torchvision
import numpy as np
from PIL import ImageDraw

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])

def save_validation_results(input_images, keypoints_xy, gaussian_maps, reconstructed_images, epoch, save_dir="val_results"):
    """
    Saves a grid of images:
    Row 1: Original Input
    Row 2: Input with Keypoints
    Row 3: Gaussian Map
    Row 4: Reconstructed Image
    """
    from pathlib import Path
    Path(save_dir).mkdir(exist_ok=True)
    
    # Take first 4 images from batch
    n_images = min(input_images.shape[0], 4)
    
    # Denormalize input images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(input_images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(input_images.device)
    input_images_denorm = (input_images * std + mean).clamp(0, 1)

    inputs = input_images_denorm[:n_images]
    recons = reconstructed_images[:n_images]
    gaussians = gaussian_maps[:n_images].sum(dim=1).unsqueeze(1).repeat(1, 3, 1, 1) # Sum keypoints, make 3-channel
    
    # Create image with keypoints
    keypoints_on_img = []
    for i in range(n_images):
        img_tensor = inputs[i]
        kps = keypoints_xy[i] # (K, 2)
        
        # Convert tensor to PIL Image
        img_pil = transforms.ToPILImage()(img_tensor)
        draw = ImageDraw.Draw(img_pil)
        
        # kps are in [0, 1] range, scale to image size
        img_w, img_h = img_pil.size
        for kp in kps:
            x = kp[0] * img_w
            y = kp[1] * img_h
            draw.ellipse((x-2, y-2, x+2, y+2), fill='red', outline='red')
        keypoints_on_img.append(transforms.ToTensor()(img_pil))
        
    keypoints_on_img = torch.stack(keypoints_on_img).to(inputs.device)

    # Make grid
    grid = torchvision.utils.make_grid(torch.cat([inputs, keypoints_on_img, gaussians, recons]), nrow=n_images)
    
    # Save
    torchvision.utils.save_image(grid, f"{save_dir}/epoch_{epoch:03d}.png")
