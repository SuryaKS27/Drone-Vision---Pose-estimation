#
# data.py
#
import torch
from torchvision import datasets, transforms
from PIL import Image

class CroppedObjectDataset(datasets.CocoDetection):
    """
    A COCO-style dataset that returns cropped images of objects
    based on their bounding box annotations.
    """
    def __init__(self, root, annFile, transform=None, padding_factor=0.1):
        super(CroppedObjectDataset, self).__init__(root, annFile)
        self.transform = transform
        self.padding_factor = padding_factor
        
        # We need to filter the image list to only include
        # images that HAVE annotations.
        self.ids = sorted(self.coco.getImgIds())
        valid_ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                valid_ids.append(img_id)
        self.ids = valid_ids

    def __getitem__(self, index):
        # Get image and annotation info
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        
        img = Image.open(f"{self.root}/{path}").convert('RGB')
        img_w, img_h = img.size

        # For this project, we only train on *one* object per image
        # for simplicity. You could expand this to return all objects.
        ann = anns[0] # Just take the first annotation
        
        # COCO [x, y, w, h] format
        x, y, w, h = ann['bbox']

        # Add padding to the bounding box
        pad_w = w * self.padding_factor
        pad_h = h * self.padding_factor
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        # Crop the image
        cropped_img = img.crop((x1, y1, x2, y2))
        
        # Apply standard transforms
        if self.transform is not None:
            cropped_img = self.transform(cropped_img)
            
        return cropped_img

def get_data_loader(img_dir, ann_file, img_size=128, batch_size=32, num_workers=4):
    """Helper function to create dataset and dataloader."""
    
    # These transforms are vital. All crops must be the same size.
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # You can add more augmentations here (e.g., RandomHorizontalFlip)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CroppedObjectDataset(
        root=img_dir,
        annFile=ann_file,
        transform=transform
    )
    
    # We must filter out None types in case some crops fail
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return data_loader
