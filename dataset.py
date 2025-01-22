from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms.v2 as transforms_v2
import numpy as np
import pandas as pd
import torch
import random
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision.transforms.functional import crop


class InpaintingDataset(Dataset):
    """
    A dataset to prepare the training and conditioning images and
    the masks with the dummy prompt for fine-tuning the model.
    It pre-processes the images, masks and tokenizes the prompts.
    """

    def __init__(
        self,
        train_data_root,
        tokenizer,
        size=512,
        resolution=1024,
        validation=False,
    ):
        self.size = size
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.ref_data_root = Path(train_data_root)
        self.train_images_dir = Path(train_data_root) / "images"
        self.train_masks_dir = Path(train_data_root) / "masks"
        caption_path = Path(train_data_root) / "captions.csv"
        if not (self.ref_data_root.exists()):
            raise ValueError("Train images root doesn't exist.")
        self.validation = validation
        
        # sort images and masks while keeping the target image and mask at the end
        train_images_list = sorted(list(self.train_images_dir.iterdir()))
        train_masks_list = sorted(list(self.train_masks_dir.iterdir()))
        self.train_images_path = list(train_images_list) 
        self.train_masks_path = list(train_masks_list)
        self.num_train_images = len(self.train_images_path)

        # self.train_prompt = "a photo of sks" # default prompt from realfill
        self.prompt_list = pd.read_csv(caption_path)
        self.train_crop = transforms_v2.RandomCrop(size)
        self.transform = transforms_v2.Compose(
            [
                transforms_v2.RandomResize(size, int(1.125 * size)),
                transforms_v2.PILToTensor(),
                transforms_v2.ToDtype(torch.float32)
            ]
        )

        self.normalize = transforms_v2.Normalize([0.5], [0.5])  # normalize only the image, weighting separately

    def __len__(self):
        return self.num_train_images
    
    def random_crop(self, img, output_size):
        img_width, img_height = img.size
        crop_height, crop_width = output_size

        top = random.randint(0, img_height - crop_height)
        left = random.randint(0, img_width - crop_width)

        cropped_image = img.crop(
            (left, top, left + crop_width, top + crop_height)
        )
        return cropped_image, (top, left)

    def resize_if_smaller(self, image):
        # Check if either dimension of the image is smaller than the desired resolution
        if image.width < self.resolution or image.height < self.resolution:
            scaling_factor = max(self.resolution / image.width, self.resolution / image.height)
            new_width = int(image.width * scaling_factor)
            new_height = int(image.height * scaling_factor)
            image = image.resize((new_width, new_height), Image.BILINEAR)
        return image

    def __getitem__(self, index):
        if self.validation:
            return self.validation_samples(index)
        example = {}

        image = Image.open(self.train_images_path[index])
        image = exif_transpose(image)

        mask = Image.open(self.train_masks_path[index])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        if mask.mode == "RGB": # to ensure single channel mask
            mask = mask.convert("L")

        if index < len(self) - 1:
            weighting = Image.new("L", image.size)
        else:
            weighting = Image.new("L", image.size)
            # weighting = Image.open(self.target_mask).convert("L")
            weighting = exif_transpose(weighting)
        
        example["original_size"] = (image.height, image.width)
        image, weighting, mask = self.resize_if_smaller(image), self.resize_if_smaller(weighting), self.resize_if_smaller(mask)        
        y1, x1, h, w = self.train_crop.get_params(image, (self.size, self.size))
        image, weighting, mask = self.transform(image, weighting, mask)
        image, mask = crop(image, y1, x1, h, w), crop(mask, y1, x1, h, w)
        image, weighting = self.normalize(image, weighting)
        example["images"], example["weightings"] = image, weighting < 0
        example["crop_top_left"] = (y1, x1)
        
        if random.random() < 0.1:
            example["masks"] = torch.ones_like(example["images"][0:1, :, :])
        else:
            # use saved masks instead of generating using the make_mask function. Transform that as well
            # example["masks"] = make_mask(example["images"], self.size) # default used by realfill
            example["masks"] = mask

        # Eq.3 (1-m) * x
        example["conditioning_images"] = example["images"] * (example["masks"] < 0.5)
        image_name = str(self.train_images_path[index]).split('/')[-1]
        if image_name in self.prompt_list["images"]:
            train_prompt = self.prompt_list[self.prompt_list["images"]==image_name]["Captions"][0]
        else:
            train_prompt = "A plane reflective mirror"
        
        example["prompt"] = train_prompt
        example["prompt_ids"] = self.tokenizer(
            train_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example

    def validation_samples(self, index):
            example = {}
            image = Image.open(self.train_images_path[index])
            image = exif_transpose(image)
            mask = Image.open(self.train_masks_path[index])
            example["masks"], example["images"], example["weightings"] = mask, image, None
            image_name = str(self.train_images_path[index]).split('/')[-1]
            if image_name in self.prompt_list["images"]:
                train_prompt = self.prompt_list[self.prompt_list["images"]==image_name]["Captions"][0]
            else:
                train_prompt = "A plane reflective mirror"
            example["prompt"] = train_prompt
            example["prompt_ids"], example["conditioning_images"] = None, None
            return example


def collate_fn(examples):
    input_ids = [example["prompt_ids"] for example in examples]
    images = [example["images"] for example in examples]

    masks = [example["masks"] for example in examples]
    weightings = [example["weightings"] for example in examples]
    conditioning_images = [example["conditioning_images"] for example in examples]
    original_size = [example["original_size"] for example in examples]
    crop_top_left = [example["crop_top_left"] for example in examples]
    prompt = [example["prompt"] for example in examples]


    images = torch.stack(images)
    images = images.to(memory_format=torch.contiguous_format).float()

    masks = torch.stack(masks)
    masks = masks.to(memory_format=torch.contiguous_format).float()

    # weightings = torch.stack(weightings)
    # weightings = weightings.to(memory_format=torch.contiguous_format).float()

    conditioning_images = torch.stack(conditioning_images)
    conditioning_images = conditioning_images.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "images": images,
        "masks": masks,
        "weightings": None,
        "conditioning_images": conditioning_images,
        "original_size": original_size,
        "crop_top_left": crop_top_left,
        "prompt": prompt
    }
    return batch
