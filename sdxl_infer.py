import autoroot  # root setup, do not delete
import autorootcwd  # root setup, do not delete
import random
import argparse
import copy
import itertools
import logging
import math
import os
import gc
import shutil
import h5py
from pathlib import Path
import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms_v2
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
# from hdf5dataset import HDF5Dataset
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import load_image
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import glob

from peft import PeftModel, LoraConfig, get_peft_model
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json  # Added missing dependency

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
# from metrics.metrics import compute_metrics

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    # parser.add_argument(
    #     "--data_dir",
    #     type=str,
    #     default=os.environ["TRAIN_DIR"],
    #     required=True,
    #     help="A folder containing the training data of images.",
    # )
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/Reflection-Exploration/BrushNet/baseline/sd_inpainting/model_lora_abo",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--num_validation_images", 
        type=int, 
        default=4, 
        help="Number of validation images"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=27,
        help=("The alpha constant of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="The dropout rate of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="The bias type of the Lora update matrices. Must be 'none', 'all' or 'lora_only'.",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/data/MSD/test/images",
        help="The directory containing the input images."
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/data/MSD/test/masks",
        help="The directory containing the mask images."
    )
    parser.add_argument(
        "--prompt_csv",
        type=str,
        default="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/data/MSD/test/captions.csv",
        help="The CSV file containing the prompts for each image."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The directory where the generated images will be saved."
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def is_image(path):
    ext = os.path.splitext(path.lower())[-1]
    return ext == ".png" or ext == ".jpg"

def inference(args):
    pipeline, accelerator = create_pipeline(args)
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    img_files = sorted(filter(is_image, glob.glob("{}/*".format(args.img_dir))))
    mask_files = sorted(filter(is_image, glob.glob("{}/*".format(args.mask_dir))))
    prompt_list = pd.read_csv(args.prompt_csv)
    
    for img_path, mask_path in zip(img_files, mask_files):
        validation_image = load_image(img_path).resize((1024, 1024))
        validation_mask = load_image(mask_path).resize((1024, 1024))
        img_filename = os.path.basename(img_path).split('.')[0] + ".jpg"
        validation_prompt = prompt_list[prompt_list["images"] == img_filename]['Captions'].iloc[0]
        image = pipeline(
            prompt=validation_prompt,
            image=validation_image,
            mask_image=validation_mask,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=1,
            generator=generator,
        ).images[0]
        image.save(os.path.join(args.output_dir, f"{os.path.basename(img_path)}"))


def create_pipeline(args):

    accelerator = Accelerator()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_fast=False,
        )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet"
    )
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )
    unet = get_peft_model(unet, config)

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["k_proj", "q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )
    text_encoder = get_peft_model(text_encoder, config)
    vae.requires_grad_(False)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                sub_dir = "unet" if isinstance(model.base_model.model, type(accelerator.unwrap_model(unet).base_model.model)) else "text_encoder"
                model.save_pretrained(os.path.join(output_dir, sub_dir))
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()
            sub_dir = "unet" if isinstance(model.base_model.model, type(accelerator.unwrap_model(unet).base_model.model)) else "text_encoder"
            model_cls = UNet2DConditionModel if isinstance(model.base_model.model, type(accelerator.unwrap_model(unet).base_model.model)) else CLIPTextModel
            load_model = model_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder=sub_dir)
            load_model = PeftModel.from_pretrained(load_model, input_dir, subfolder=sub_dir)
            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # Prepare everything with our `accelerator`.
    unet, text_encoder = accelerator.prepare(unet, text_encoder)
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    # Move vae to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.model_path)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None

        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.model_path, path))
    else:
        pass

    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        safety_checker=None,
    )

    # set `keep_fp32_wrapper` to True because we do not want to remove
    # mixed precision hooks while we are still training
    pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
    pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, accelerator
    
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


if __name__ == "__main__":
    args = parse_args()
    inference(args)
