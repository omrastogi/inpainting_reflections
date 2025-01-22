# import autoroot  # root setup, do not delete
# import autorootcwd  # root setup, do not delete
import random
import argparse
import copy
import itertools
import logging
import math
import os
import gc
import shutil
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
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig
from torchvision.transforms.functional import crop
from peft.utils import get_peft_model_state_dict
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import (
    check_min_version,
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.loaders import LoraLoaderMixin
from peft import LoraConfig, set_peft_model_state_dict
from dataset import InpaintingDataset, collate_fn


from peft import PeftModel, LoraConfig, get_peft_model
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Extract WANDB_API_KEY from environment variables
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.1")

logger = get_logger(__name__)

def make_mask(images, resolution, times=30):
    mask, times = torch.ones_like(images[0:1, :, :]), np.random.randint(1, times)
    min_size, max_size, margin = np.array([0.03, 0.25, 0.01]) * resolution
    max_size = min(max_size, resolution - margin * 2)

    for _ in range(times):
        width = np.random.randint(int(min_size), int(max_size))
        height = np.random.randint(int(min_size), int(max_size))

        x_start = np.random.randint(int(margin), resolution - int(margin) - width + 1)
        y_start = np.random.randint(int(margin), resolution - int(margin) - height + 1)
        mask[:, y_start:y_start + height, x_start:x_start + width] = 0

    mask = 1 - mask if random.random() < 0.5 else mask
    return mask

def apply_mask(img: Image, mask: Image):
    """returns a new image with the inpainting white region as per the mask. (255) values in mask are inpainted."""
    # Ensure both images have the same size
    if img.size != mask.size:
        raise ValueError("Image and mask must have the same size")

    if mask.mode != "RGB":
        mask = mask.convert("RGB")

    img = np.array(img)
    mask = np.array(mask)

    final_image = np.where(mask == 255, 255, img)

    return Image.fromarray(final_image)


def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    repo_folder=None,
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
prompt: "a photo of sks"
tags:
- stable-diffusion-inpainting
- stable-diffusion-inpainting-diffusers
- text-to-image
- diffusers
- realfill
inference: true
---
    """
    model_card = f"""
# RealFill - {repo_id}

This is a realfill model derived from {base_model}. The weights were trained using [RealFill](https://realfill.github.io/).
You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

def validation_collate_fn(batch):
    # Return the batch as is (bypassing the default collation)
    return batch[0]

def log_validation(
    text_encoder,
    tokenizer,
    unet,
    args,
    accelerator,
    weight_dtype,
    global_step,
):
    logger.info(
        f"Running validation... \nGenerating {args.num_validation_images} images"
    )

    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        revision=args.revision,
    )
    
    val_dataset = InpaintingDataset(
        train_data_root=args.val_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        validation=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=validation_collate_fn
    )


    # set `keep_fp32_wrapper` to True because we do not want to remove
    # mixed precision hooks while we are still training
    pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
    pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []
    for batch in tqdm(val_dataloader, desc="Validation Steps"):
        images = []
        for _ in range(args.num_validation_images):
            validation_prompt = batch["prompt"]
            # validation_image = Image.fromarray((batch["images"].squeeze(0).permute(1, 2, 0).byte().cpu().numpy()))
            # validation_mask = Image.fromarray(batch["masks"].squeeze(0).squeeze(0).byte().cpu().numpy())
            validation_image = batch["images"]
            validation_mask = batch["masks"]
            image = pipeline(
                prompt=validation_prompt,
                image=validation_image,
                mask_image=validation_mask,
                num_inference_steps=10,
                guidance_scale=1,
                generator=generator,
            ).images[0]
            images.append(image)
        image_logs.append(
            {
                "validation_image": validation_image, 
                "pred_images": images, 
                "validation_prompt": validation_prompt
            }
        )

    
    tracker_key = "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["pred_images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            columns = ["Masked Image"] + [f"Generation {i+1}" for i in range(args.num_validation_images)]
            table = wandb.Table(columns=columns)
            for log in image_logs:
                row_images = []
                pred_images = log["pred_images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                row_images.append(wandb.Image(validation_image, caption="BrushNet conditioning"))
                for image in pred_images:
                    image = wandb.Image(image, caption=validation_prompt)
                    row_images.append(image)
                table.add_data(*row_images)

            tracker.log({tracker_key: table}, step=global_step)
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return images

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="A folder containing the training images.",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        required=True,
        help="A folder containing the validation images.",
    )

    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_conditioning`.",
    )
    
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run realfill validation every X steps. RealFill validation consists of running the conditioning"
            " `args.validation_conditioning` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--unet_learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate to use for unet.",
    )
    parser.add_argument(
        "--text_encoder_learning_rate",
        type=float,
        default=4e-5,
        help="Learning rate to use for text encoder.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default=os.environ['WANDB_API_KEY'],
        help=("If report to option is set to wandb, api-key for wandb used for login to wandb "),
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=os.environ['WANDB_PROJECT_NAME'],
        help=("If report to option is set to wandb, project name in wandb for log tracking  "),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=4, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
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
        "--variant", 
        type=str, 
        default=None, 
        help="Model variant"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

        wandb.login(key=args.wandb_key, relogin=True, force=True)
        wandb.init(project=args.wandb_project_name)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=True, cache_dir="./cache")
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, low_cpu_mem_usage=True)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, low_cpu_mem_usage=True)

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

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformerspip 
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    # SDXL Utility
    def compute_time_ids(original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids
    
    # SDXL Utility
    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds
    

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

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


    # def load_model_hook(models, input_dir):
    #     unet_ = None
    #     text_encoder_one_ = None
    #     text_encoder_two_ = None

    #     while len(models) > 0:
    #         model = models.pop()

    #         if isinstance(model, type(unwrap_model(unet))):
    #             unet_ = model
    #         elif isinstance(model, type(unwrap_model(text_encoder_one))):
    #             text_encoder_one_ = model
    #         elif isinstance(model, type(unwrap_model(text_encoder_two))):
    #             text_encoder_two_ = model
    #         else:
    #             raise ValueError(f"unexpected save model: {model.__class__}")

    #     lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

    #     unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
    #     unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
    #     incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
    #     if incompatible_keys is not None:
    #         # check only for unexpected keys
    #         unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
    #         if unexpected_keys:
    #             logger.warning(
    #                 f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
    #                 f" {unexpected_keys}. "
    #             )

    #     # if args.train_text_encoder:
    #     #     # Do we need to call `scale_lora_layers()` here?
    #     #     _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

    #     #     _set_state_dict_into_text_encoder(
    #     #         lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_
    #     #     )

    #     # Make sure the trainable params are in float32. This is again needed since the base models
    #     # are in `weight_dtype`. More details:
    #     # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
    #     if args.mixed_precision == "fp16":
    #         models = [unet_]
    #         # if args.train_text_encoder:
    #         #     models.extend([text_encoder_one_, text_encoder_two_])
    #         #     # only upcast trainable parameters (LoRA) into fp32
    #         #     cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.unet_learning_rate = (
            args.unet_learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

        args.text_encoder_learning_rate = (
            args.text_encoder_learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    optimizer = optimizer_class(
        [
            {"params": unet.parameters(), "lr": args.unet_learning_rate},
            {"params": text_encoder.parameters(), "lr": args.text_encoder_learning_rate}
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = InpaintingDataset(
        train_data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        prefetch_factor=2,
    )



    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, text_encoder, optimizer, train_dataloader = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers("realfill", config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # SDXL Tokenizers -----
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder"
        )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
        )


    text_encoder_one = text_encoder_cls_one.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    text_encoder_two = text_encoder_cls_two.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
        )
    
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]
    # -----------------
    
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Convert masked images to latent space
                conditionings = vae.encode(batch["conditioning_images"].to(dtype=weight_dtype)).latent_dist.sample()
                conditionings = conditionings * 0.18215

                # Downsample mask and weighting so that they match with the latents
                masks, size = batch["masks"].to(dtype=weight_dtype), latents.shape[2:]
                masks = F.interpolate(masks, size=size)

                # weightings = batch["weightings"].to(dtype=weight_dtype)
                # weightings = F.interpolate(weightings, size=size)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Concatenate noisy latents, masks and conditionings to get inputs to unet
                inputs = torch.cat([noisy_latents, masks, conditionings], dim=1)

                # Get the text embedding for conditioning
                # encoder_hidden_states = text_encoder(batch["input_ids"])[0]


                '''
                The dreambooth code has - 
                https://github.com/huggingface/diffusers/blob/f29b93488dee9af000fab6e7bdb68ab565d50564/examples/dreambooth/train_dreambooth_lora_sdxl.py#L1439

                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids
                '''
                add_time_ids = torch.cat(
                    [
                        compute_time_ids(original_size=s, crops_coords_top_left=c)
                        for s, c in zip(batch["original_size"], batch["crop_top_left"])
                    ]
                )

                # add_time_ids = compute_time_ids(batch["original_size"], batch["crop_top_left"])
                prompt_embeds, unet_add_text_embeds = compute_text_embeddings(batch["prompt"], text_encoders, tokenizers)
                unet_added_conditions = {
                        "time_ids": add_time_ids, #[1, 6] 
                        "text_embeds": unet_add_text_embeds, # [1, 1280]
                    }
                # RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x1376 and 2816x1280)
                # Predict the noise residual
                # change the source of - encoder_hidden_states and input
                """
                I am getting:
                unet_added_conditions["text_embeds"].shape = [256, 1280] which was expected to be [16, 1280]
                encoder_hidden_states.shape = [16,77,768] whih is expected to be [B, 77, 2048]
                input.shape = [16, 9, 64, 64] which is expected to be [B, 4, 128, 128]
                here inchannel 9 is fine as Unet is set for that 
                """
                # model_pred = unet(
                #     inputs, # [B, 4, 128, 128]
                #     timesteps, # tensor([633], device='cuda:0') shape [B]
                #     encoder_hidden_states, # [B, 77, 2048]
                #     added_cond_kwargs=unet_added_conditions,
                #     ).sample
                
                model_pred = unet(
                    inputs, # [B, 4, 128, 128]
                    timesteps, # tensor([633], device='cuda:0') shape [B]
                    prompt_embeds, # [B, 77, 2048]
                    added_cond_kwargs=unet_added_conditions,
                    ).sample


                # Compute the diffusion loss
                assert noise_scheduler.config.prediction_type == "epsilon"
                loss = (F.mse_loss(model_pred.float(), noise.float(), reduction="none")).mean()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(
                        unet.parameters(), text_encoder.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if args.report_to == "wandb":
                    accelerator.print(progress_bar)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        log_validation(
                            text_encoder,
                            tokenizer,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr":lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet, keep_fp32_wrapper=True).merge_and_unload(),
            text_encoder=accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True).merge_and_unload(),
            revision=args.revision,
        )

        pipeline.save_pretrained(args.output_dir)

        # # Final inference
        # images = data/MSD/test/masks/22_512x640.png data/MSD/test/masks/66_512x640.png data/MSD/test/masks/37_512x640.png(
        #     text_encoder,
        #     tokenizer,
        #     unet,
        #     args,
        #     accelerator,
        #     weight_dtype,
        #     global_step,
        # )

        # if args.push_to_hub:
        #     save_model_card(
        #         repo_id,
        #         images=images,
        #         base_model=args.pretrained_model_name_or_path,
        #         repo_folder=args.output_dir,
        #     )
        #     upload_folder(
        #         repo_id=repo_id,
        #         folder_path=args.output_dir,
        #         commit_message="End of training",
        #         ignore_patterns=["step_*", "epoch_*"],
        #     )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    # prepare_target_image(args)
    main(args)
