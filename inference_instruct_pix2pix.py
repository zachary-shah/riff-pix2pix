import PIL
import requests
import torch
import os, argparse

from diffusers import StableDiffusionInstructPix2PixPipeline
from datasets import load_dataset
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for pretrained InstructPix2Pix.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="zachary-shah/riff-pix2pix-v2",
        help="Path to pretrained model from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default='zachary-shah/musdb18-spec-pix2pix-test',
        help="Path to training dataset to conduct inference on.",
    )
    parser.add_argument(
        "--base_save_path",
        type=str,
        default="/data/pix2pix-inference",
        help="Output directory for audio and image samples.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=25,
        help="Max number of test examples to sample on",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps during sampling",
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=int,
        default=1.5,
        help="image guidance",
    )
    parser.add_argument(
        "--guidance_scale",
        type=int,
        default=10,
        help="text guidance",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        default=False,
        help="True to seed random processed",
    )

    args = parser.parse_args()

    return args

args = parse_args()

# add seed 
if args.seed: torch.manual_seed(364)

# set up model and test data
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)
test_dataset = load_dataset(args.dataset_id)["train"]
test_dataset = torch.utils.data.random_split(test_dataset, [args.max_samples, len(test_dataset)-args.max_samples])[0]
img_converter_to_audio = SpectrogramImageConverter(SpectrogramParams(sample_rate=44100, min_frequency=0, max_frequency=10000))
os.makedirs(args.base_save_path, exist_ok=True)

def save_img_and_audio(img, filename):
    # save image
    img.save(os.path.join(args.base_save_path, filename +".png"))
    # reconstruct and save audio
    out_audio_recon = img_converter_to_audio.audio_from_spectrogram_image(img, apply_filters=True).set_channels(2)
    out_audio_recon.export(os.path.join(args.base_save_path,filename + ".wav"), format="wav") 

print(f"Beginning inference for {len(test_dataset)} samples.")

for (i, item) in enumerate(test_dataset):
    prompt = item["edited_prompt"]
    print(f"Sampling {i+1}/{len(test_dataset)}: prompt=\"{prompt}\"")
    
    # get sample
    edited_image_sample = pipe(
        item["edited_prompt"],
        image=item["original_image"],
        num_inference_steps=args.num_inference_steps,
        image_guidance_scale=args.image_guidance_scale,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    # save images
    base_name = f"s_{i+1}_" + prompt.replace(" ", "_").replace(",", "").replace(".","")
    save_img_and_audio(edited_image_sample, base_name + "_edit_sample")
    save_img_and_audio(item["original_image"], base_name + "_original")
    save_img_and_audio(item["edited_image"], base_name + "_edit_target")

print("Inference complete!")