import PIL
import requests
import torch
import os, argparse

from diffusers import StableDiffusionInstructPix2PixPipeline
from datasets import load_dataset
from tqdm import tqdm
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
        default="/data/pix2pix-inference/guidance_search",
        help="Output directory for audio and image samples.",
    )
    parser.add_argument(
        "--sample_indx",
        type=int,
        default=0,
        help="Index in test dataset to conduct exploration on.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="Number of inference steps during sampling",
    )
    args = parser.parse_args()

    return args

args = parse_args()


torch.manual_seed(364)

# set up model and test data
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)
test_dataset = load_dataset(args.dataset_id)["train"]
test_dataset = torch.utils.data.random_split(test_dataset, [25, len(test_dataset)-25])[0]
img_converter_to_audio = SpectrogramImageConverter(SpectrogramParams(sample_rate=44100, min_frequency=0, max_frequency=10000))
os.makedirs(args.base_save_path, exist_ok=True)

def save_img_and_audio(img, filename):
    # save image
    img.save(os.path.join(args.base_save_path, filename +".png"))
    # reconstruct and save audio
    out_audio_recon = img_converter_to_audio.audio_from_spectrogram_image(img, apply_filters=True).set_channels(2)
    out_audio_recon.export(os.path.join(args.base_save_path,filename + ".wav"), format="wav") 

# just get first idem
item = test_dataset[args.sample_indx]
prompt = item["edited_prompt"]
original_img = item["original_image"]
base_name = prompt.replace(" ", "_").replace(",", "").replace(".","")

# save orig and edit for reference
save_img_and_audio(item["original_image"], "original")
save_img_and_audio(item["edited_image"], "target_" + base_name)

print(f"Beginning inference for sample prompt: \"{prompt}\"")

image_guidance_scales = [1.5, 2, 3, 5]
text_guidance_scales = [5, 10, 15]

for image_guidance_scale in image_guidance_scales:
    for text_guidance_scale in text_guidance_scales:

        # seed process each time
        torch.manual_seed(364)

        # get sample
        edited_image_sample = pipe(
            prompt,
            original_img,
            num_inference_steps=args.num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=text_guidance_scale,
            generator=generator,
        ).images[0]

        # save sample
        save_img_and_audio(edited_image_sample, f"sT={text_guidance_scale}_sI={image_guidance_scale}")

print("Inference complete!")