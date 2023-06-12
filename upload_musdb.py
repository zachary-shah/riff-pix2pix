"""
Script to upload preprocessed dataset to HuggingfaceHub for access during training.
"""


import argparse

from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value

from musdb_dataset import MusdbDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Musdb18 spectrograms dataset for InstructPix2Pix style training."
    )
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--num_samples_to_use", type=int, default=None)
    args = parser.parse_args()
    return args


def gen_examples(dataset):
    def fn():
        for sample in dataset:
            yield {
                "original_prompt": sample["original_prompt"],
                "original_image": {"path": str(sample["original_image"])},
                "edit_prompt": sample["edit_prompt"],
                "edited_prompt": sample["edited_prompt"],
                "edited_image": {"path": str(sample["edited_image"])},
            }
    return fn


def main(args):
    musdb_dataset = MusdbDataset(rootdir=args.data_root, num_samples_to_use=args.num_samples_to_use, return_paths=True)
    generator_fn = gen_examples(musdb_dataset)

    print("Creating dataset...")
    mini_ds = Dataset.from_generator(
        generator_fn,
        features=Features(
            original_prompt=Value("string"),
            original_image=ImageFeature(),
            edit_prompt=Value("string"),
            edited_prompt=Value("string"),
            edited_image=ImageFeature(),
        ),
    )

    print("Pushing to the Hub...")
    ds_name = f"musdb18-spec-pix2pix"
    if args.num_samples_to_use is not None:
        num_samples = args.num_samples_to_use
        ds_name += f"-{num_samples}-samples"
    mini_ds.push_to_hub(ds_name)

if __name__ == "__main__":
    args = parse_args()
    main(args)
