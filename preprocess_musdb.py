"""
Preprocess Musdb for instruct pix2pix training

Requires Musdb to be downloaded, and manual prompts for each song to be written and stored in a json file with format:
{"file": <filepath_to_song_example>, "edit_prompt": <edit_prompt_for_song>}
"""

import os, argparse, json
import pydub
import numpy as np
from time import time
import torch 
from librosa.effects import pitch_shift

from utils.riffusion_utils import audio_array_to_image
from utils.audio_utils import write_wav_file, get_stem_frames

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_data_dir",
    type=str,
    nargs="?",
    default='../musdb18-wav/train',
    help="directory path to musdb dataset to preprocess"
)
parser.add_argument(
    "--output_dir",
    type=str,
    nargs="?",
    default='./musdb-pix2pix/train',
    help="where to save data to"
)
parser.add_argument(
    "--gen_stem",
    type=str,
    nargs="?",
    default="vocals",
    help="stem to generate. must be one of [vocals, bass, drums, other]."
)
parser.add_argument(
    "--prompt_filepath",
    type=str,
    nargs="?",
    default='./manual_prompts/musdb_vocal_edit_instructions.json',
    help="json file with prompts for all the songs"
)
parser.add_argument(
    "--fs",
    type=int, 
    nargs="?",
    default=44100,
    help="sampling rate"
) 
parser.add_argument(
    "--frame_overlap",
    type=float,
    nargs="?",
    default=0.,
    help="overlap in generated frames, in percent."
) 
parser.add_argument(
    "--frame_len_seconds",
    type=float,
    nargs="?",
    default=5.11,
    help="frame length, in seconds"
) 
parser.add_argument(
    "--frame_min_power_prop",
    type=float,
    nargs="?",
    default=0.4,
    help="minimum power of stem in a frame for power threshold filtering"
) 
parser.add_argument(
    "--device",
    type=str,
    nargs="?",
    default="cpu",
    help="device: either cuda or cpu"
)
parser.add_argument(
    "--max_examples",
    type=int,
    nargs="?",
    default=np.infty,
    help="Max number of examples to create in total"
)
parser.add_argument(
    "--silence",
    default=False,
    action="store_true",
    help="true to silence training detail outputs"
)
parser.add_argument(
    "--save_wav",
    default=False,
    action="store_true",
    help="select to save .wav files for all examples for debugging"
)
parser.add_argument(
    "--pitch_augment",
    default=False,
    action="store_true",
    help="True to augment pitch of each example"
)

opt = parser.parse_args()

def make_train_example(gen_wav, input_wav, edit_prompt, song_no, frame_no, ex_no, opt):

    # path naming 
    example_name = f"s{song_no}_f{frame_no}_e{ex_no}.jpg"
    input_path = os.path.join(opt.output_dir, "input_image", example_name)
    edited_path = os.path.join(opt.output_dir, "edited_image", example_name)

    # make input spectrogram
    audio_array_to_image(input_wav, 
                        save_img=True,
                        outpath=input_path[:-4],
                        sample_rate=opt.fs,
                        device=opt.device)
    
    # make target as combo of source and generated stems
    edited_wav = gen_wav + input_wav

    # make edited spectrogram
    audio_array_to_image(edited_wav, 
                        save_img=True,
                        outpath=edited_path[:-4],
                        sample_rate=opt.fs,
                        device=opt.device,
                        image_extension="jpg")
    
    # add to metadata
    with open(os.path.join(opt.output_dir,"metadata.jsonl"), 'a') as outfile:
        packet = {
            "input_file_path": str(os.path.join("input_image", example_name)),
            "edited_file_path": str(os.path.join("edited_image", example_name)),
            "edit_prompt": str(edit_prompt)
        }
        json.dump(packet, outfile)
        outfile.write('\n')
    outfile.close()

    # optionally save audio for debugging
    if opt.save_wav:
        write_wav_file(input_wav, os.path.join(opt.output_dir, 'wav', example_name[:-4]+"_input"+".wav"), fs=opt.fs)
        write_wav_file(edited_wav, os.path.join(opt.output_dir, 'wav', example_name[:-4]+"_edited"+".wav"), fs=opt.fs)

    ex_no += 1
    return ex_no

def get_audio_seg(filepath):
    seg = pydub.AudioSegment.from_file(filepath)
    if seg.channels != 1:
        seg = seg.set_channels(1)
    if seg.frame_rate != opt.fs:
        seg = seg.set_frame_rate(opt.fs)
    assert seg.channels == 1
    assert seg.frame_rate == opt.fs
    return seg
         
# tracking
num_examples_total = 0
time_start = time()

# cuda if possible
if torch.cuda.is_available():
    opt.device = "cuda" 

if not opt.silence:
    print("Beginning preprocessing!")

## INPUT VALIDATION 
STEMS = np.array(["vocals", "bass", "drums", "other"])
assert opt.gen_stem in STEMS, f"Input stem must be one of {STEMS}"
bgnd_stems = STEMS[np.array([opt.gen_stem != stem for stem in STEMS])]

# check for prompt path
assert os.path.exists(opt.prompt_filepath), "Edit prompts filepath invalid."

# get all prompts in prompt_file as dictionary
prompt_dict = {}
p_count = 0
with open(opt.prompt_filepath, 'r') as prompt_file:
    for line in prompt_file:
        data = json.loads(line)
        prompt_dict[data['file']] = data['edit_prompt']
        p_count += 1
if not opt.silence: print(f"Read {p_count} edit prompts from {opt.prompt_filepath}.")

# make all directories needed
os.makedirs(opt.output_dir, exist_ok=True)
os.makedirs(os.path.join(opt.output_dir, 'input_image'), exist_ok=True)
os.makedirs(os.path.join(opt.output_dir, 'edited_image'), exist_ok=True)
if opt.save_wav: os.makedirs(os.path.join(opt.output_dir, 'wav'), exist_ok=True)

# data folder
example_dirs = sorted([f for f in os.listdir(opt.root_data_dir) if f != '.DS_Store'])

for song_no, example_dir in enumerate(example_dirs):

    if not opt.silence: print(f"Processing song {song_no+1}/{len(example_dirs)}: {example_dir}...")

    # get song prompt
    try:
        prompt = prompt_dict[example_dir]
    except:
        print(f" -- Warning: could not find prompt for {example_dir}. Using default prompt instead.")
        prompt = f"Generate a {opt.gen_stem} stem."

    if not opt.silence: print(f"  Prompt for song: \"{prompt}\"")

    # get audios for given file. make sure they are 1 channel audio
    gen_seg = get_audio_seg(os.path.join(opt.root_data_dir, example_dir, opt.gen_stem+".wav"))
    
    bgnd_seg = get_audio_seg(os.path.join(opt.root_data_dir, example_dir, bgnd_stems[0]+".wav"))
    for i in range(1, len(bgnd_stems)):
        bgnd_seg = bgnd_seg.overlay(get_audio_seg(os.path.join(opt.root_data_dir, example_dir, bgnd_stems[i]+".wav")))

    # make audio frames
    gen_frames = get_stem_frames(gen_seg, 
                            overlap = opt.frame_overlap,
                            frame_seconds = opt.frame_len_seconds,
                            min_power_prop = opt.frame_min_power_prop,
                            fs = opt.fs)

    bgnd_frames = get_stem_frames(bgnd_seg, 
                            overlap = opt.frame_overlap,
                            frame_seconds = opt.frame_len_seconds,
                            min_power_prop = -1,
                            fs = opt.fs)

    valid_frames = np.array(list(gen_frames.keys()))

    if not opt.silence:
        print(f"  FRAMING: ")
        print(f"    Number of valid frames in {opt.gen_stem}: {len(gen_frames)}")
        print(f"    Number of total frames in bgnd: {len(bgnd_frames)}")

    for fno, f in enumerate(valid_frames):
        if not opt.silence: print(f"   Frame {fno+1}/{len(valid_frames)}")
        ex_no = 0

        try:
            ex_no = make_train_example(gen_frames[f], bgnd_frames[f], prompt, song_no, fno, ex_no, opt)

            # modulate through 12 keys
            if opt.pitch_augment:
                if not opt.silence: print("    -- Pitch agumenting frame!")
                # pitch modulation for data augmentation
                for pitch_offset in [-3, -2, -1, 1, 2, 3]:
                    ex_no = make_train_example(pitch_shift(np.squeeze(gen_frames[f].astype('float')), sr=opt.fs, n_steps=pitch_offset),
                                            pitch_shift(np.squeeze(bgnd_frames[f].astype('float')), sr=opt.fs, n_steps=pitch_offset),
                                            prompt,
                                            song_no, 
                                            fno, 
                                            ex_no, 
                                            opt)

            num_examples_total += ex_no
            if num_examples_total >= opt.max_examples: break
        except:
            print(f"      -- WARNING: error making example for frame {f} -- ")

    if num_examples_total >= opt.max_examples:
        print(f"Max example count reached, terminating processing.") 
        break

# script information
time_elapsed = (time() - time_start) / 60
print(f"""Preprocessing complete! Summary:
      - preprocessed {song_no+1} songs
      - generated {num_examples_total} examples total
      - runtime: {time_elapsed} min""")