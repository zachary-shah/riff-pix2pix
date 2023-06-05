import wave
import pydub
import numpy as np
import librosa

import os
import sys

default_fs = 44100

# Notes on terminology:
# - clip: 5.12 second (or some fixed length) audio
# - stem: a specific musical component of a given clip
# - segment: synonymous with stem
def read_wav_file(file_path, fs):
    audio_data, sr = librosa.load(file_path, sr=fs, mono=True)

    # Split audio into 6 5.12 second clips (TODO: remove length hardcoding, make CL arg)
    num_samples_per_segment = int(5.110* fs)
    num_segments = 6
    hop_len = int(np.floor((len(audio_data) - num_samples_per_segment) / (num_segments - 1)))
    audio_segments = librosa.util.frame(audio_data, frame_length=num_samples_per_segment, hop_length=hop_len).T

    return audio_segments

"""
Given a segment of audio, return the frames for that stem with sufficient power indexed as a dictionary 
"""
def get_stem_frames(segment: pydub.AudioSegment, 
                    overlap: int = 0,
                    frame_seconds: int = 5.11,
                    min_power_prop: int = 0.5,
                    fs: int = 44100)-> dict:
    
    # get samples and time vec as numpy array
    samples = np.array(segment.get_array_of_samples())
    #samples = np.frombuffer(segment.raw_data, dtype=np.int16)

    # get power for samples 
    stem_power = np.mean(np.square(samples[(samples != 0)]))

    # get frames
    num_samples_per_segment = int(frame_seconds * fs)
    hop_length = int(num_samples_per_segment*(1-overlap))
    frames = librosa.util.frame(samples, frame_length=num_samples_per_segment, hop_length=hop_length).T

    # calculate frame powers
    frame_powers = []
    for i in range(frames.shape[0]):
        frame_powers.append(np.mean(np.square(frames[i])))

    # filter frames by ones with low power
    valid_frames = frames[frame_powers > min_power_prop*stem_power, :]
    valid_frame_inds = np.where(frame_powers > min_power_prop*stem_power)[0]

    # return frames as dict indexed by index
    valid_frames = {}
    for ind in valid_frame_inds:
        valid_frames[ind] = frames[ind,:]
    
    return valid_frames

def segment_audio(audio_data, fs=44100, num_segments=6, pitch_augment=True):
    # make audio data one channel
    audio_data = np.reshape(audio_data, (len(audio_data),))

    # Split audio into 5.12 second clips
    num_samples_per_segment = int(5.110* fs)
    hop_len = int(np.floor((len(audio_data) - num_samples_per_segment) / (num_segments - 1)))

    audio_segments = librosa.util.frame(audio_data, frame_length=num_samples_per_segment, hop_length=hop_len).T

    # modulate through 12 keys
    if pitch_augment:
        pitch_augmented_segs = []
        # pitch modulation for data augmentation
        for pitch_offset in [-3, -2, -1, 1, 2, 3]:
            for audio_segment in audio_segments:
                pitch_augmented_segs.append(librosa.effects.pitch_shift(np.squeeze(audio_segment), sr=fs, n_steps=pitch_offset))
        audio_segments = np.vstack([np.squeeze(audio_segments), np.stack(pitch_augmented_segs)])

    # TODO: consider time dilation with librosa.effects.time_stretch(audio_segment, rate=2.0)

    return audio_segments


def write_wav_file(audio_data, file_path, fs, verbose=False):
    with wave.open(file_path, 'w') as wave_file:
        num_channels = 1 # Mono audio
        sample_width = 2 # 16-bit audio
        num_frames = len(audio_data)
        comp_type = 'NONE'
        comp_name = 'not compressed'
        wave_params = (num_channels, sample_width, fs, num_frames, comp_type, comp_name)
        # Wave library expects a pretty specific set of info
        wave_file.setparams(wave_params)

        # Scale the audio data to fit within the range of a 16-bit integer
        # Without this, output sounds like garbage
        audio_data = audio_data / np.max(np.abs(audio_data))
        audio_data = np.int16(audio_data * 2**15-1)

        # Write the audio data to the file
        wave_file.writeframes(audio_data.tobytes())
    wave_file.close()
    
    if verbose:
        print(f"Saved audio file at {file_path} with sampling rate {fs} and length {len(audio_data) / fs:.2f} seconds.")

    
