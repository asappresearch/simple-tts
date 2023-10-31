from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio, concatenate_datasets
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import os
import random
import math

from audio_datasets.text_grid_utils import get_partial_transcript

def lowercase_text(example):
    example["text"] = example["text"].lower()
    return example



MAX_DURATION_IN_SECONDS = 20
ENCODEC_SAMPLING_RATE = 24000
ENCODEC_REDUCTION_FACTOR = 320
LATENT_SAMPLING_RATE = 75
ALIGNED_DATA_DIR = '/persist/data/aligned_librispeech'

def round_up_to_multiple(number, multiple):
    remainder = number % multiple
    if remainder == 0:
        return number
    else:
        return number + (multiple - remainder)

def compute_max_length(multiple=128):
    max_len = MAX_DURATION_IN_SECONDS*ENCODEC_SAMPLING_RATE
    waveform_multiple = multiple*ENCODEC_REDUCTION_FACTOR

    max_len = round_up_to_multiple(max_len, waveform_multiple)
    return max_len

def is_audio_length_in_range(audio):
    return len(audio['array']) <= (MAX_DURATION_IN_SECONDS*ENCODEC_SAMPLING_RATE)


def is_audio_length_in_test_range(audio):
    return ((4*ENCODEC_SAMPLING_RATE) <= len(audio['array'])) and (len(audio['array']) <= (10*ENCODEC_SAMPLING_RATE))


class LibriSpeech(Dataset):
    """ 
    Wrapper around HuggingFace dataset for processing. 
    """
    def __init__(self, split='train', debug=False, tokenizer=None, max_seq_len=None, sampling_rate=None):
        super().__init__()
        self.sr = ENCODEC_SAMPLING_RATE if sampling_rate is None else sampling_rate
        self.split = split
        self.split2dir = {'valid': 'dev-clean', 'test': 'test-clean'}
        if split == 'train':
            train100 = load_dataset('librispeech_asr', 'clean', split='train.100')
            train360 = load_dataset('librispeech_asr', 'clean', split='train.360')

            self.hf_dataset = concatenate_datasets([train100, train360])
        elif split == 'valid':
            self.hf_dataset = load_dataset('librispeech_asr', 'clean', split='validation')
        elif split == 'test':
            self.hf_dataset = load_dataset('librispeech_asr', 'clean', split='test')
        else:
            raise ValueError(f"invalid split: {split}, must be in ['train', 'valid'] ")

        # Downsample to accelerate processing for debugging purposes
        if debug:
            self.hf_dataset = self.hf_dataset.select(range(100))
        # Resample to 24kHz for Encodec
        self.hf_dataset = self.hf_dataset.cast_column("audio", Audio(sampling_rate=self.sr))

        self.hf_dataset = self.hf_dataset.map(lowercase_text)
        if split == 'train':
            self.hf_dataset = self.hf_dataset.filter(is_audio_length_in_range, input_columns=['audio'])
        elif split == 'test':
            self.hf_dataset = self.hf_dataset.filter(is_audio_length_in_test_range, input_columns=['audio'])
        
        if self.split in {'valid', 'test'}:
            unique_speaker_ids = set(self.hf_dataset['speaker_id'])
            self.speaker_datasets = {speaker_id:self.hf_dataset.filter(lambda example: example["speaker_id"] == speaker_id) for speaker_id in unique_speaker_ids}
                

        self.max_seq_len = max_seq_len if max_seq_len is not None else compute_max_length()
        print(f'Max seq length: {self.max_seq_len/ENCODEC_REDUCTION_FACTOR}')

        if tokenizer is not None:
            self.hf_dataset = self.hf_dataset.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256), batched=True)
        self.tokenizer = tokenizer


    def __getitem__(self, index):
        example = self.hf_dataset[index]
        text = example['text']
        wav = example['audio']['array'][:self.max_seq_len]
        wavpath = example['audio']['path']
        npad = self.max_seq_len - len(wav)
        assert npad>=0, f'Waveform length {len(wav)} needs to be less than {self.max_seq_len}'
        # [1, L]: Channels x length
        audio_duration_sec = len(wav)/ENCODEC_SAMPLING_RATE
        wav = torch.tensor(np.pad(wav, pad_width=(0, npad), mode='constant'), dtype=torch.float).unsqueeze(0)

        data = {'wav': wav, 'text': text, 'audio_duration': audio_duration_sec, 'path':wavpath}


        # Speaker prompting
        if self.split in {'valid', 'test'}:
            split_dir = self.split2dir[self.split]
            speaker_id = example['speaker_id']
            speaker_ds = self.speaker_datasets[speaker_id]
            # Sample idx for n-1 elements and remap matching element to the last element
            speaker_idx = random.randint(0, len(speaker_ds)-2)
            if speaker_ds[speaker_idx]['id'] == example['id']:
                speaker_idx = len(speaker_ds)-1
            speaker_example = speaker_ds[speaker_idx]
            textgrid_path = os.path.join(ALIGNED_DATA_DIR, split_dir, f'{speaker_id}', f'{speaker_example["id"]}.TextGrid')
            partial_transcript = get_partial_transcript(textgrid_path)

            speaker_text = partial_transcript['transcript']
            speaker_wav_frames = math.ceil(partial_transcript['end_time'] * ENCODEC_SAMPLING_RATE)
            speaker_audio_duration_sec = speaker_wav_frames/ENCODEC_SAMPLING_RATE
            speaker_wav = speaker_example['audio']['array'][:speaker_wav_frames]
            speaker_npad = self.max_seq_len - len(speaker_wav)
            assert speaker_npad>=0, f'Waveform length {len(speaker_wav)} needs to be less than {self.max_seq_len}'
            # [1, L]: Channels x length
            
            speaker_wav = torch.tensor(np.pad(speaker_wav, pad_width=(0, speaker_npad), mode='constant'), dtype=torch.float).unsqueeze(0)

            speaker_data = {'speaker_wav': speaker_wav, 'speaker_text': speaker_text, 'speaker_audio_duration': speaker_audio_duration_sec}
            data.update(speaker_data)


        if self.tokenizer is not None:
            data['input_ids'] = torch.tensor(example['input_ids'], dtype=torch.long)
            data['attention_mask'] = torch.tensor(example['attention_mask'], dtype=torch.long)

        data['audio_mask'] = torch.ones((self.max_seq_len//ENCODEC_REDUCTION_FACTOR,), dtype=torch.bool)
        
        return data

    def __len__(self):
        return len(self.hf_dataset)
    
if __name__ == "__main__":
    dataset = LibriSpeech(split='test')

    example = dataset.__getitem__(0)
    import soundfile as sf
    import pdb; pdb.set_trace()
    sf.write(f'example_audio/librispeech_sample.wav', example['wav'], ENCODEC_SAMPLING_RATE)
    with open(f'example_audio/librispeech_text.txt', 'w') as f:
        print(example['text'], file=f)
