from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset, Audio, concatenate_datasets
from datasets import Dataset as HFDataset
import numpy as np
import os
import random
import csv

import torch
from tqdm import tqdm
import torch.nn.functional as F

from transformers import AutoTokenizer

from audio_datasets.constants import ENCODEC_REDUCTION_FACTOR, ENCODEC_SAMPLING_RATE

def read_csv_into_dict(filename):
    data_dir = os.path.dirname(filename)
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        data = {'audio': [], 'text':[]}

        for row in reader:
            for key, value in row.items():
                if key == 'file_name':
                    value = os.path.join(data_dir, value)
                    data['audio'].append(value)
                    continue
                elif key == 'text':
                    data[key].append(value)
                else:
                    raise ValueError(f'Unexpected csv key: {key}')

    return data

MAX_DURATION_IN_SECONDS = 20

def round_up_to_multiple(number, multiple):
    remainder = number % multiple
    if remainder == 0:
        return number
    else:
        return number + (multiple - remainder)
    
def round_up_to_waveform_multiple(number, multiple=16):
    waveform_multiple = multiple*ENCODEC_REDUCTION_FACTOR
    rounded_number = round_up_to_multiple(number, waveform_multiple)
    return rounded_number

def compute_max_length(multiple=16):
    max_len = MAX_DURATION_IN_SECONDS*ENCODEC_SAMPLING_RATE

    max_len = round_up_to_waveform_multiple(max_len, multiple)
    return max_len

def is_audio_length_in_range(audio):
    return len(audio['array']) < (MAX_DURATION_IN_SECONDS*ENCODEC_SAMPLING_RATE)


class MLS(TorchDataset):
    """ 
    Wrapper around HuggingFace dataset for processing. 
    """
    def __init__(self, data_dir='/persist/data/mls/mls_english/' , split='train', debug=False, tokenizer=None, sampling_rate=None, max_text_len=256):
        super().__init__()
        self.sr = ENCODEC_SAMPLING_RATE if sampling_rate is None else sampling_rate
        print('Loading audio dataset...')
        if split == 'train':
            data_dict = read_csv_into_dict(os.path.join(data_dir, 'train', 'metadata.csv'))
            
            self.hf_dataset = HFDataset.from_dict(data_dict).cast_column("audio", Audio(sampling_rate=self.sr))
        elif split == 'valid':
            self.hf_dataset = load_dataset("audiofolder", data_dir=os.path.join(data_dir, 'dev'))['train'].cast_column("audio", Audio(sampling_rate=self.sr))
        else:
            raise ValueError(f"invalid split: {split}, must be in ['train', 'valid'] ")

        # Downsample to accelerate processing for debugging purposes
        if debug:
            self.hf_dataset = self.hf_dataset.select(range(100))
        # Resample to 24kHz for Encodec

        self.max_text_len = max_text_len

        self.max_seq_len = compute_max_length()
        print(f'Max seq length: {self.max_seq_len/ENCODEC_REDUCTION_FACTOR}')

        self.tokenizer = tokenizer


    def __getitem__(self, index):
        example = self.hf_dataset[index]
        text = example['text']
        wav = example['audio']['array']
        npad = self.max_seq_len - len(wav)
        assert npad>=0, f'Waveform length {len(wav)} needs to be less than {self.max_seq_len}'
        # [1, L]: Channels x length
        wav_len = len(wav)
        audio_duration_sec = len(wav)/self.sr
        wav = torch.tensor(np.pad(wav, pad_width=(0, npad), mode='constant'), dtype=torch.float).unsqueeze(0)
        
        silence_tokens = round(random.random()*(self.max_seq_len-wav_len))
        num_unmasked_tokens = round_up_to_waveform_multiple(wav_len+silence_tokens)//ENCODEC_REDUCTION_FACTOR
        audio_mask = torch.zeros((self.max_seq_len//ENCODEC_REDUCTION_FACTOR,), dtype=torch.bool)
        audio_mask[:num_unmasked_tokens] = True

        if self.tokenizer is not None:
            tokenized_text = self.tokenizer(example['text'], padding="max_length", truncation=True, max_length=self.max_text_len)
            input_ids = torch.tensor(tokenized_text['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(tokenized_text['attention_mask'], dtype=torch.long)
        else:
            input_ids = None
            attention_mask = None
            return {'wav': wav, 'text': text, 'path':example['audio']['path'], 'audio_duration': audio_duration_sec, 'wav_len':wav_len }
        

        return {'wav': wav, 'text': text, 'input_ids': input_ids, 'attention_mask': attention_mask, 'audio_duration': audio_duration_sec, 'wav_len':wav_len, 'audio_mask':audio_mask}

    def __len__(self):
        return len(self.hf_dataset)
    
if __name__ == "__main__":
    text_tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
    train_ds = MLS(split='train', tokenizer=text_tokenizer)
    import pdb; pdb.set_trace()
    val_ds = MLS(split='valid')
    
    example = train_ds.__getitem__(0)
    import soundfile as sf
    sf.write(f'example_audio/mls_sample.wav', example['wav'].squeeze().numpy(), ENCODEC_SAMPLING_RATE)
    with open(f'example_audio/mls_text.txt', 'w') as f:
        print(example['text'], file=f)
