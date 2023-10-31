import csv
import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
from audio_datasets.constants import ENCODEC_REDUCTION_FACTOR, ENCODEC_SAMPLING_RATE

MAX_DURATION_IN_SECONDS = 20


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

def is_audio_length_in_range(audio, sampling_rate):
    return len(audio) <= (MAX_DURATION_IN_SECONDS*sampling_rate)

def main():
    max_length = compute_max_length()
    # Define the header names
    headers = ['file_name', 'text']
    data_dir = '/persist/data/mls/mls_english/'

    for split in ['train', 'dev', 'test']:
        print(f'Converting {split} split...')
        # Specify the input and output file paths
        split_dir = os.path.join(data_dir, split)
        input_file = os.path.join(split_dir, f'transcripts.txt')
        output_file = os.path.join(split_dir, f'metadata.csv')

        # Open the input file for reading
        with open(input_file, 'r') as file:
            # Create a CSV writer for the output file
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write the headers to the CSV file
                writer.writerow(headers)
                # Read each line in the input file
                for line in tqdm(file):
                    # Split the line into file path and description
                    audio_id, description = line.strip().split('\t')
                    speaker_id, book_id, file_id = audio_id.split('_')
                    file_path = os.path.join('audio', speaker_id, book_id, f'{audio_id}.flac')
                    audio, samplerate = sf.read(os.path.join(split_dir, file_path))
                    if is_audio_length_in_range(audio, samplerate):
                        # Write the file path and description as a row in the CSV file
                        writer.writerow([file_path, description])
                    


        print('Conversion complete!')


if __name__ == "__main__":
    main()