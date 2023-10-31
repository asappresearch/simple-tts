# Simple-TTS

This repo contains the implementation for Simple-TTS, a latent diffusion model for text-to-speech generation. Our submission describing this work is currently under review:

**Simple-TTS: End-to-End Text-to-Speech Synthesis with Latent Diffusion**\
by Justin Lovelace, Soham Ray, Kwangyoun Kim, Kilian Q. Weinberger, and Felix Wu

## Environment
Install the required dependencies with:
```bash
pip install -r requirements.txt
```

## Datasets

We train our models using the English subset of the Multilingual LibriSpeech (MLS) dataset and use the standard LibriSpeech dataset for evaluation.

For the MLS dataset, download `mls_english.tar.gz` from [https://www.openslr.org/94/](https://www.openslr.org/94/). Store the unzipped dataset at `/persist/data/mls/mls_english/` or update the `data_dir` path in `audio_datasets/mls.py` accordingly. The MLS dataset can be processed by running the `audio_datasets/preprocess_mls.py` script.

We access LibriSpeech through the Huggingface Hub. For speaker-prompted generation, we utilize the first three seconds of another prompt. To extract the corresponding transcript from the first three seconds, we utilized the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/example.html#example-1-aligning-librispeech-english). An aligned version of LibriSpeech can be found at

```bash
data/aligned_librispeech.tar.gz
```

It should be untarred and you should update the update the `ALIGNED_DATA_DIR` path in `audio_datasets/librispeech.py` to point to the data directory. 

## Training

We trained all of the models using `bf16` mixed precision with Nvidia A10G GPUs. Things like batch size, etc. in the provided scripts should be adjusted depending on the hardware setup.

We provide a sample script to train the diffusion model with reasonable hyperparameters on a single Nvidia A10G GPU. Distributed training is recommended to increase the batch size, but this is useful for debugging.
```bash
./scripts/train/train.sh
``` 

We use [Huggingface Accelerate](https://huggingface.co/docs/accelerate/index) for distributed training and trained our final model on a `g5.48xlarge` instance (8 Nvidia A10Gs). After running `accelerate config` to set the appropriate environment variables (e.g. number of GPUs), a distributed training job with our hyperparameter settings can be launced with
```bash
./scripts/train/train_distributed.sh
``` 


## Model Checkpoint

Our model checkpoint can be downloaded from [here](https://simple-tts.awsdev.asapp.com/ckpt.tar.gz).

The checkpoint folders contain an `args.json` with the hyperparameter settings for the model as well as the checkpoint itself. The model was trained for 200k steps with a global batch size of 256. The model is likely undertrained and quality improvements could be gained from additional training. Using the `init_model` argument with the training scripts will initialize the model from the provided path. 

## Sampling
We provide a script for synthesizing speech for the Librispeech test-clean set:
```bash
./scripts/sample/sample_16_ls_testclean.sh
``` 
The `--resume_dir` argument should be updated with the path of a trained model. 

## Contact
Feel free to create an issue if have any questions. 


## Acknowledgement
This work built upon excellent open-source implementations from [Phil Wang (Lucidrains)](https://github.com/lucidrains). Specifically, we built off of his Pytorch [DDPM implementation](https://github.com/lucidrains/denoising-diffusion-pytorch).
