 # Training Guide
 
 This guide lays out the various parameters in the codebase related to training and some commentary on best practices. We also list default values from the codebase.

 ## Standard Training Hyperparameters

 | Argument | Default | 
|-|-|
| --optimizer | 'adamw' |
| --batch_size | 16 |
| --num_train_steps | 60000 |
| --gradient_accumulation_steps | 1 |
| --learning_rate | 1e-4 |
| --clip_grad_norm | 1.0 |
| --lr_schedule | 'cosine' |
| --lr_warmup_steps | 1000 |
| --adam_beta1 | 0.9 |
| --adam_beta2 | 0.999 |
| --adam_weight_decay | 0 |
| --dropout | 0.0 |  |
| --clip_grad_norm | 1.0 |
| --mixed_precision | 'no' | Accelerate argument |

### Commentary

The training setup is pretty standard. We use the AdamW optimizer with a cosine learning rate schedule with a short linear warmup. Diffusion models are quite robust to overfitting so we don't employ any explicit regularization (weight decay or dropout). Regularization may be helpful if training the model for much longer. We clip gradients to a norm of 1.0 which improves training stability.

For training the model, longer training times will be better. Our released checkpoint was trained for 200k steps with a global batch size of 256 (per-device batch size of 16, 2 gradient accumulation steps, and 8 GPUs). It is likely signficantly under-trained and further training would be beneficial. Large batch sizes tend to be beneficial for diffusion models due to the stochastisty of the training objective so distributed training and/or gradient accumulation are recommended. 

We trained our models with `bf16` mixed precision.

 ## Architecture Hyperparameters

| Argument | Default |
|-|-|
| --dim | 512 |
| --dim_mults | (1, 1, 1, 1.5) |
| --conformer_transformer | False |
| --scale_skip_connection | False |
| --num_transformer_layers | 3 |

### Commentary
The diffusion model consists of a U-Net and transformer. The first half of the U-Net downsamples the input to produce low-resolution features that are processed by the transformer. The output of the transformer is then upsampled by the second half of the U-Net to the original resolution to generate the final prediction.

The structure of the U-Net model is determined by the `dim` and `dim_mults` arguments. The `dim` argument controls the initial dimensionality of the model. The `dim_mults` argument controls the number of downsampling layers in the U-Net and the feature dimensionality at each level. The feature dimensionality is defined as a multiple of the original `dim` value. 

Therefore, our U-Net model has 4 layers and the dimensionality in the middle of the network is `768=512*1.5`. The final dimensionality of the U-Net model is also the dimensionality of the transformer model. The `num_transformer_layers` and the final dimensionality of the U-Net control the size of the transformer. The transformer model contains the cross-attention layers and is therefore primarily responsible for the text-audio alignment. Our released model has a 768d transformer with 12 layers.

To scale up the model, past work on image generation has shown that it's sufficient to scale the middle of the network, leaving the downsampling/upsampling layers unchanged. Therefore, scaling up the transformer dimensionality and depth is likely the most effective way to scale up the network. 

For diffusion models, it's important for the model dimensionality to be meaningfully larger than the input data. Given that the dimensionality of the EnCodec features is `128`, using `dim=512` for the initial dimensionality is a reasonable choice. I would be cautious about decreasing the input dimensionality. 

It's been shown for text-to-image diffusion models that scaling the U-Net skip connections by a constant factor significantly accelerates convergence. We used this trick (controlled by the `--scale_skip_connection` flag) when training our model, but did not investigate it's impact in detail. 

We also included an option to introduce a conformer-style convolution layer into the transformer (the `--conformer_transformer` flag), but did not end up using it in our primary model. Its use didn't seem to make a significant difference from our preliminary investigation, but we didn't explore it in great detail. The additional feedfoward block does meaningfully increase the size of the transformer so a fair comparison would need to control for that.

## Speaker-Prompted Generation Arguments

| Argument | Default |
|-|-|
| --inpainting_embedding | False |
| --inpainting_prob | 0.5 |  |

We train our model for both zero-shot TTS (i.e. generating speech given only the transcript) and speaker-prompted TTS (i.e. generating speech in the style of some speaker) in a mult-task manner. We train the model for speaker-prompted TTS by only adding noise to the latter portion of the audio latent, providing the model with a clean speech prompt. The `--inpainting_prob` flag controls the portion of instances used in the speaker-prompted setting. The `--inpainting_embedding` introduces binary embeddings which are added to the input to specify the prompt speech frames. We recommend enabling this flag.
 
## Diffusion Hyperparameters

| Argument | Default | 
|-|-|
| --objective | 'pred_v' |
| --parameterization | 'pred_v' |
| --loss_type | 'l1' |
| --scale | 1.0 |
| --unconditional_prob | 0.1 |

Diffusion models consist of a denoising network that accepts some noisy data as input and attempts to recover the original data. In practice, this network can be parameterized in a variety of different ways. We parameterize our denoising network as a velocity prediction (or v-prediction) network. See [[1]](https://openreview.net/forum?id=TIdIXIpzhoI) for a discussion of the various parameterizations. The v-parameterization has been pretty widely adopted (e.g. [[2]](https://huggingface.co/stabilityai/stable-diffusion-2-1)[[3]](https://arxiv.org/abs/2301.11093)[[4]](https://arxiv.org/abs/2204.03458)) and is therefore a reasonable choice. One nuance is that you can treat the output of the network (`--parameterization`) and the loss function (`--objective`) as two separate design decisions (also discussed in [[1]](https://openreview.net/forum?id=TIdIXIpzhoI)). However, setting both to pred_v is a reasonable default choice.

The noise schedule is one of the most critical hyperparameters for the quality of the generations. We use the widespread cosine noise schedule and find adjusting the scale factor of the noise schedule to be important for assuring text-speech alignment. See [[5]](https://arxiv.org/abs/2301.10972) for a detailed discussion of this choice. We set the `--scale` flag to `0.5` for our final checkpoint. 

Diffusion models are trained with a regression loss. Both the `l1` and `l2` loss are commonly used in the literature. There's some work [[6]](https://arxiv.org/abs/2111.05826) suggesting that the `l1` loss leads to more conservative generations while the `l2` loss leads to more diverse generations, potentially at the cost of quality. We utilized the `l1` loss to emphasize quality over diversity, but the `l2` may be adviseable depending on the application.

To enable the use of classifier-free guidance [[7]](https://arxiv.org/abs/2207.12598), we drop the conditioning information with some probability. The probability is controlled by the `--unconditional_prob` flag. It should generally be in the `0.1-0.2` range and we use `0.1` by default.

## Validation Parameters

| Argument | Default | 
|-|-|
| --save_and_sample_every | 5000 |
| --sampling_timesteps | 250 | 
| --num_samples | None |
| --sampler | 'ddim' | 

We generate some samples every `--save_and_sample_every` steps from the validation set. The `--num_samples` flag controls how many validation samples to generate and the `--sampling_timesteps` controls the number of timesteps used for generation. The `--num_samples` flag should be set reasonably low (e.g. 128) to avoid spending too much time on validation. We discuss sampling in more detail in the other README.