import math
from pathlib import Path
import random 
from functools import partial
from collections import namedtuple, Counter
from multiprocessing import cpu_count
import os
import numpy as np
import csv
import timeit
import json
import argparse
from collections import defaultdict
from contextlib import nullcontext
import soundfile as sf


import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce, repeat

from tqdm.auto import tqdm
from ema_pytorch import EMA

from transformers import get_scheduler, AutoTokenizer, T5ForConditionalGeneration

from accelerate import Accelerator, DistributedDataParallelKwargs

from diffusion.optimizer import get_adamw_optimizer
from utils.utils import compute_grad_norm
import utils.utils as file_utils
from diffusion.noise_schedule import *
from audio_datasets import LibriSpeech, ENCODEC_SAMPLING_RATE, LATENT_SAMPLING_RATE, MLS
from neural_codec.encodec_wrapper import EncodecWrapper
from utils.utils import get_output_dir


ModelPrediction =  namedtuple('ModelPrediction', ['pred_eps', 'pred_x_start', 'pred_v', 'latents'])

# Recommendation from https://arxiv.org/abs/2303.09556
MIN_SNR_GAMMA = 5

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def l2norm(t):
    return F.normalize(t, dim = -1)

# Avoid log(0)
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim)
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        max_seq_len,
        sampling_timesteps = 250,
        text_encoder = 'google/byt5-small',
        loss_type = 'l1',
        objective = 'pred_v',
        parameterization = 'pred_v',
        train_schedule = 'cosine',
        ema_decay = 0.9999,
        sampling_schedule = None,
        scale = 1.,
        scale_by_std=True,
        sampler = 'ddim',
        unconditional_prob = 0.2,
        inpainting_prob = 0.5,
        inpainting_duration_mode = 0.01,
        inpainting_duration_concentration = 5, 
    ):
        super().__init__()

        self.denoising_network = EMA(model, beta = ema_decay, update_every = 1, power=3/4)

        self.max_seq_len = max_seq_len

        self.objective = objective
        self.parameterization = parameterization
        self.sampler=sampler

        # Min-SNR weighting from https://arxiv.org/abs/2303.09556
        # Option no longer supported; buffer kept for backwards compatibility with statedict of prior checkpoints
        self.register_buffer('min_snr_gamma', torch.tensor(MIN_SNR_GAMMA))

        self.loss_type = loss_type

        assert objective in {'pred_eps', 'pred_x0', 'pred_v'}, f'objective {objective} must be one of pred_eps, pred_x0, pred_v'

        if train_schedule == "simple_linear":
            alpha_schedule = simple_linear_schedule
        elif train_schedule == "beta_linear":
            alpha_schedule = beta_linear_schedule
        elif train_schedule == "cosine":
            alpha_schedule = cosine_schedule
        elif train_schedule == "sigmoid":
            alpha_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid noise schedule {train_schedule}')
        
        self.train_schedule = partial(time_to_alpha, alpha_schedule=alpha_schedule, scale=scale)

        # Sampling schedule
        if sampling_schedule is None:
            sampling_alpha_schedule = None
        elif sampling_schedule == "simple_linear":
            sampling_alpha_schedule = simple_linear_schedule
        elif sampling_schedule == "beta_linear":
            sampling_alpha_schedule = beta_linear_schedule
        elif sampling_schedule == "cosine":
            sampling_alpha_schedule = cosine_schedule
        elif sampling_schedule == "sigmoid":
            sampling_alpha_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid sampling schedule {sampling_schedule}')
        
        if exists(sampling_alpha_schedule):
            self.sampling_schedule = partial(time_to_alpha, alpha_schedule=sampling_alpha_schedule, scale=scale)
        else:
            self.sampling_schedule = self.train_schedule


        # Optionally rescale data to have unit variance
        self.scale_by_std = scale_by_std
        if scale_by_std:
            self.register_buffer('std_scale_factor', torch.tensor(-1.0))
        else:
            self.std_scale_factor = 1.0

        # gamma schedules

        self.sampling_timesteps = sampling_timesteps

        # probability for self conditioning during training

        self.unconditional_prob = unconditional_prob
        self.inpainting_prob = inpainting_prob

        if self.unconditional_prob > 0:
            self.unconditional_bernoulli = torch.distributions.Bernoulli(probs=self.unconditional_prob)
        if self.inpainting_prob > 0:
            self.inpainting_bernoulli = torch.distributions.Bernoulli(probs=self.inpainting_prob)
            # Mode/Concentration parameterization of Beta distribution
            alpha = inpainting_duration_mode*(inpainting_duration_concentration-2) + 1
            beta = (1-inpainting_duration_mode)*(inpainting_duration_concentration-2)+1
            self.inpainting_duration_beta = torch.distributions.Beta(alpha, beta)

        self.text_encoder_id = text_encoder
        self.text_encoder = T5ForConditionalGeneration.from_pretrained(text_encoder, torch_dtype=torch.bfloat16).get_encoder()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder)

        self.audio_codec = EncodecWrapper()
        for param in self.audio_codec.parameters():
            param.requires_grad = False


    def predict_start_from_noise(self, z_t, t, noise, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        return (z_t - (1-alpha).sqrt() * noise) / alpha.sqrt().clamp(min = 1e-8)
        
    def predict_noise_from_start(self, z_t, t, x0, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        return (z_t - alpha.sqrt() * x0) / (1-alpha).sqrt().clamp(min = 1e-8)

    def predict_start_from_v(self, z_t, t, v, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        x = alpha.sqrt() * z_t - (1-alpha).sqrt() * v

        return x
    
    def predict_noise_from_v(self, z_t, t, v, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        eps = (1-alpha).sqrt() * z_t + alpha.sqrt() * v

        return eps
    
    def predict_v_from_start_and_eps(self, z_t, t, x, noise, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        v = alpha.sqrt() * noise - x* (1-alpha).sqrt()

        return v

    def diffusion_model_predictions(self, z_t, t, *, text_cond, text_cond_mask, sampling=False, cls_free_guidance=1.0, fill_mask=None, audio_mask=None):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        time_cond = time_to_alpha(t).sqrt()
        latents = None
        inpainting_mask = fill_mask[:, 0, :].long()
        if sampling:
            model_output = self.denoising_network.ema_model(z_t, time_cond, text_cond=text_cond, text_cond_mask=text_cond_mask, inpainting_mask=inpainting_mask, audio_mask=audio_mask)
            if cls_free_guidance != 1.0:
                unc_text_cond = torch.zeros_like(text_cond)[:,:1,:]
                unc_text_cond_mask = torch.full_like(text_cond_mask, fill_value=False)[:,:1]                
                if exists(fill_mask):
                    alpha = rearrange(time_to_alpha(t), 'b -> b () ()')
                    noise = torch.randn_like(z_t)
                    z_t[fill_mask] = (z_t*alpha.sqrt() + (1-alpha).sqrt()*noise)[fill_mask] 
                unc_inpainting_mask = torch.full_like(inpainting_mask, fill_value=0)
                unc_model_output = self.denoising_network.ema_model(z_t, time_cond, text_cond=unc_text_cond, text_cond_mask=unc_text_cond_mask, inpainting_mask=unc_inpainting_mask, audio_mask=audio_mask)
                model_output = model_output*cls_free_guidance + unc_model_output*(1-cls_free_guidance)
        else:
            model_output = self.denoising_network.online_model(z_t, time_cond, text_cond=text_cond, text_cond_mask=text_cond_mask, inpainting_mask=inpainting_mask, audio_mask=audio_mask)

        pred_v = None
        if self.parameterization == 'pred_eps':
            pred_eps = model_output
            x_start = self.predict_start_from_noise(z_t, t, pred_eps, sampling=sampling)
        elif self.parameterization =='pred_x0':
            x_start = model_output
            pred_eps = self.predict_noise_from_start(z_t, t, x_start, sampling=sampling)
            pred_v = self.predict_v_from_start_and_eps(z_t, t, x_start, pred_eps, sampling=sampling)
        elif self.parameterization == 'pred_v':
            pred_v = model_output
            x_start = self.predict_start_from_v(z_t, t, pred_v, sampling=sampling)
            pred_eps = self.predict_noise_from_v(z_t, t, pred_v, sampling=sampling)
        else:
            raise ValueError(f'invalid objective {self.parameterization}')

        return ModelPrediction(pred_eps, x_start, pred_v, latents)

    def get_sampling_timesteps(self, batch, *, device, start_time=1.0):
        times = torch.linspace(start_time, 0., self.sampling_timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times
        

    @torch.no_grad()
    def ddim_or_ddpm_sample(self, shape, text_cond, text_cond_mask, prefix_seconds=0, audio_latent=None, cls_free_guidance=1.0, speaker_frames=None, sampler='ddim'):
        batch, device = shape[0], next(self.denoising_network.ema_model.parameters()).device

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        z_t = torch.randn(shape, device=device)

        fill_mask = None

        if prefix_seconds > 0:
            assert exists(audio_latent)
            if exists(speaker_frames):
                num_inpainting_frames = speaker_frames
            else:
                num_inpainting_frames = round(prefix_seconds*LATENT_SAMPLING_RATE)
                torch.full((batch), fill_value=num_inpainting_frames, dtype=torch.int, device=device)

            indices = torch.arange(z_t.shape[2], device=device)

            # Construct mask to insert clean data
            fill_mask = repeat((indices <= num_inpainting_frames[:, None]), 'b l -> b c l', c=z_t.shape[1])
        else:
            fill_mask = torch.full_like(z_t, fill_value=0, dtype=torch.bool)
            
        x_start = None
        latents = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.sampling_timesteps):
            # get predicted x0
            if prefix_seconds > 0:
                z_t[fill_mask] = audio_latent[fill_mask]
                if exists(x_start):
                    x_start[fill_mask] = audio_latent[fill_mask]

            model_output = self.diffusion_model_predictions(z_t, time, text_cond=text_cond, text_cond_mask=text_cond_mask, sampling=True, cls_free_guidance=cls_free_guidance, fill_mask=fill_mask)

            # get alpha sigma of time and next time

            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(partial(right_pad_dims_to, z_t), (alpha, alpha_next))

            # calculate x0 and noise

            x_start = model_output.pred_x_start

            eps = model_output.pred_eps

            if time_next[0] <= 0:
                z_t = x_start
                continue

            # get noise
            if sampler == 'ddim':
                z_t = x_start * alpha_next.sqrt() + eps * (1-alpha_next).sqrt()
            elif sampler == 'ddpm':
                # get noise
                noise = torch.randn_like(z_t)
                alpha_now = alpha/alpha_next

                min_var = torch.exp(torch.log1p(-alpha_next) - torch.log1p(-alpha)) * (1.0 -alpha_now)
                max_var = (1.0 - alpha_now)
                noise_param = 0.2
                sigma = torch.exp(noise_param * torch.log(max_var) + (1 - noise_param) * torch.log(min_var) )
                z_t = 1/alpha_now.sqrt() * (z_t - (1-alpha_now)/(1-alpha).sqrt() * eps) + torch.sqrt(sigma) * noise
        if prefix_seconds > 0:
            z_t[fill_mask] = audio_latent[fill_mask]
        return z_t

    @torch.no_grad()
    def sample(self, data, prefix_seconds=0, cls_free_guidance=1.0):
        # [B, L, d_lm]: Embedded text
        if prefix_seconds > 0:
            merged_text = [' '.join((speaker_text, text)) for speaker_text, text in zip(data['speaker_text'], data['text'])]
            tokenizer_output = self.text_tokenizer(merged_text, padding="max_length", truncation=True, max_length=256, return_tensors='pt').to(data['input_ids'].device)
            text_cond = self.text_encoder(tokenizer_output['input_ids'], tokenizer_output['attention_mask']).last_hidden_state.float()
            text_cond_mask = tokenizer_output['attention_mask'].bool()
            audio_latent = self.audio_codec.encode(data['speaker_wav'])
            speaker_frames = torch.floor(data['speaker_audio_duration'] * LATENT_SAMPLING_RATE).int()
        else:
            text_cond = self.text_encoder(data['input_ids'], data['attention_mask']).last_hidden_state.float()
            text_cond_mask = data['attention_mask'].bool()
            # [B, d_audio, L]
            audio_latent = self.audio_codec.encode(data['wav'])
            speaker_frames = None

        audio_latent *= self.std_scale_factor
        latent_shape = audio_latent.shape
        assert self.sampler in {'ddim', 'ddpm'}
        sample_fn = partial(self.ddim_or_ddpm_sample, sampler=self.sampler)
        return sample_fn(latent_shape, text_cond, text_cond_mask, prefix_seconds=prefix_seconds, audio_latent=audio_latent, cls_free_guidance=cls_free_guidance, speaker_frames=speaker_frames) / self.std_scale_factor

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
        
    def inpainting_enabled(self):
        return self.inpainting_prob > 0
    
    def forward(self, data, accelerator=None):
        
        with torch.no_grad():
            # [B, L, d_lm]: Embedded text
            text_cond = self.text_encoder(data['input_ids'], data['attention_mask']).last_hidden_state.float()
            # [B, L]: Cross-attn mask
            text_cond_mask = data['attention_mask'].bool()
            
                
                # [B, d_audio, L]: embedded audio
            with torch.cuda.amp.autocast(enabled=False):
                audio_latent = self.audio_codec.encode(data['wav'])

                if self.scale_by_std:
                    # Estimate standard deviation of the data from the first batch
                    if self.std_scale_factor < 0:
                        del self.std_scale_factor
                        gathered_audio_latent = accelerator.gather(audio_latent)
                        self.register_buffer('std_scale_factor', 1. / gathered_audio_latent.flatten().std())
                        print(f'Setting std scale factor: {self.std_scale_factor.item()}')

                    audio_latent *= self.std_scale_factor

                batch, audio_channels, audio_length = audio_latent.shape
                device = audio_latent.device

                # Mask out text-conditioning with some probability to enable clf-free guidance
                if self.unconditional_prob > 0:
                    unconditional_mask = self.unconditional_bernoulli.sample((batch,)).bool()
                    text_cond_mask[unconditional_mask, :] = False
                
                # sample random times

                times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)
                    
                # noise sample

                noise = torch.randn_like(audio_latent)

                alpha = self.train_schedule(times)
                alpha = right_pad_dims_to(audio_latent, alpha)

                z_t = alpha.sqrt() * audio_latent + (1-alpha).sqrt() * noise

                # Inpainting logic
                inpainting_mask = None
                if self.inpainting_prob > 0:
                    inpainting_batch_mask = self.inpainting_bernoulli.sample((batch,)).bool().to(device)
                    # Sample durations to mask
                    inpainting_durations = self.inpainting_duration_beta.sample((batch,)).to(device) * data['audio_duration']
                    num_inpainting_frames = torch.round(inpainting_durations*LATENT_SAMPLING_RATE).int()

                    # Sample where to mask
                    indices = torch.arange(audio_length, device=device)

                    # Construct mask to insert clean data
                    inpainting_length_mask = ((indices <= num_inpainting_frames[:, None]))
                    inpainting_mask = (inpainting_length_mask) & inpainting_batch_mask.unsqueeze(-1)
                    fill_mask = repeat(inpainting_mask, 'b l -> b c l', c=audio_channels)

                    z_t[fill_mask] = audio_latent[fill_mask]
                else:
                    fill_mask = torch.full_like(z_t, fill_value=0, dtype=torch.bool)

                velocity = alpha.sqrt() * noise - (1-alpha).sqrt() * audio_latent
        
        # predict and take gradient step
        predictions = self.diffusion_model_predictions(z_t, times, text_cond=text_cond, text_cond_mask=text_cond_mask, fill_mask=fill_mask, audio_mask=data['audio_mask'])

        
        if self.objective == 'pred_x0':
            target = audio_latent
            pred = predictions.pred_x_start
        elif self.objective == 'pred_eps':
            target = noise
            pred = predictions.pred_eps
        elif self.objective == 'pred_v':
            # V-prediction from https://openreview.net/forum?id=TIdIXIpzhoI
            target = velocity
            assert exists(predictions.pred_v)
            pred = predictions.pred_v
        else:
            raise NotImplementedError
        
        loss = self.loss_fn(pred, target, reduction = 'none')
        if self.inpainting_prob > 0:
            loss = reduce(loss, 'b c l -> b l', 'mean')
            # Standard diffusion loss
            diff_batch = loss[~inpainting_batch_mask]
            diff_loss = masked_mean(diff_batch, dim=1, mask=data['audio_mask'][~inpainting_batch_mask])

            # Masked inpainting loss
            inpainting_batch = loss[inpainting_batch_mask]
            loss_mask = torch.logical_and((~inpainting_length_mask[inpainting_batch_mask]), data['audio_mask'][inpainting_batch_mask])
            inpainting_loss = masked_mean(inpainting_batch, dim=1, mask=loss_mask)
            loss = torch.cat([diff_loss, inpainting_loss], dim=0)
        else:
            loss = reduce(loss, 'b c l -> b l', 'mean')
            loss = masked_mean(inpainting_batch, dim=1, mask=data['audio_mask'])

        return loss.mean()
    
# trainer class

class Trainer(object):
    def __init__(
        self,
        args,
        diffusion,
        dataset_name,
        *,
        batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        lr_schedule = 'cosine',
        num_warmup_steps = 500,
        adam_betas = (0.9, 0.999),
        adam_weight_decay = 0.01,
        save_and_sample_every = 5000,
        num_samples = 25,
        mixed_precision = 'no',
        prefix_inpainting_seconds=0,
        seed=None
    ):
        super().__init__()

        assert prefix_inpainting_seconds in {0., 3.0}, 'Currently only supports 3sec for inpainting'
        if exists(seed):
            set_seeds(seed)

        self.args = args

        self.accelerator = Accelerator(
            mixed_precision = mixed_precision,
            log_with=['mlflow'],
        )
        self.num_devices = self.accelerator.num_processes
        args.num_devices = self.num_devices

        args.output_dir = get_output_dir(args)

        if self.accelerator.is_main_process:     
            os.makedirs(args.output_dir)
            print(f'Created {args.output_dir}')

            with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            run = os.path.split(__file__)[-1].split(".")[0] 
            if self.num_devices > 1:
                run += '_multi'
            else:
                run += '_debug'
            self.accelerator.init_trackers(run, config=vars(args), init_kwargs={"mlflow": {"logging_dir": args.output_dir, "run_name": args.run_name}})

    
        self.diffusion = diffusion

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.prefix_inpainting_seconds = prefix_inpainting_seconds

        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.max_seq_len = diffusion.max_seq_len


        # dataset and dataloader
        if dataset_name == 'librispeech':
            self.dataset = LibriSpeech(split='train', tokenizer=diffusion.text_tokenizer)
            self.val_dataset = LibriSpeech(split='valid', tokenizer=diffusion.text_tokenizer)
            self.test_dataset = LibriSpeech(split='test', tokenizer=diffusion.text_tokenizer, max_seq_len=self.dataset.max_seq_len)
        elif dataset_name == 'mls':
            self.dataset = MLS(split='train', tokenizer=diffusion.text_tokenizer, max_text_len=256 if 'byt5' in diffusion.text_encoder_id else 128)
            self.val_dataset = LibriSpeech(split='valid', tokenizer=diffusion.text_tokenizer, max_seq_len=self.dataset.max_seq_len)
            self.test_dataset = LibriSpeech(split='test', tokenizer=diffusion.text_tokenizer, max_seq_len=self.dataset.max_seq_len)
        else:
            raise ValueError(f'invalid dataset: {dataset_name}')

        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=2)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        
        # optimizer

        self.opt = get_adamw_optimizer(diffusion, lr = train_lr, betas = adam_betas, weight_decay=adam_weight_decay)

        # scheduler

        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps*self.num_devices,
            num_training_steps=train_num_steps*self.num_devices, # Accelerate does num_devices steps at a time
        )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            
            self.results_folder = args.output_dir

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.diffusion, self.opt, self.dataloader, self.val_dataloader, self.test_dataloader, self.lr_scheduler = self.accelerator.prepare(self.diffusion, self.opt, self.dataloader, self.val_dataloader, self.test_dataloader, lr_scheduler)
        self.data_iter = cycle(self.dataloader)
        self.val_data_iter = cycle(self.val_dataloader)

    def save(self, save_step=False):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.diffusion),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'scheduler': self.lr_scheduler.state_dict(),
        }
        if save_step:
            torch.save(data, f'{self.results_folder}/model_{self.step}.pt')
        else:
            torch.save(data, f'{self.results_folder}/model.pt')

    def load(self, file_path=None, best=False, init_only=False):
        file_path = file_path if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        if best:
            data = torch.load(f'{file_path}/best_model.pt', map_location=device)
        else:
            data = torch.load(f'{file_path}/model.pt', map_location=device)

        model = self.accelerator.unwrap_model(self.diffusion)
        strict_load = not (init_only)
        model.load_state_dict(data['model'], strict=strict_load)

        if init_only:
            return
        
        # For backwards compatibility with earlier models
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])    

        self.opt.load_state_dict(data['opt'])
            
        self.step = data['step']
        self.lr_scheduler.load_state_dict(data['scheduler'])
        

    @torch.no_grad()
    def sample(self, num_samples=None, seed=None, cls_free_guidance=1.0, test=False, prefix_seconds=0.):
        if exists(seed):
            set_seeds(seed)
        diffusion = self.accelerator.unwrap_model(self.diffusion)
        num_samples = default(num_samples, self.num_samples)
        self.diffusion.eval() 
        inpainting_enabled = diffusion.inpainting_enabled() and diffusion.sampler != 'dpmpp' and exists(num_samples)
        num_sampled = 0
        dataloader = self.test_dataloader if test else self.val_dataloader
        for batch in dataloader:
            sampled_codec_latents = diffusion.sample(batch, prefix_seconds=prefix_seconds, cls_free_guidance=cls_free_guidance)
            sampled_wavs = diffusion.audio_codec.decode(sampled_codec_latents).squeeze()
            sampled_wavs = self.accelerator.gather_for_metrics(sampled_wavs).to('cpu')
            
            input_ids = self.accelerator.gather_for_metrics(batch['input_ids']).to('cpu')

            speaker_durations = self.accelerator.gather_for_metrics(batch['speaker_audio_duration']).to('cpu')
        
            if self.accelerator.is_main_process:
                inpainting_suffix = f'_prefix{prefix_seconds}' if prefix_seconds>0 else ''
                samples_folder = os.path.join(self.results_folder, 'samples', f'step_{self.step}', f'guide{cls_free_guidance}{inpainting_suffix}')
                os.makedirs(samples_folder, exist_ok=True)
                text_list = [diffusion.text_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in input_ids]
                ref_frames = torch.ceil(speaker_durations*ENCODEC_SAMPLING_RATE).int()
                for idx in range(len(text_list)):
                    text = text_list[idx]
                    print(f'Saving idx: {idx+num_sampled}')
                    with open(os.path.join(samples_folder, f'text_{idx+num_sampled}.txt'), 'w') as f:
                        print(text, file=f)
                    if prefix_seconds > 0:
                        ref_frames_idx = ref_frames[idx].item()
                        sf.write(os.path.join(samples_folder, f'audio_{idx+num_sampled}.wav'), sampled_wavs[idx][ref_frames_idx:], ENCODEC_SAMPLING_RATE)
                        sf.write(os.path.join(samples_folder, f'ref_{idx+num_sampled}.wav'), sampled_wavs[idx][:ref_frames_idx], ENCODEC_SAMPLING_RATE)
                    else:
                        sf.write(os.path.join(samples_folder, f'audio_{idx+num_sampled}.wav'), sampled_wavs[idx], ENCODEC_SAMPLING_RATE)
            batch_size = self.num_devices * batch['wav'].shape[0] 
            num_sampled += batch_size


            if exists(num_samples) and num_sampled >= num_samples:
                break        
        

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.data_iter)
                    loss = self.diffusion(data, accelerator)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()

                    self.accelerator.backward(loss)


                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(self.diffusion.parameters(), self.args.clip_grad_norm)
                self.opt.step()
                grad_norm = compute_grad_norm(self.diffusion.parameters())
                self.lr_scheduler.step()
                self.opt.zero_grad()
                self.step += 1

                if self.step % 10 == 0:
                    logs = {
                            "loss": total_loss,
                            "learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "grad_norm": grad_norm,
                            "step": self.step, 
                            "epoch": (self.step*self.gradient_accumulate_every)/len(self.dataloader), 
                            "samples": self.step*self.batch_size*self.gradient_accumulate_every*self.num_devices
                            }
                    # Validation loss
                    if self.step % 50 == 0:
                        with torch.no_grad():
                            total_val_loss = 0
                            data = next(self.val_data_iter)
                            loss = self.diffusion(data)
                            total_val_loss += loss.item()
                        logs['val_loss'] = total_val_loss
                    if accelerator.is_main_process:
                        pbar.set_postfix(**logs)  
                        accelerator.log(logs, step=self.step) 

                accelerator.wait_for_everyone()
                # Update EMA
                accelerator.unwrap_model(self.diffusion).denoising_network.update()             

                if self.step % self.save_and_sample_every == 0:
                    self.sample()
                    for cls_free_guidance in [2.0, 3.0, 5.0]:
                        self.sample(cls_free_guidance=cls_free_guidance)
                    
                    if self.prefix_inpainting_seconds > 0:
                        self.sample(prefix_seconds=self.prefix_inpainting_seconds)
                        for cls_free_guidance in [2.0, 3.0, 5.0]:
                            self.sample(cls_free_guidance=cls_free_guidance, prefix_seconds=self.prefix_inpainting_seconds)
                    self.save()
                    if self.step % (self.save_and_sample_every*2) == 0: 
                        self.save(save_step=True)
                    
                    self.diffusion.train() 

                pbar.update(1)

        accelerator.end_training()
        accelerator.print('training complete')