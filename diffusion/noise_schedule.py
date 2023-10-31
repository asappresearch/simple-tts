import torch
import math
import numpy as np
from functools import partial
import matplotlib.pyplot as plt


# Avoid log(0)
def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps))

# noise schedules

def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def beta_linear_schedule(t, clip_min = 1e-9):
    return torch.exp(-1e-4 - 10 * (t ** 2)).clamp(min = clip_min, max = 1.)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

# converting gamma to alpha, sigma or logsnr
def log_snr_to_alpha(log_snr):
    alpha = torch.sigmoid(log_snr)
    return alpha

# Log-SNR shifting (https://arxiv.org/abs/2301.10972)
def alpha_to_shifted_log_snr(alpha, scale = 1):
    return (log(alpha) - log(1 - alpha)).clamp(min=-20, max=20) + 2*np.log(scale).item()

def time_to_alpha(t, alpha_schedule, scale):
    alpha = alpha_schedule(t)
    shifted_log_snr = alpha_to_shifted_log_snr(alpha, scale = scale)
    return log_snr_to_alpha(shifted_log_snr)

def plot_noise_schedule(unscaled_sampling_schedule, name, y_value):
    assert y_value in {'alpha^2', 'alpha', 'log(SNR)'}
    t = torch.linspace(0, 1, 100)  # 100 points between 0 and 1
    scales = [.2, .5, 1.0]
    for scale in scales:
        sampling_schedule = partial(time_to_alpha, alpha_schedule=unscaled_sampling_schedule, scale=scale)
        alphas = sampling_schedule(t)  # Obtain noise schedule values for each t
        if y_value == 'alpha^2':
            y_axis_label = r'$\alpha^2_t$'
            y = alphas
        elif y_value == 'alpha':
            y_axis_label = r'$\alpha_t$'
            y = alphas.sqrt()
        elif y_value == 'log(SNR)':
            y_axis_label = r'$\log(\lambda_t)$'
            y = alpha_to_shifted_log_snr(alphas, scale=1)

        plt.plot(t.numpy(), y.numpy(), label=f'Scale: {scale:.1f}')
    if y_value == 'log(SNR)':
        plt.ylim(-15, 15)
    plt.xlabel('t')
    plt.ylabel(y_axis_label)
    plt.title(f'{name}')
    plt.legend()
    plt.savefig(f'viz/{name.lower()}_{y_value}.png')
    plt.clf()


def plot_side_by_side_noise_schedule(unscaled_sampling_schedule, name):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    
    t = torch.linspace(0, 1, 100) 
    scales = [1.0, .5, .2]
    
    for i, y_value in enumerate(['alpha', 'log(SNR)']):
    
        for scale in scales:
            sampling_schedule = partial(time_to_alpha, alpha_schedule=unscaled_sampling_schedule, scale=scale)
            alphas = sampling_schedule(t)
            
            if y_value == 'alpha':
                y_axis_label = r'$\alpha_t$' 
                y = alphas.sqrt()
            elif y_value == 'log(SNR)':
                y_axis_label = r'$\log(\lambda_t)$'
                y = alpha_to_shifted_log_snr(alphas, scale=1)
                
            ax = ax1 if i == 0 else ax2
            ax.plot(t.numpy(), y.numpy(), label=f'Scale: {scale:.1f}')
            
            if y_value == 'log(SNR)':
                ax.set_ylim(-15, 15)
                
            ax.set_xlabel('t', fontsize=14)
            ax.set_ylabel(y_axis_label, fontsize=14)
            ax.legend()

                        
    fig.suptitle(f'{name}', fontsize=18)
    fig.tight_layout()
    plt.savefig(f'viz/{name.lower().replace(" ", "_")}.png')
    plt.clf()


def plot_cosine_schedule():
    t = torch.linspace(0, 1, 100)  # 100 points between 0 and 1
    sampling_schedule = cosine_schedule
    alphas = sampling_schedule(t)  # Obtain noise schedule values for each t
    y = alphas
    plt.plot(t.numpy(), y.numpy())
    plt.xlabel('t')
    plt.ylabel(f'alpha^2')
    plt.title(f'Cosine Noise Schedule')
    plt.savefig(f'viz/standard_cosine.png')
    plt.clf()

def visualize():
    unscaled_sampling_schedule = cosine_schedule
    
    plot_side_by_side_noise_schedule(unscaled_sampling_schedule, 'Shifted Cosine Noise Schedules')



if __name__=='__main__':
    visualize()