import torch
import torchaudio
from torch import nn, einsum
from encodec import EncodecModel
from encodec.utils import _linear_overlap_add
from encodec.utils import convert_audio
import typing as tp
import numpy as np
from tqdm import tqdm

from einops import rearrange
from functools import partial
import os

EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]

class EncodecWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_bandwidth(24.)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each frames is a tuple `(codebook, scale)`, with `codebook` of
        shape `[B, K, T]`, with `K` the number of codebooks.
        """
        assert x.dim() == 3
        _, channels, length = x.shape
        assert channels > 0 and channels <= 2
        segment_length = self.codec.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.codec.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: tp.List[EncodedFrame] = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset: offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        assert len(encoded_frames) == 1
        assert encoded_frames[0][1] is None
        return encoded_frames[0][0]

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        length = x.shape[-1]
        duration = length / self.codec.sample_rate
        assert self.codec.segment is None or duration <= 1e-5 + self.codec.segment

        if self.codec.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        emb = self.codec.encoder(x)
        return emb, scale

    def decode(self, emb: torch.Tensor, quantize: bool=True) -> torch.Tensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        encoded_frames = [(emb, None)]
        segment_length = self.codec.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = [self._decode_frame(frame, quantize=quantize) for frame in encoded_frames]
        return _linear_overlap_add(frames, self.segment_stride or 1)

    def _decode_frame(self, encoded_frame: EncodedFrame, quantize: bool=True) -> torch.Tensor:
        emb, scale = encoded_frame
        if quantize:
            codes = self.codec.quantizer.encode(emb, self.codec.frame_rate, self.codec.bandwidth)
            emb = self.codec.quantizer.decode(codes)

        # codes is [B, K, T], with T frames, K nb of codebooks.
        out = self.codec.decoder(emb)
        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out


    def forward(self, wav:torch.tensor, sr:int, quantize:bool=True):
        # TODO: Revisit where to handle processing
        wav = convert_audio(wav, sr, self.codec.sample_rate, self.codec.channels)
        frames = self.encode(wav)
        return self.decode(frames, quantize=quantize)[:, :, :wav.shape[-1]]



def test():
    codec = EncodecWrapper().to('cuda')
    import sys
    import soundfile as sf
    sys.path.append('../')
    from audio_datasets.librispeech import LibriSpeech, ENCODEC_SAMPLING_RATE
    from diffusion.noise_schedule import simple_linear_schedule, time_to_alpha, cosine_schedule

    test_dataset = LibriSpeech(split='test')
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            example = test_dataset.__getitem__(idx)

            # [B, 1, L]: batch x channels x length
            batched_wav = example['wav'][:,:int(example['audio_duration']*ENCODEC_SAMPLING_RATE)].unsqueeze(0).to('cuda')

            scales = [1.0, .5, .2]
            for scale in scales:
                times = [.05*x for x in range(0, 21)]
                alpha_schedule = partial(time_to_alpha, alpha_schedule=cosine_schedule, scale=scale)
                os.makedirs(f'example_audio/cosine/scale{scale}/', exist_ok=True)
                std_scale_factor = .22
                for time in times:
                    os.makedirs(f'example_audio/cosine/scale{scale}/time{time}', exist_ok=True)
                    alpha = alpha_schedule(torch.tensor([time], device=batched_wav.device))

                    wav_emb = codec.encode(batched_wav)*std_scale_factor
                    noisy_wav_emb = alpha.sqrt()*wav_emb + (1-alpha).sqrt()*torch.randn_like(wav_emb)
                    noisy_wav_emb /= alpha.sqrt()
                    noisy_wav_emb /= std_scale_factor
                    noisy_reconstruction = codec.decode(noisy_wav_emb)
                    
                    sf.write(f'example_audio/cosine/scale{scale}/time{time}/audio_{idx}.wav', noisy_reconstruction.squeeze().to('cpu').numpy(), ENCODEC_SAMPLING_RATE)    

        


if __name__=='__main__':
    test()