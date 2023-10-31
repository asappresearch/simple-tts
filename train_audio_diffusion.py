import argparse
from utils.utils import get_output_dir, parse_float_tuple
import json
import os
import numpy as np

from diffusion.audio_denoising_diffusion import GaussianDiffusion, Trainer
from models.unet import Unet1D
from transformers import AutoConfig


def main(args):

    config = AutoConfig.from_pretrained(args.text_encoder)
    text_dim = config.d_model

    model = Unet1D(
        dim=args.dim,
        text_dim=text_dim,
        dim_mults=args.dim_mults,
        inpainting_embedding = args.inpainting_embedding,
        conformer_transformer=args.conformer_transformer,
        num_transformer_layers=args.num_transformer_layers,
        scale_skip_connection=args.scale_skip_connection,
        dropout=args.dropout,
    )
    
    args.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {args.trainable_params}')

    diffusion = GaussianDiffusion(
        model,
        max_seq_len = 2048, 
        text_encoder=args.text_encoder,
        sampling_timesteps = args.sampling_timesteps,     # number of sampling steps
        sampler=args.sampler,
        train_schedule= args.train_schedule, 
        sampling_schedule= args.sampling_schedule,
        loss_type = args.loss_type,            # L1 or L2
        objective = args.objective,
        parameterization = args.parameterization,
        ema_decay = args.ema_decay,
        scale = args.scale,
        unconditional_prob = args.unconditional_prob,
        inpainting_prob = args.inpainting_prob,
    )

    trainer = Trainer(
        args=args,
        diffusion=diffusion,
        dataset_name=args.dataset_name,
        batch_size= args.batch_size,
        gradient_accumulate_every = args.gradient_accumulation_steps,
        train_lr = args.learning_rate,
        train_num_steps = args.num_train_steps,
        lr_schedule = args.lr_schedule,
        num_warmup_steps = args.lr_warmup_steps,
        adam_betas = (args.adam_beta1, args.adam_beta2),
        adam_weight_decay = args.adam_weight_decay,
        save_and_sample_every = args.save_and_sample_every,
        num_samples = args.num_samples,
        mixed_precision = args.mixed_precision,
        prefix_inpainting_seconds = args.prefix_inpainting_seconds,
        seed=args.seed,
    )

    if args.eval or args.eval_test:
        trainer.load(args.resume_dir)
        # trainer.sample()
        if args.prefix_inpainting_seconds > 0:
            cls_free_guidances = args.guidance
        elif 'ablation' in args.run_name:
            cls_free_guidances = [5.0] # Use stronger guidance for ablations b/c they are under-trained
        else:
            cls_free_guidances = args.guidance
        for cls_free_guidance in cls_free_guidances:
            trainer.sample(cls_free_guidance=cls_free_guidance, prefix_seconds=args.prefix_inpainting_seconds, test=args.eval_test, seed=42)
        return
    
    if args.init_model is not None:
        trainer.load(args.init_model, init_only=True)

    if args.resume:
        trainer.load(args.resume_dir)

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--dataset_name", type=str, default='librispeech')
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--text_encoder", type=str, default="google/byt5-small")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--init_model", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    # Architecture hyperparameters
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument('--dim_mults', type=parse_float_tuple, default=(1, 1, 1, 1.5), help='Tuple of integer values for dim_mults')
    parser.add_argument("--conformer_transformer", action="store_true", default=False)
    parser.add_argument("--scale_skip_connection", action="store_true", default=False)
    parser.add_argument("--num_transformer_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--inpainting_embedding", action="store_true", default=False)

    # Optimization hyperparameters
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_train_steps", type=int, default=60000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_schedule", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    # Diffusion Hyperparameters
    parser.add_argument(
        "--objective",
        type=str,
        default="pred_v",
        choices=["pred_eps", "pred_x0", "pred_v"],
        help=(
            "Which loss objective to use for the diffusion objective."
        ),
    )
    parser.add_argument(
        "--parameterization",
        type=str,
        default="pred_v",
        choices=["pred_eps", "pred_x0", "pred_v"],
        help=(
            "Which output parameterization to use for the diffusion network."
        ),
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l1",
        choices=["l1", "l2"],
        help=(
            "Which loss function to use for diffusion."
        ),
    )
    parser.add_argument(
        "--train_schedule",
        type=str,
        default="cosine",
        choices=["beta_linear", "simple_linear", "cosine", 'sigmoid'],
        help=(
            "Which noise schedule to use."
        ),
    )
    parser.add_argument(
        "--sampling_schedule",
        type=str,
        default=None,
        choices=["beta_linear", "cosine", "simple_linear", None],
        help=(
            "Which noise schedule to use."
        ),
    )
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--sampling_timesteps", type=int, default=250)
    # Audio Training Parameters
    parser.add_argument("--unconditional_prob", type=float, default=.1)
    parser.add_argument("--inpainting_prob", type=float, default=.5)
    # Generation Arguments
    parser.add_argument("--save_and_sample_every", type=int, default=5000)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=["ddim", "ddpm"],
        help=(
            "Which sampler use for diffusion."
        ),
    )
    parser.add_argument("--prefix_inpainting_seconds", type=float, default=0.)
    # Accelerate arguments
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    # Load and eval model
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--eval_test", action="store_true", default=False)
    parser.add_argument('--guidance', type=parse_float_tuple, help='Tuple of float values for dim_mults')
    
    args = parser.parse_args()
    if args.eval or args.eval_test:
        assert args.resume_dir is not None

    if args.eval or args.eval_test:
        with open(os.path.join(args.resume_dir, 'args.json'), 'rt') as f:
            saved_args = json.load(f)
        args_dict = vars(args)
        heldout_params = {'run_name', 'output_dir', 'resume_dir', 'eval', 'eval_test', 'prefix_inpainting_seconds', 'num_samples', 'sampling_timesteps', 'sampling_schedule', 'scale', 'sampler', 'mixed_precision', 'guidance',}
        for k,v in saved_args.items():
            if k in heldout_params:
                continue
            args_dict[k] = v
        
    main(args)