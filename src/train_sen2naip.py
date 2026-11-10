"""
train_sen2naip: the attn -> attn_gan_v3 recipe, retrained on real Sentinel-2 ->
NAIP pairs (src/data/sen2naip.py) instead of synthetically degraded S2 patches.

What changes. The original pipeline makes LR by blurring and subsampling
Sentinel-2 itself, so the model learns 40 m -> 10 m on a degradation it was
told about. Here LR is a real S2 image and HR is real 2.5 m aerial imagery, so
the model learns S2 10 m -> 2.5 m from the sensor's actual blur, noise and
misregistration -- the problem the project exists to solve.

Sized for a 1-2 hour budget on one GPU (an M-series Mac's MPS backend included):
an epoch is `train_length` random crops (300 batches of 8), not a pass over the
data, so the run time is set by the epoch count alone and does not grow with
the number of downloaded pairs.

Outputs (no existing checkpoint is touched):
    stage 1 -> checkpoints/best_model_attn_sen2naip.pt      accuracy model
    stage 2 -> checkpoints/best_model_attn_gan_sen2naip.pt  visual model

Prerequisite:
    python -m src.data.sen2naip --pairs 1500

Usage:
    python -m src.train_sen2naip                    # stage 1 only
    python -m src.train_sen2naip --stage both       # stage 1, then the GAN refinement
    python -m src.train_sen2naip --stage gan        # GAN only, from an existing stage 1
    python -m src.eval.perceptual_metrics --data sen2naip --compare
"""

import argparse
from pathlib import Path

BASE_CKPT = "checkpoints/best_model_attn_sen2naip.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="fidelity", choices=["fidelity", "gan", "both"])
    parser.add_argument("--epochs", type=int, default=25,
                        help="stage 1 epochs (early stopping may end it sooner)")
    parser.add_argument("--gan_epochs", type=int, default=10)
    parser.add_argument("--train_length", type=int, default=2400,
                        help="random crops per epoch")
    args = parser.parse_args()

    if not Path("data/sen2naip/index.json").exists():
        raise SystemExit("data/sen2naip/index.json not found. Download the pairs first:\n"
                         "    python -m src.data.sen2naip --pairs 1500")

    if args.stage in ("fidelity", "both"):
        from src.train_v2 import train as train_fidelity
        print("Stage 1 · Attention-RRDB on real S2 -> NAIP pairs\n")
        train_fidelity(data="sen2naip", run_suffix="_sen2naip", num_epochs=args.epochs,
                       train_length=args.train_length)

    if args.stage in ("gan", "both"):
        if not Path(BASE_CKPT).exists():
            raise SystemExit(f"{BASE_CKPT} not found -- run stage 1 first (--stage fidelity).")
        from src.train_v2_gan import train as train_gan
        print("\nStage 2 · adversarial refinement, attn_gan_v3 settings\n")
        # attn_gan_v3's settings: v1's adversarial pressure, v2's slow discriminator, EMA
        train_gan(data="sen2naip", warm_start=BASE_CKPT, run_suffix="_sen2naip",
                  w_adv=5e-3, lr_d=2e-5, ema_decay=0.999, num_epochs=args.gan_epochs,
                  train_length=args.train_length, save_every=max(args.gan_epochs, 1))


if __name__ == "__main__":
    main()
