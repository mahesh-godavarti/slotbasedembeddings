"""CIFAR-100 runner for axial dense rotation (commutative by construction)."""
import sys
_original_print = print
import builtins
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, 'flush': True})
import torch
import lightning
from lightning.pytorch.callbacks import Callback
from lightning_modules.lightning_liere_image_classification import LiereImageClassification
from lightning_modules.lightning_data_image import Cifar100
from models.rope_vit_axial_dense import RoPEViT as RoPEViTAxialDense
import argparse


class EpochLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss", 0)
        val_acc = metrics.get("val_acc", 0)
        lr = metrics.get("lr", 0)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: lr={lr:.1e}, train_loss={train_loss:.4f}, val_acc={val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="tiny", choices=["tiny", "base", "large"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--rotate_v", action="store_true", default=False)
    args = parser.parse_args()

    size_configs = {
        "tiny": {"dim": 384, "depth": 12, "heads": 6, "mlp_dim": 768},
        "base": {"dim": 768, "depth": 12, "heads": 12, "mlp_dim": 3072},
    }
    cfg = size_configs[args.model_size]

    # Create LiereImageClassification and swap in axial dense model
    model = LiereImageClassification(
        learning_rate=args.lr,
        imsize=32,
        patch_size=[4, 4],
        rotate_v=args.rotate_v,
        model_architecture="liere",
        model_size=args.model_size,
        num_classes=100,
        input_dimensionality=2,
        emb_dropout=0.1,
        attn_dropout=0.1,
        num_channels=3,
        shuffle_patches=False,
        freeze_liere=False,
        rotary_embedding_per_layer=True,
        rotary_embedding_per_head=True,
        generator_dim=64,
    )

    # Replace with axial dense model
    model.model = RoPEViTAxialDense(
        image_size=32,
        patch_size=[4, 4],
        num_classes=100,
        dim=cfg["dim"],
        depth=cfg["depth"],
        heads=cfg["heads"],
        mlp_dim=cfg["mlp_dim"],
        positional_encoding_type="liere",
        input_dimensionality=2,
        phase_type="naver",
        position_sequencing_type="sequential",
        generator_dim=32,  # each axis gets 32-dim rotation
        force_absolute_encodings=False,
        rotary_embedding_per_layer=True,
        rotary_embedding_per_head=True,
        shuffle_patches=False,
        enable_ape=False,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_rot_params = sum(p.numel() for name, p in model.named_parameters() if 'gen_y' in name or 'gen_x' in name)
    print(f"Axial dense rotation, rotate_v={args.rotate_v}")
    print(f"Total params: {n_params:,} (rotation params: {n_rot_params:,})")

    data = Cifar100(per_device_batch_size=args.batch_size, imsize=32, ablation_factor=1, num_workers=13)

    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=[args.gpu],
        max_epochs=args.epochs,
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        deterministic=True,
        enable_progress_bar=False,
        callbacks=[EpochLogger()],
    )

    trainer.fit(model, data)
    result = trainer.validate(model, data)
    print(f"\nFinal val accuracy: {result[0]['val_acc']:.4f}")

if __name__ == "__main__":
    main()
