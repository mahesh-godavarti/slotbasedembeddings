"""CIFAR-100 runner for axial ortho rotation (free matrices + orthogonality penalty)."""
import sys
_original_print = print
import builtins
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, 'flush': True})
import torch
import lightning
from lightning.pytorch.callbacks import Callback
from lightning_modules.lightning_liere_image_classification import LiereImageClassification
from lightning_modules.lightning_data_image import Cifar100
from models.rope_vit_axial_ortho import RoPEViT as RoPEViTAxialOrtho
import argparse


class EpochLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss", 0)
        val_acc = metrics.get("val_acc", 0)
        lr = metrics.get("lr", 0)
        ortho = metrics.get("ortho_penalty", 0)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: lr={lr:.1e}, train_loss={train_loss:.4f}, val_acc={val_acc:.2f}%, ortho={ortho:.4f}")


class AxialOrthoImageClassification(LiereImageClassification):
    """Extends LiereImageClassification to add orthogonality penalty."""

    def __init__(self, ortho_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ortho_weight = ortho_weight

    def training_step(self, batch, batch_idx):
        inputs, targets = self.parse_batch(batch)
        outputs = self.model(inputs, rotate_v=self.rotate_v)
        ce_loss = self.criterion_train(outputs, targets)

        # Add orthogonality penalty
        ortho_penalty = self.model.position_encoder.ortho_penalty()
        loss = ce_loss + self.ortho_weight * ortho_penalty

        self.log(name="train_loss", value=ce_loss, prog_bar=True)
        self.log(name="lr", value=self.lr_schedulers().get_last_lr()[0])
        self.log(name="ortho_penalty", value=ortho_penalty, prog_bar=True)
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="tiny", choices=["tiny", "base", "large"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--rotate_v", action="store_true", default=False)
    parser.add_argument("--ortho_weight", type=float, default=0.1)
    args = parser.parse_args()

    size_configs = {
        "tiny": {"dim": 384, "depth": 12, "heads": 6, "mlp_dim": 768},
        "base": {"dim": 768, "depth": 12, "heads": 12, "mlp_dim": 3072},
    }
    cfg = size_configs[args.model_size]

    model = AxialOrthoImageClassification(
        ortho_weight=args.ortho_weight,
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

    # Replace with axial ortho model
    model.model = RoPEViTAxialOrtho(
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
        generator_dim=32,
        force_absolute_encodings=False,
        rotary_embedding_per_layer=True,
        rotary_embedding_per_head=True,
        shuffle_patches=False,
        enable_ape=False,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_rot_params = sum(p.numel() for name, p in model.named_parameters() if 'R_y' in name or 'R_x' in name)
    print(f"Axial ortho rotation, rotate_v={args.rotate_v}, ortho_weight={args.ortho_weight}")
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
