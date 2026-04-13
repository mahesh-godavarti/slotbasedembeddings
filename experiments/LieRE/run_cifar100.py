"""Simple CIFAR-100 runner for LieRE, bypassing LightningCLI."""
import sys
import builtins
_original_print = print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, 'flush': True})
import torch
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from lightning_modules.lightning_liere_image_classification import LiereImageClassification
from lightning_modules.lightning_data_image import Cifar100
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
    parser.add_argument("--model_architecture", default="liere", choices=["liere", "rope_mixed", "absolute"])
    parser.add_argument("--model_size", default="base", choices=["tiny", "base", "large"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--generator_dim", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--per_layer", action="store_true", default=True)
    parser.add_argument("--per_head", action="store_true", default=True)
    parser.add_argument("--rotate_v", action="store_true", default=False)
    args = parser.parse_args()

    model = LiereImageClassification(
        learning_rate=args.lr,
        imsize=32,
        patch_size=[4, 4],
        model_architecture=args.model_architecture,
        model_size=args.model_size,
        num_classes=100,
        input_dimensionality=2,
        emb_dropout=0.1,
        attn_dropout=0.1,
        num_channels=3,
        shuffle_patches=False,
        freeze_liere=False,
        rotary_embedding_per_layer=args.per_layer,
        rotary_embedding_per_head=args.per_head,
        generator_dim=args.generator_dim,
        rotate_v=args.rotate_v,
    )

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
