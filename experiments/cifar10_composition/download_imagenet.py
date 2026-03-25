"""Download ImageNet-1K from Hugging Face and save as ImageFolder format."""
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
OUT_DIR = Path("/home/ubuntu/cifar10_composition/data/imagenet")

def save_split(dataset, split_name, class_names):
    split_dir = OUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Create class directories
    for name in class_names:
        (split_dir / name).mkdir(exist_ok=True)

    # Count existing files to support resuming
    existing = sum(1 for _ in split_dir.rglob("*.JPEG"))
    skip = existing
    if existing > 0:
        print(f"  Found {existing} existing files, skipping those")

    # Save images (streaming — no local cache needed)
    for i, example in enumerate(tqdm(dataset, desc=f"Saving {split_name}")):
        if i < skip:
            continue

        label = example['label']
        class_name = class_names[label]
        out_path = split_dir / class_name / f"{i:08d}.JPEG"

        if out_path.exists():
            continue

        img = example['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(out_path)

def get_class_names():
    """Get class names from dataset info without downloading."""
    from datasets import load_dataset_builder
    builder = load_dataset_builder("ILSVRC/imagenet-1k", token=TOKEN)
    return builder.info.features['label'].names

def main():
    class_names = get_class_names()
    print(f"Got {len(class_names)} class names")

    for split, total in [("train", 1281167), ("validation", 50000)]:
        print(f"\nStreaming ImageNet-1K {split} split...")
        ds = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=True, token=TOKEN)
        save_split(ds, "val" if split == "validation" else split, class_names)

    print("Done!")

if __name__ == "__main__":
    main()
