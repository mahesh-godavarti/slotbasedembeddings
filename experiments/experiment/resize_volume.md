# Resizing an EBS Volume on a Running EC2 Instance

## Steps

### 1. Resize the volume in AWS Console (or CLI)

In the AWS Console:
1. Go to **EC2 → Volumes**
2. Select the volume attached to the instance
3. **Actions → Modify Volume**
4. Change the size (e.g., 50GB → 200GB)
5. Confirm

Or via CLI:
```bash
aws ec2 modify-volume --volume-id vol-XXXXXXXXX --size 200
```

Wait for the volume state to change from `optimizing` to `completed` (the instance can be used during this time).

### 2. Grow the partition (on the instance)

The block device now shows the new size, but the partition still has the old size.

```bash
# Check current state
lsblk
df -h /

# Grow partition 1 to fill the disk
sudo growpart /dev/nvme0n1 1
```

### 3. Resize the filesystem

```bash
# For ext4:
sudo resize2fs /dev/nvme0n1p1

# For xfs (use this instead if your filesystem is xfs):
# sudo xfs_growfs /
```

### 4. Verify

```bash
df -h /
```

The filesystem should now show the new size.

## Notes

- No reboot is required. All steps work on a running instance.
- The device name (`nvme0n1`) and partition number (`1`) may differ on other instances. Use `lsblk` to identify them.
- `growpart` extends the partition table entry; `resize2fs` extends the filesystem within it. Both are needed.
