import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser("Train pose model with UniRep Context-Residual blocks")

    # 基础路径
    parser.add_argument("--model", type=str, required=True, help="model yaml path")
    parser.add_argument("--data", type=str, required=True, help="data yaml path")
    parser.add_argument("--pretrained", type=str, default="", help="pretrained .pt path")
    parser.add_argument("--project", type=str, default="runs/pose_unirep_cr")
    parser.add_argument("--name", type=str, default="exp")

    # 训练超参
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["SGD", "Adam", "AdamW", "auto"])
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--lrf", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=float, default=3.0)
    parser.add_argument("--close_mosaic", type=int, default=10)

    # 训练策略
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cos_lr", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache", type=str, default="", choices=["", "ram", "disk"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freeze", type=int, default=0, help="freeze first N layers")
    parser.add_argument("--patience", type=int, default=100)

    # 数据增强
    parser.add_argument("--hsv_h", type=float, default=0.015)
    parser.add_argument("--hsv_s", type=float, default=0.7)
    parser.add_argument("--hsv_v", type=float, default=0.4)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)

    # pose 任务建议参数
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--cls", type=float, default=0.5)
    parser.add_argument("--dfl", type=float, default=1.5)
    parser.add_argument("--pose", type=float, default=12.0)
    parser.add_argument("--kobj", type=float, default=2.0)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    model_path = Path(args.model)
    data_path = Path(args.data)

    assert model_path.exists(), f"model yaml not found: {model_path}"
    assert data_path.exists(), f"data yaml not found: {data_path}"

    print("=" * 80)
    print("Train Pose with UniRep Context-Residual")
    print("=" * 80)
    print(f"model      : {model_path}")
    print(f"data       : {data_path}")
    print(f"pretrained : {args.pretrained if args.pretrained else 'None'}")
    print(f"device     : {args.device}")
    print("=" * 80)

    # 1) 从 yaml 构建
    model = YOLO(str(model_path))

    # 2) 可选：加载官方 pose 权重做 warm start
    #    会自动按名字/形状匹配可加载参数
    if args.pretrained:
        ckpt = Path(args.pretrained)
        assert ckpt.exists(), f"pretrained checkpoint not found: {ckpt}"
        model = model.load(str(ckpt))
        print(f"[OK] loaded pretrained weights from: {ckpt}")

    # 3) 开始训练
    train_args = dict(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        seed=args.seed,
        deterministic=args.deterministic,
        cos_lr=args.cos_lr,
        amp=args.amp,
        cache=(args.cache if args.cache else False),
        resume=args.resume,
        freeze=args.freeze,
        patience=args.patience,
        # augment
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        # loss gain
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        pose=args.pose,
        kobj=args.kobj,
        # 建议保留验证
        val=True,
        save=True,
        plots=True,
        verbose=True,
    )

    results = model.train(**train_args)

    print("=" * 80)
    print("Training finished.")
    print(results)
    print("=" * 80)


if __name__ == "__main__":
    main()