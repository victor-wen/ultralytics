#!/usr/bin/env python3
"""Convenience entrypoint for training YOLO26 pose models with RTMO/legacy head selection."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """Add a boolean flag compatible with both old and new Python argparse versions."""
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=default, help=help_text)
        return

    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help_text.lower()}")
    parser.set_defaults(**{name: default})


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="Train YOLO26 pose with the RTMO-style or legacy head.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--head", choices=("rtmo", "legacy"), default="rtmo", help="Pose head variant to train.")
    parser.add_argument("--scale", choices=("n", "s", "m", "l", "x"), default="n", help="YOLO26 model scale.")
    parser.add_argument("--weights", "--weight", dest="weights", default="", help="Optional pretrained weights to load before training.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="", help="Training device, e.g. '0', '0,1', or 'cpu'.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--project", default="runs/pose", help="Project directory for outputs.")
    parser.add_argument("--name", default="", help="Optional run name. If empty, one is generated from head/scale.")
    parser.add_argument("--cache", action="store_true", help="Enable dataset caching.")
    add_bool_arg(parser, "amp", default=True, help_text="Enable AMP training.")
    add_bool_arg(parser, "val", default=True, help_text="Run validation.")
    add_bool_arg(parser, "pretrained", default=True, help_text="Enable pretrained behavior.")
    add_bool_arg(parser, "end2end", default=None, help_text="Override model end-to-end mode.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Epochs before disabling mosaic.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--save-period", type=int, default=-1, help="Checkpoint save period. -1 disables periodic saves.")
    return parser


def resolve_model_yaml(scale: str, head: str) -> str:
    """Resolve the model YAML name from scale and head selection."""
    suffix = "-pose-legacy.yaml" if head == "legacy" else "-pose.yaml"
    return f"yolo26{scale}{suffix}"


def resolve_model_source(model_yaml: str, end2end: bool | None) -> str:
    """Return the model source path, optionally overriding end-to-end mode via a temp YAML."""
    if end2end is None:
        return model_yaml

    from ultralytics.nn.tasks import yaml_model_load
    from ultralytics.utils import YAML

    cfg = yaml_model_load(model_yaml)
    cfg["end2end"] = end2end
    cfg.pop("yaml_file", None)
    temp_path = Path(tempfile.gettempdir()) / f"{Path(model_yaml).stem}-end2end-{str(end2end).lower()}-{os.getpid()}.yaml"
    YAML.save(temp_path, cfg)
    return str(temp_path)


def build_run_name(scale: str, head: str, explicit_name: str) -> str:
    """Build a default run name if the user did not supply one."""
    if explicit_name:
        return explicit_name
    return f"yolo26{scale}-pose-{head}"


def main() -> None:
    """Train a YOLO26 pose model."""
    args = build_parser().parse_args()
    from ultralytics import YOLO

    model_yaml = resolve_model_yaml(args.scale, args.head)
    model_source = resolve_model_source(model_yaml, args.end2end)
    run_name = build_run_name(args.scale, args.head, args.name)

    print(f"[train_pose26] model={model_yaml} head={args.head} scale={args.scale}")
    if args.end2end is not None:
        print(f"[train_pose26] overriding end2end={args.end2end} via {model_source}")
    print(f"[train_pose26] data={args.data} project={args.project} name={run_name}")

    model = YOLO(model_source, task="pose")
    if args.weights:
        print(f"[train_pose26] loading weights from {args.weights}")
        model.load(args.weights)

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": run_name,
        "cache": args.cache,
        "amp": args.amp,
        "val": args.val,
        "pretrained": args.pretrained,
        "close_mosaic": args.close_mosaic,
        "seed": args.seed,
        "save_period": args.save_period,
    }
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"[train_pose26] training finished. artifacts: {Path(save_dir)}")


if __name__ == "__main__":
    main()
