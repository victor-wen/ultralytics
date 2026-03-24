#!/usr/bin/env python3
"""Migration-friendly entrypoint for training YOLO26 pose models with RTMO/legacy head selection."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """Add a boolean flag compatible with both old and new Python argparse versions."""
    dest = name.replace("-", "_")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(f"--{name}", dest=dest, action=argparse.BooleanOptionalAction, default=default, help=help_text)
        return

    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable {help_text.lower()}")
    parser.set_defaults(**{dest: default})


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="Train YOLO26 pose with the RTMO-style or legacy head.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--head", choices=("rtmo", "legacy"), default="rtmo", help="Pose head variant to train.")
    parser.add_argument("--scale", choices=("n", "s", "m", "l", "x"), default="n", help="YOLO26 model scale.")
    parser.add_argument("--weights", "--weight", dest="weights", default="", help="Optional pretrained weights to load before training.")
    parser.add_argument("--epochs", type=int, default=300, help="Total number of fine-tuning epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--val-batch", type=int, default=None, help="Optional validation batch size override.")
    parser.add_argument("--device", default="", help="Training device, e.g. '0', '0,1', or 'cpu'.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--project", default="runs/pose", help="Project directory for outputs.")
    parser.add_argument("--name", default="", help="Optional run name. If empty, one is generated from head/scale.")
    parser.add_argument("--optimizer", default=None, help="Optimizer override, e.g. SGD, AdamW, or auto.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate override.")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning rate fraction override.")
    parser.add_argument("--momentum", type=float, default=None, help="Momentum override.")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=None, help="Weight decay override.")
    parser.add_argument("--warmup-epochs", dest="warmup_epochs", type=float, default=None, help="Warmup epochs override.")
    parser.add_argument("--patience", type=int, default=None, help="Early-stopping patience override.")
    parser.add_argument("--box", type=float, default=None, help="Box loss gain override.")
    parser.add_argument("--cls", type=float, default=None, help="Cls loss gain override.")
    parser.add_argument("--pose", type=float, default=None, help="Pose loss gain override.")
    parser.add_argument("--kobj", type=float, default=None, help="Keypoint visibility loss gain override.")
    parser.add_argument("--proxy-pose", dest="proxy_pose", type=float, default=None, help="Proxy pose loss gain override.")
    parser.add_argument("--freeze-mode", choices=("rtmo_head_only", "none"), default="rtmo_head_only", help="Stage-1 freeze preset.")
    parser.add_argument("--stage1-epochs", type=int, default=20, help="Stage-1 epochs when staged fine-tuning is enabled.")
    parser.add_argument("--stage2-lr-scale", type=float, default=0.2, help="LR multiplier applied in stage 2.")
    parser.add_argument("--resume", default="", help="Resume from checkpoint path. Disables staged fine-tuning when set.")
    parser.add_argument("--cache", action="store_true", help="Enable dataset caching.")
    add_bool_arg(parser, "amp", default=True, help_text="Enable AMP training.")
    add_bool_arg(parser, "val", default=True, help_text="Run validation.")
    add_bool_arg(parser, "pretrained", default=True, help_text="Enable pretrained initialization behavior.")
    add_bool_arg(parser, "end2end", default=None, help_text="Override model end-to-end mode.")
    add_bool_arg(parser, "staged-ft", default=True, help_text="Run two-stage fine-tuning for RTMO migration.")
    add_bool_arg(parser, "save-init-report", default=True, help_text="Print a grouped weight-transfer report before training.")
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


def resolve_init_weights(args: argparse.Namespace) -> str:
    """Resolve the initialization weights to use for migration."""
    if args.weights:
        return args.weights
    if args.head == "rtmo" and args.pretrained:
        return f"yolo26{args.scale}-pose.pt"
    return ""


def summarize_weight_transfer(pose_model, weights: str) -> None:
    """Print a grouped report of which parameters will transfer from the initialization checkpoint."""
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.torch_utils import intersect_dicts

    try:
        loaded_model, _ = load_checkpoint(weights)
    except Exception as exc:
        print(f"[train_pose26] init report skipped for {weights}: {exc}")
        return

    source_state = loaded_model.float().state_dict()
    target_state = pose_model.state_dict()
    matched = intersect_dicts(source_state, target_state)
    head = pose_model.model[-1]
    head_prefix = f"model.{head.i}."
    groups = {
        "backbone_neck": {"total": 0, "matched": 0, "fresh": 0},
        "detect_head": {"total": 0, "matched": 0, "fresh": 0},
        "pose_stem": {"total": 0, "matched": 0, "fresh": 0},
        "rtmo_only": {"total": 0, "matched": 0, "fresh": 0},
    }

    def group_for_key(key: str) -> str:
        if not key.startswith(head_prefix):
            return "backbone_neck"
        if any(x in key for x in ("cv2.", "cv3.", "one2one_cv2.", "one2one_cv3.")):
            return "detect_head"
        if any(
            x in key
            for x in (
                "cv4_proxy",
                "cv4_pose_vec",
                "cv4_vis",
                "dcc.",
                "one2one_cv4_proxy",
                "one2one_cv4_pose_vec",
                "one2one_cv4_vis",
            )
        ):
            return "rtmo_only"
        if any(x in key for x in ("cv4.", "one2one_cv4.")):
            return "pose_stem"
        return "rtmo_only"

    for key in target_state:
        group = group_for_key(key)
        groups[group]["total"] += 1
        if key in matched:
            groups[group]["matched"] += 1
        else:
            groups[group]["fresh"] += 1

    print(f"[train_pose26] init report for {weights}")
    for group, stats in groups.items():
        print(
            f"[train_pose26]   {group}: matched={stats['matched']}/{stats['total']} "
            f"fresh_init={stats['fresh']}"
        )


def build_stage1_freeze(model, freeze_mode: str) -> tuple[int | None, list[str]]:
    """Return stage-1 freeze settings for migration-focused training."""
    if freeze_mode == "none":
        return None, []

    pose_model = model.model
    head = pose_model.model[-1]
    head_prefix = f"model.{head.i}."
    freeze_names = [
        f"{head_prefix}cv2.",
        f"{head_prefix}cv3.",
    ]
    if getattr(head, "end2end", False):
        freeze_names.extend([f"{head_prefix}one2one_cv2.", f"{head_prefix}one2one_cv3."])
    return head.i, freeze_names


def build_train_kwargs(
    args: argparse.Namespace,
    run_name: str,
    close_mosaic: int,
    init_weights: str,
    freeze: int | None = None,
    freeze_names: list[str] | None = None,
    lr0: float | None = None,
    epochs: int | None = None,
    resume: str | bool | None = None,
) -> dict:
    """Construct training kwargs, applying RTMO-friendly defaults when needed."""
    train_kwargs = {
        "data": args.data,
        "epochs": epochs if epochs is not None else args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "val_batch": args.val_batch,
        "workers": args.workers,
        "project": args.project,
        "name": run_name,
        "cache": args.cache,
        "amp": args.amp,
        "val": args.val,
        "pretrained": init_weights if init_weights else args.pretrained,
        "close_mosaic": close_mosaic,
        "seed": args.seed,
        "save_period": args.save_period,
    }

    if args.device:
        train_kwargs["device"] = args.device
    if args.optimizer is not None:
        train_kwargs["optimizer"] = args.optimizer
    if lr0 is not None:
        train_kwargs["lr0"] = lr0
    elif args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf
    if args.momentum is not None:
        train_kwargs["momentum"] = args.momentum
    if args.weight_decay is not None:
        train_kwargs["weight_decay"] = args.weight_decay
    if args.warmup_epochs is not None:
        train_kwargs["warmup_epochs"] = args.warmup_epochs
    if args.patience is not None:
        train_kwargs["patience"] = args.patience
    if freeze is not None:
        train_kwargs["freeze"] = freeze
    if freeze_names:
        train_kwargs["freeze_names"] = freeze_names
    if resume:
        train_kwargs["resume"] = resume

    if args.box is not None:
        train_kwargs["box"] = args.box
    elif args.head == "rtmo":
        train_kwargs["box"] = 5.0
    if args.cls is not None:
        train_kwargs["cls"] = args.cls
    elif args.head == "rtmo":
        train_kwargs["cls"] = 2.0
    if args.pose is not None:
        train_kwargs["pose"] = args.pose
    elif args.head == "rtmo":
        train_kwargs["pose"] = 5.0
    if args.kobj is not None:
        train_kwargs["kobj"] = args.kobj
    if args.proxy_pose is not None:
        train_kwargs["proxy_pose"] = args.proxy_pose
    elif args.head == "rtmo":
        train_kwargs["proxy_pose"] = 10.0

    return train_kwargs


def resolve_stage_checkpoint(results) -> Path | None:
    """Return the most useful checkpoint path from a training stage."""
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        return None
    weights_dir = Path(save_dir) / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    if best.exists():
        return best
    return last if last.exists() else None


def run_stage(model, train_kwargs: dict, label: str):
    """Run a single train stage and print a concise summary."""
    print(
        f"[train_pose26] {label}: epochs={train_kwargs['epochs']} batch={train_kwargs['batch']} "
        f"close_mosaic={train_kwargs['close_mosaic']}"
    )
    if train_kwargs.get("freeze_names"):
        print(f"[train_pose26] {label}: freeze_names={','.join(train_kwargs['freeze_names'])}")
    results = model.train(**train_kwargs)
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"[train_pose26] {label} finished. artifacts: {Path(save_dir)}")
    return results


def main() -> None:
    """Train a YOLO26 pose model."""
    args = build_parser().parse_args()
    from ultralytics import YOLO

    run_name = build_run_name(args.scale, args.head, args.name)
    init_weights = resolve_init_weights(args)
    staged_ft = args.staged_ft and args.head == "rtmo" and not args.resume

    if args.resume:
        print(f"[train_pose26] resuming from {args.resume}; staged fine-tuning disabled")
        model = YOLO(args.resume, task="pose")
        train_kwargs = build_train_kwargs(args, run_name, args.close_mosaic, init_weights="", resume=True)
        run_stage(model, train_kwargs, "resume")
        return

    model_yaml = resolve_model_yaml(args.scale, args.head)
    model_source = resolve_model_source(model_yaml, args.end2end)
    print(f"[train_pose26] model={model_yaml} head={args.head} scale={args.scale}")
    if args.end2end is not None:
        print(f"[train_pose26] overriding end2end={args.end2end} via {model_source}")
    if args.val_batch is not None:
        print(f"[train_pose26] overriding val_batch={args.val_batch}")
    if init_weights:
        print(f"[train_pose26] init_weights={init_weights}")
    print(f"[train_pose26] data={args.data} project={args.project} name={run_name}")

    model = YOLO(model_source, task="pose")
    if init_weights:
        if args.save_init_report:
            summarize_weight_transfer(model.model, init_weights)
        print(f"[train_pose26] loading weights from {init_weights}")
        model.load(init_weights)

    if not staged_ft:
        train_kwargs = build_train_kwargs(args, run_name, args.close_mosaic, init_weights=init_weights)
        run_stage(model, train_kwargs, "train")
        return

    stage1_epochs = max(min(args.stage1_epochs, args.epochs), 1)
    freeze, freeze_names = build_stage1_freeze(model, args.freeze_mode)
    stage1_name = f"{run_name}-stage1"
    stage1_kwargs = build_train_kwargs(
        args,
        stage1_name,
        close_mosaic=0,
        init_weights=init_weights,
        freeze=freeze,
        freeze_names=freeze_names,
        epochs=stage1_epochs,
    )
    stage1_results = run_stage(model, stage1_kwargs, "stage1")

    remaining_epochs = args.epochs - stage1_epochs
    if remaining_epochs <= 0:
        print("[train_pose26] stage2 skipped because total epochs are exhausted by stage1")
        return

    stage1_ckpt = resolve_stage_checkpoint(stage1_results)
    if stage1_ckpt is None:
        raise FileNotFoundError("Unable to locate stage1 checkpoint for stage2 fine-tuning.")

    stage2_lr0 = (args.lr0 if args.lr0 is not None else 0.01) * args.stage2_lr_scale
    print(f"[train_pose26] stage2 init checkpoint={stage1_ckpt} lr0={stage2_lr0}")
    stage2_model = YOLO(str(stage1_ckpt), task="pose")
    stage2_kwargs = build_train_kwargs(
        args,
        f"{run_name}-stage2",
        close_mosaic=args.close_mosaic,
        init_weights="",
        lr0=stage2_lr0,
        epochs=remaining_epochs,
        freeze=None,
        freeze_names=[],
    )
    run_stage(stage2_model, stage2_kwargs, "stage2")


if __name__ == "__main__":
    main()
