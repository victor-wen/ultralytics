# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn

from ultralytics.nn.modules import build_pose_supervision_mask
from ultralytics.utils.loss import KeypointLoss, PoseLoss26, RLELoss


class _DummyFlow:
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


def test_build_pose_supervision_mask_adds_center_positive_without_overwriting_fg():
    anchor_points = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    stride_tensor = torch.tensor([[8.0], [8.0], [32.0]], dtype=torch.float32)
    gt_bboxes = torch.tensor([[[12.0, 4.0, 20.0, 12.0]]], dtype=torch.float32)
    mask_gt = torch.tensor([[[True]]])
    fg_mask = torch.tensor([[False, True, False]])
    target_gt_idx = torch.zeros((1, 3), dtype=torch.long)

    pose_mask, pose_gt_idx, pose_weight, extra_pose_mask = build_pose_supervision_mask(
        anchor_points=anchor_points,
        stride_tensor=stride_tensor,
        gt_bboxes=gt_bboxes,
        mask_gt=mask_gt,
        fg_mask=fg_mask,
        target_gt_idx=target_gt_idx,
        allowed_strides=(8, 16),
        center_radius=1.5,
        extra_weight=0.5,
    )

    assert torch.equal(pose_mask, torch.tensor([[True, True, False]]))
    assert torch.equal(extra_pose_mask, torch.tensor([[True, False, False]]))
    assert torch.equal(pose_gt_idx, torch.zeros((1, 3), dtype=torch.long))
    assert torch.allclose(pose_weight, torch.tensor([[0.5, 1.0, 0.0]]))


def test_pose_loss26_applies_extra_weight_to_location_and_rle():
    criterion = PoseLoss26.__new__(PoseLoss26)
    criterion.keypoint_loss = KeypointLoss(sigmas=torch.ones(2))
    criterion.rle_loss = RLELoss(use_target_weight=True)
    criterion.target_weights = torch.ones(2)
    criterion.flow_model = _DummyFlow()
    criterion.bce_pose = nn.BCEWithLogitsLoss()

    pose_mask = torch.tensor([[True, True]])
    pose_gt_idx = torch.tensor([[0, 0]], dtype=torch.long)
    gt_kpts = torch.tensor([[[[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]], dtype=torch.float32)
    gt_bboxes = torch.tensor([[[0.0, 0.0, 4.0, 4.0]]], dtype=torch.float32)
    stride_tensor = torch.ones((2, 1), dtype=torch.float32)
    pred_kpts = torch.tensor(
        [[[[0.0, 0.0, 2.0, 0.0, 0.0], [1.0, 1.0, 2.0, 0.0, 0.0]],
          [[2.0, 2.0, -2.0, 0.0, 0.0], [3.0, 3.0, -2.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )

    full_weight = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    half_extra_weight = torch.tensor([[1.0, 0.5]], dtype=torch.float32)

    full_losses = criterion.calculate_keypoints_loss(
        pose_mask,
        pose_gt_idx,
        full_weight,
        gt_kpts,
        gt_bboxes,
        stride_tensor,
        pred_kpts,
    )
    weighted_losses = criterion.calculate_keypoints_loss(
        pose_mask,
        pose_gt_idx,
        half_extra_weight,
        gt_kpts,
        gt_bboxes,
        stride_tensor,
        pred_kpts,
    )

    assert weighted_losses[0] < full_losses[0]
    assert weighted_losses[1] < full_losses[1]
    assert weighted_losses[2] < full_losses[2]


def test_pose_loss26_rle_none_handles_mixed_precision():
    criterion = PoseLoss26.__new__(PoseLoss26)
    criterion.rle_loss = RLELoss(use_target_weight=True)
    criterion.target_weights = torch.ones(2)
    criterion.flow_model = _DummyFlow()

    pred_kpt = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]], dtype=torch.float16)
    gt_kpt = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]], dtype=torch.float32)
    kpt_mask = torch.tensor([[True, True]])

    loss = criterion.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask, reduction="none")

    assert loss.shape == (1,)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss).all()
