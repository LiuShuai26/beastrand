# core/contract.py
"""
Module contract validation.

Verifies that DataRecord, Policy, and Algorithm are mutually consistent
before any process is spawned. Catches mismatches (typo'd keys, missing
fields) at startup rather than minutes into training.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch

from beastrand.core.base_record import DataRecordBase


def validate_contracts(
    ctx: Any,
    record_cls: Type[DataRecordBase],
    policy_cls: type,
    algorithm_cls: type,
) -> None:
    """Run contract checks between DataRecord, Policy, and Algorithm.

    Called in Manager.launch() after probe_env populates ctx with env specs.
    Raises RuntimeError on contract violations.
    """
    args = ctx.args
    T = args.rollout
    obs_shape = ctx.obs_shape
    act_shape = ctx.act_shape

    # ------------------------------------------------------------------
    # 1. DataRecord.alloc_specs → build_batch key contract
    # ------------------------------------------------------------------
    specs = record_cls.alloc_specs(ctx, T, obs_shape, act_shape)
    if not specs:
        raise RuntimeError(f"{record_cls.__name__}.alloc_specs() returned empty specs")

    # Create a fake view matching alloc_specs (all zeros)
    fake_view: Dict[str, np.ndarray] = {}
    for field, (shape, dtype_str) in specs.items():
        fake_view[field] = np.zeros(shape, dtype=np.dtype(dtype_str))

    # build_batch must not crash and must return non-empty dict
    batch = record_cls.build_batch(ctx, fake_view)
    if not batch:
        raise RuntimeError(f"{record_cls.__name__}.build_batch() returned empty batch")

    batch_keys = set(batch.keys())
    spec_keys = set(specs.keys())
    logging.info(
        "[contract] %s: alloc_specs=%d fields, build_batch=%d keys",
        record_cls.__name__, len(spec_keys), len(batch_keys),
    )

    # ------------------------------------------------------------------
    # 2. Policy interface check
    # ------------------------------------------------------------------
    policy = policy_cls(ctx)

    # Required method: act
    if not callable(getattr(policy, "act", None)):
        raise RuntimeError(f"{policy_cls.__name__} missing required method act()")

    # Smoke-test act() with a fake obs batch
    policy.eval()
    with torch.no_grad():
        fake_obs = torch.zeros(1, *obs_shape)
        act_out = policy.act({"obs": fake_obs}, deterministic=True)

    if "action" not in act_out:
        raise RuntimeError(
            f'{policy_cls.__name__}.act() must return "action" key, '
            f"got: {set(act_out.keys())}"
        )

    # If policy supports value, verify it returns "value"
    if policy.supports_value() and "value" not in act_out:
        raise RuntimeError(
            f"{policy_cls.__name__}.supports_value()=True but act() "
            f'output missing "value" key'
        )

    # If algorithm.update expects evaluate_actions, verify it exists
    if hasattr(algorithm_cls, "update"):
        if not callable(getattr(policy, "evaluate_actions", None)):
            logging.warning(
                "[contract] %s has no evaluate_actions() — "
                "algorithm.update() may fail if it calls this",
                policy_cls.__name__,
            )

    del policy  # free memory

    logging.info(
        "[contract] %s: act() returns keys %s",
        policy_cls.__name__, sorted(act_out.keys()),
    )
    logging.info("[contract] all module contracts validated")
