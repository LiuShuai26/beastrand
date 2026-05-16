"""Regression test for the obs[T] trajectory boundary write.

The worker stores per-step observations into ``traj_tensors["obs"][:, 0:T]``;
``obs[T]`` is the rollout boundary read by consumers like AMP (which builds
transition pairs via ``obs[:-1] / obs[1:]``). The prior implementation never
wrote ``obs[T]``, leaving stale data from a previous slot owner.

The fix lives in :func:`beastrand.nodes.rollout_worker.write_boundary_obs`,
called from ``_finalize_trajectory``. Tested via the helper directly so the
invariant has explicit coverage without standing up a real worker process.
"""

import numpy as np
import torch

from beastrand.nodes.rollout_worker import write_boundary_obs


def _make_traj_tensors(num_traj: int, T: int, obs_dim: int) -> dict:
    return {"obs": torch.zeros(num_traj, T + 1, obs_dim, dtype=torch.float32)}


def test_writes_real_next_obs_when_not_done():
    T = 4
    obs_dim = 5
    traj = _make_traj_tensors(num_traj=2, T=T, obs_dim=obs_dim)
    # Prime obs[T] with stale data so we can prove the write actually happened.
    traj["obs"][0, T] = 99.0
    es_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    write_boundary_obs(traj, traj_idx=0, T=T, obs=es_obs, done=False)

    assert torch.allclose(traj["obs"][0, T], torch.from_numpy(es_obs))


def test_zeros_boundary_when_done():
    """env auto-reset has already pointed es.obs at a NEW episode; writing it
    at obs[T] would corrupt downstream transition consumers. Zero instead."""
    T = 4
    obs_dim = 3
    traj = _make_traj_tensors(num_traj=2, T=T, obs_dim=obs_dim)
    # Prime with non-zero so the zero is visible.
    traj["obs"][0, T] = torch.tensor([7.0, 8.0, 9.0])
    es_obs_after_reset = np.array([42.0, 42.0, 42.0], dtype=np.float32)

    write_boundary_obs(traj, traj_idx=0, T=T, obs=es_obs_after_reset, done=True)

    assert torch.allclose(traj["obs"][0, T], torch.zeros(obs_dim))


def test_no_op_when_no_obs_field():
    """The schema for some custom records may omit "obs"; the helper must
    silently skip rather than crash."""
    write_boundary_obs(
        {"value": torch.zeros(1, 5)}, traj_idx=0, T=4, obs=np.zeros(3), done=False
    )  # no exception


def test_does_not_touch_obs_at_other_indices():
    """Write must be scoped to (traj_idx, T) only — no spillover into
    obs[:T] (which the per-step writes own) or other trajectories."""
    T = 4
    obs_dim = 2
    traj = _make_traj_tensors(num_traj=3, T=T, obs_dim=obs_dim)
    # Fill obs[0:T] for traj 0 and all rows for traj 1/2 with sentinels.
    traj["obs"][0, :T] = 1.0
    traj["obs"][1] = 2.0
    traj["obs"][2] = 3.0
    es_obs = np.array([5.0, 6.0], dtype=np.float32)

    write_boundary_obs(traj, traj_idx=0, T=T, obs=es_obs, done=False)

    assert torch.allclose(traj["obs"][0, :T], torch.ones(T, obs_dim))
    assert torch.allclose(traj["obs"][1], 2.0 * torch.ones(T + 1, obs_dim))
    assert torch.allclose(traj["obs"][2], 3.0 * torch.ones(T + 1, obs_dim))
