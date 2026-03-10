"""Integration test: full multi-process PPO pipeline with CartPole-v1.

Spawns Manager → DataServer → Learner → InferenceServer → Workers,
runs a small number of steps, and verifies the pipeline completes.
"""

import multiprocessing as mp
import pytest


def _run_training(result_dict):
    """Run full pipeline; store final step count in shared dict."""
    from beastrand.core.common import set_start_method
    set_start_method()

    from beastrand.ppo.config import Args
    from beastrand.nodes.manager import Manager

    args = Args()
    # Minimal config for fast test
    args.env_id = "CartPole-v1"
    args.total_env_steps = 2048
    args.num_workers = 2
    args.num_envs_per_worker = 2
    args.rollout = 32
    args.batch_size = 128
    args.minibatch_size = 64
    args.replay_capacity = 128
    args.learning_starts = 128
    args.train_epochs = 1
    args.learning_rate = 1e-3
    args.inference_device = "cpu"
    args.learner_device = "cpu"
    args.mlp_layers = [32, 32]
    args.normalize_input = False
    args.normalize_returns = False
    args.checkpoint_interval = 0
    args.run_name = "test_integration"

    mgr = Manager(args)
    mgr.launch()
    mgr.run_until_complete()

    result_dict["steps"] = mgr.ctx.global_step.value


@pytest.mark.slow
def test_full_pipeline_cartpole():
    """End-to-end: PPO trains on CartPole-v1 for 2048 steps without crashing."""
    ctx = mp.get_context("spawn")
    result_dict = ctx.Manager().dict()

    p = ctx.Process(target=_run_training, args=(result_dict,))
    p.start()
    p.join(timeout=60)

    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        pytest.fail("Integration test timed out after 60s")

    assert p.exitcode == 0, f"Training process exited with code {p.exitcode}"
    assert result_dict.get("steps", 0) >= 2048
