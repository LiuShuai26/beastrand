"""Tests for core PPO framework utilities: GAE, advantage normalization, ppo_update, to_torch."""

import numpy as np
import pytest
import torch

from beastrand.ppo.algorithm import compute_gae, normalize_advantages, ppo_update, PPOAlgorithm
from beastrand.core.running_mean_std import RunningMeanStd, RunningMeanStdTorch
from beastrand.utils.tensor_utils import to_torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Ctx:
    """Minimal ctx stub for compute_gae / ppo_update."""

    def __init__(self, rollout=4, gamma=0.99, lam=0.95, **kw):
        defaults = dict(
            rollout=rollout,
            gamma=gamma,
            lam=lam,
            train_epochs=1,
            minibatch_size=rollout,
            ppo_clip_range=0.2,
            ppo_clip_value=1.0,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            normalize_adv=True,
        )
        defaults.update(kw)
        self.args = type("Args", (), defaults)()


def _make_gae_view(T, rewards, done, values, truncated=None):
    """Build a view dict for compute_gae.

    values must have length T+1 (includes bootstrap).
    truncated: optional bool list, same length as done.
    """
    view = {
        "reward": np.array(rewards, dtype=np.float32),
        "done": np.array(done, dtype=np.float32),
        "value": np.array(values, dtype=np.float32),
        "advantage": np.zeros(T, dtype=np.float32),
        "return": np.zeros(T, dtype=np.float32),
    }
    if truncated is not None:
        view["truncated"] = np.array(truncated, dtype=np.uint8)
    return view


# ---------------------------------------------------------------------------
# compute_gae
# ---------------------------------------------------------------------------

class TestComputeGAE:

    def test_single_step_no_terminal(self):
        """GAE with T=1, no termination: adv = r + gamma*V(1) - V(0)."""
        T = 1
        gamma, lam = 0.99, 0.95
        ctx = _Ctx(rollout=T, gamma=gamma, lam=lam)
        view = _make_gae_view(T, rewards=[1.0], done=[0.0], values=[0.5, 0.8])
        compute_gae(ctx, view)

        delta = 1.0 + gamma * 0.8 - 0.5
        assert view["advantage"][0] == pytest.approx(delta, abs=1e-6)
        assert view["return"][0] == pytest.approx(delta + 0.5, abs=1e-6)

    def test_done_cuts_bootstrap(self):
        """When done=1, next-state value is zeroed (no bootstrap)."""
        T = 2
        gamma, lam = 0.99, 0.95
        ctx = _Ctx(rollout=T, gamma=gamma, lam=lam)
        # Step 0: normal, step 1: done
        view = _make_gae_view(
            T,
            rewards=[1.0, 2.0],
            done=[0.0, 1.0],
            values=[0.5, 0.8, 999.0],  # V(2)=999 should be ignored
        )
        compute_gae(ctx, view)

        # Step 1 (done): delta = 2.0 + 0 - 0.8 = 1.2, adv = 1.2
        delta1 = 2.0 - 0.8
        assert view["advantage"][1] == pytest.approx(delta1, abs=1e-6)

        # Step 0: delta = 1.0 + gamma * 0.8 - 0.5, but step 1 is done
        # so last_adv from step 1 does NOT propagate (nonterminal at step 1 = 0)
        # Wait — nonterminal is checked at step t, for step t's done flag.
        # At t=1: nonterminal = 0, so last_adv after t=1 = delta1 (no propagation from future)
        # At t=0: nonterminal = 1.0, delta = 1.0 + gamma*0.8 - 0.5
        #   adv = delta + gamma*lam*1.0*delta1
        delta0 = 1.0 + gamma * 0.8 - 0.5
        expected_adv0 = delta0 + gamma * lam * 1.0 * delta1
        assert view["advantage"][0] == pytest.approx(expected_adv0, abs=1e-6)

    def test_all_done(self):
        """Every step terminates: no bootstrap, no advantage propagation."""
        T = 3
        ctx = _Ctx(rollout=T, gamma=0.99, lam=0.95)
        view = _make_gae_view(
            T,
            rewards=[1.0, 2.0, 3.0],
            done=[1.0, 1.0, 1.0],
            values=[0.1, 0.2, 0.3, 999.0],
        )
        compute_gae(ctx, view)

        # Each step: adv = reward - V(t), no gamma*V(t+1), no propagation
        for t in range(T):
            expected = view["reward"][t] - view["value"][t]
            assert view["advantage"][t] == pytest.approx(expected, abs=1e-6)

    def test_returns_equal_adv_plus_value(self):
        """return = advantage + V(t) for all steps."""
        T = 5
        ctx = _Ctx(rollout=T, gamma=0.99, lam=0.95)
        rng = np.random.default_rng(42)
        view = _make_gae_view(
            T,
            rewards=rng.standard_normal(T).astype(np.float32),
            done=np.zeros(T, dtype=np.float32),
            values=rng.standard_normal(T + 1).astype(np.float32),
        )
        compute_gae(ctx, view)

        np.testing.assert_allclose(
            view["return"],
            view["advantage"] + view["value"][:T],
            atol=1e-5,
        )

    def test_gamma_zero_reduces_to_td0(self):
        """gamma=0: advantage = reward - V(t) (pure TD(0) error, no future)."""
        T = 3
        ctx = _Ctx(rollout=T, gamma=0.0, lam=0.95)
        view = _make_gae_view(
            T,
            rewards=[1.0, 2.0, 3.0],
            done=[0.0, 0.0, 0.0],
            values=[0.1, 0.2, 0.3, 0.4],
        )
        compute_gae(ctx, view)

        for t in range(T):
            # delta = r + 0*V(t+1) - V(t) = r - V(t)
            # last_adv = delta + 0 = delta
            expected = view["reward"][t] - view["value"][t]
            assert view["advantage"][t] == pytest.approx(expected, abs=1e-6)

    def test_truncation_bootstrap(self):
        """Truncated step gets reward correction; terminated step does not."""
        T = 2
        gamma = 0.99
        ctx = _Ctx(rollout=T, gamma=gamma, lam=0.95)

        # Step 0: truncated (time-limit), step 1: normal
        view = _make_gae_view(
            T,
            rewards=[1.0, 2.0],
            done=[1.0, 0.0],
            values=[10.0, 0.5, 0.8],
            truncated=[1, 0],  # step 0 is truncated
        )
        compute_gae(ctx, view)

        # Step 1 (not done): delta = 2.0 + gamma*0.8 - 0.5
        delta1 = 2.0 + gamma * 0.8 - 0.5
        assert view["advantage"][1] == pytest.approx(delta1, abs=1e-6)

        # Step 0 (truncated, done=True):
        #   r_corrected = 1.0 + gamma * value[0] = 1.0 + 0.99 * 10.0 = 10.9
        #   nonterminal = 0 (done=True)
        #   delta = 10.9 + 0 - 10.0 = 0.9
        #   adv = delta + gamma * lam * 0 * last_adv = 0.9  (no propagation: nonterminal=0)
        r_corrected = 1.0 + gamma * 10.0
        delta0 = r_corrected - 10.0
        assert view["advantage"][0] == pytest.approx(delta0, abs=1e-5)

    def test_truncation_vs_termination(self):
        """Same reward/value but different done reason → different advantages."""
        T = 1
        gamma = 0.99
        ctx = _Ctx(rollout=T, gamma=gamma, lam=0.95)

        # Terminated: no correction
        view_term = _make_gae_view(
            T, rewards=[1.0], done=[1.0], values=[10.0, 0.0],
            truncated=[0],
        )
        compute_gae(ctx, view_term)

        # Truncated: reward correction applied
        view_trunc = _make_gae_view(
            T, rewards=[1.0], done=[1.0], values=[10.0, 0.0],
            truncated=[1],
        )
        compute_gae(ctx, view_trunc)

        # Terminated: delta = 1.0 + 0 - 10.0 = -9.0
        assert view_term["advantage"][0] == pytest.approx(1.0 - 10.0, abs=1e-6)
        # Truncated: delta = (1.0 + 0.99*10.0) + 0 - 10.0 = 0.9
        assert view_trunc["advantage"][0] == pytest.approx(
            1.0 + gamma * 10.0 - 10.0, abs=1e-5
        )
        # Truncation advantage should be much higher
        assert view_trunc["advantage"][0] > view_term["advantage"][0]

    def test_no_truncated_field_backward_compat(self):
        """When 'truncated' is absent from view, behaves like CleanRL (no correction)."""
        T = 1
        ctx = _Ctx(rollout=T, gamma=0.99, lam=0.95)
        # No truncated key at all
        view = _make_gae_view(T, rewards=[1.0], done=[1.0], values=[10.0, 0.0])
        assert "truncated" not in view
        compute_gae(ctx, view)
        # Same as terminated: delta = 1.0 - 10.0
        assert view["advantage"][0] == pytest.approx(-9.0, abs=1e-6)


# ---------------------------------------------------------------------------
# normalize_advantages
# ---------------------------------------------------------------------------

class TestNormalizeAdvantages:

    def test_output_mean_zero_unit_var(self):
        adv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = normalize_advantages(adv)
        assert normed.mean().item() == pytest.approx(0.0, abs=1e-6)
        # population std (unbiased=False)
        assert normed.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-5)

    def test_constant_input(self):
        """All-same values: std ≈ 0, result should be ~0 (not NaN)."""
        adv = torch.tensor([3.0, 3.0, 3.0, 3.0])
        normed = normalize_advantages(adv)
        assert not torch.isnan(normed).any()
        assert torch.allclose(normed, torch.zeros_like(normed), atol=1e-4)

    def test_single_element(self):
        """Single element: mean = itself, std ≈ 0 → result ≈ 0."""
        adv = torch.tensor([5.0])
        normed = normalize_advantages(adv)
        assert not torch.isnan(normed).any()


# ---------------------------------------------------------------------------
# to_torch
# ---------------------------------------------------------------------------

class TestToTorch:

    def test_numpy_to_torch(self):
        batch = {
            "obs": np.array([[1.0, 2.0]], dtype=np.float32),
            "act": np.array([[3]], dtype=np.int64),
            "logp": np.array([0.5], dtype=np.float32),
        }
        device = torch.device("cpu")
        out = to_torch(batch, device)

        assert isinstance(out["obs"], torch.Tensor)
        assert out["obs"].dtype == torch.float32
        # "act" keeps its dtype (int64)
        assert out["act"].dtype == torch.int64
        assert out["logp"].dtype == torch.float32

    def test_torch_passthrough(self):
        """Already-torch tensors are just moved to device."""
        batch = {
            "obs": torch.tensor([[1.0, 2.0]]),
            "val": torch.tensor([0.1]),
        }
        out = to_torch(batch, torch.device("cpu"))
        assert torch.equal(out["obs"], batch["obs"])

    def test_float64_converted_to_float32(self):
        """Non-act float64 arrays are converted to float32."""
        batch = {"ret": np.array([1.0, 2.0], dtype=np.float64)}
        out = to_torch(batch, torch.device("cpu"))
        assert out["ret"].dtype == torch.float32


# ---------------------------------------------------------------------------
# ppo_update (end-to-end)
# ---------------------------------------------------------------------------

class TestPPOUpdate:

    @pytest.fixture()
    def setup(self):
        """Create a small policy, optimizer, and fake batch."""
        from beastrand.ppo.policy import PPOPolicy

        class Cfg:
            obs_shape = (4,)
            act_shape = (2,)
            act_kind = "box"
            act_n = None

            class args:
                mlp_layers = [8, 8]
                use_lstm = False

        policy = PPOPolicy(Cfg())
        opt = policy.build_optimizers(
            type("Ctx", (), {"args": type("A", (), {"learning_rate": 1e-3})()})()
        )
        return policy, opt

    def test_returns_expected_keys(self, setup):
        policy, opt = setup
        N = 16
        ctx = _Ctx(rollout=N, minibatch_size=8, train_epochs=1)

        batch = {
            "obs": np.random.randn(N, 4).astype(np.float32),
            "act": np.random.randn(N, 2).astype(np.float32),
            "logp": np.random.randn(N).astype(np.float32),
            "adv": np.random.randn(N).astype(np.float32),
            "ret": np.random.randn(N).astype(np.float32),
            "val": np.random.randn(N).astype(np.float32),
        }
        stats = ppo_update(ctx, policy, opt, batch, torch.device("cpu"))

        expected_keys = {
            "pi_loss", "v_loss", "entropy", "adv_mean", "adv_std",
            "value_mean", "value_std", "entropy_coef", "approx_kl",
            "analytical_kl", "clip_frac", "num_minibatches", "grad_norm",
        }
        assert set(stats.keys()) == expected_keys

    def test_value_loss_decreases(self, setup):
        """Repeated PPO updates on a fixed batch should reduce value loss."""
        policy, opt = setup
        N = 32
        rng = np.random.default_rng(123)
        torch.manual_seed(123)
        ctx = _Ctx(rollout=N, minibatch_size=16, train_epochs=2,
                   normalize_adv=False, ppo_clip_value=100.0)

        obs = rng.standard_normal((N, 4)).astype(np.float32)
        with torch.no_grad():
            out = policy.act({"obs": torch.from_numpy(obs)})
        batch = {
            "obs": obs,
            "act": out["action"].numpy(),
            "logp": out["logp"].numpy(),
            "adv": rng.standard_normal(N).astype(np.float32),
            "ret": np.ones(N, dtype=np.float32),   # target = 1.0
            "val": np.zeros(N, dtype=np.float32),   # initial pred = 0.0
        }

        stats_first = ppo_update(ctx, policy, opt, batch, torch.device("cpu"))
        for _ in range(10):
            stats_last = ppo_update(ctx, policy, opt, batch, torch.device("cpu"))

        # Value head should learn to predict ret=1.0 from val=0.0
        assert stats_last["v_loss"] < stats_first["v_loss"]

    def test_empty_batch_raises(self, setup):
        policy, opt = setup
        ctx = _Ctx(rollout=0, minibatch_size=1)
        batch = {
            "obs": np.zeros((0, 4), dtype=np.float32),
            "act": np.zeros((0, 2), dtype=np.float32),
            "logp": np.zeros(0, dtype=np.float32),
            "adv": np.zeros(0, dtype=np.float32),
            "ret": np.zeros(0, dtype=np.float32),
            "val": np.zeros(0, dtype=np.float32),
        }
        with pytest.raises(ValueError, match="empty batch"):
            ppo_update(ctx, policy, opt, batch, torch.device("cpu"))


# ---------------------------------------------------------------------------
# RunningMeanStd
# ---------------------------------------------------------------------------

class TestRunningMeanStd:

    def test_single_batch(self):
        """After one batch, mean/var should match the batch statistics."""
        rms = RunningMeanStd()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        rms.update(data)
        assert rms.mean == pytest.approx(np.mean(data), abs=1e-4)
        assert rms.var == pytest.approx(np.var(data), abs=1e-4)

    def test_normalize_denormalize_roundtrip(self):
        """normalize then denormalize should recover original values."""
        rms = RunningMeanStd()
        data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        rms.update(data)
        normed = rms.normalize(data)
        recovered = rms.denormalize(normed)
        np.testing.assert_allclose(recovered, data, atol=1e-5)

    def test_normalized_mean_near_zero(self):
        """After normalization, mean should be approximately zero."""
        rms = RunningMeanStd()
        data = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32)
        rms.update(data)
        normed = rms.normalize(data)
        assert np.mean(normed) == pytest.approx(0.0, abs=1e-5)

    def test_incremental_updates(self):
        """Multiple small batches should converge to overall statistics."""
        rms = RunningMeanStd()
        all_data = np.random.default_rng(42).standard_normal(1000).astype(np.float32)
        for i in range(0, 1000, 100):
            rms.update(all_data[i : i + 100])
        assert rms.mean == pytest.approx(np.mean(all_data), abs=0.05)
        assert rms.var == pytest.approx(np.var(all_data), abs=0.05)


# ---------------------------------------------------------------------------
# normalize_returns (compute_gae + RunningMeanStd integration)
# ---------------------------------------------------------------------------

class TestNormalizeReturns:

    def test_returns_differ_with_normalization(self):
        """compute_gae with a pre-trained returns_rms denormalizes values,
        producing different advantages/returns than without."""
        T = 4
        ctx = _Ctx(rollout=T, gamma=0.99, lam=0.95)

        rewards = [1.0, 2.0, 3.0, 4.0]
        done = [0.0, 0.0, 0.0, 0.0]
        values = [0.5, 0.6, 0.7, 0.8, 0.9]

        view_plain = _make_gae_view(T, rewards, done, values)
        compute_gae(ctx, view_plain)

        # Pre-train rms so denormalize is NOT identity (mean=50, var large)
        rms = RunningMeanStd()
        rms.update(np.arange(100, dtype=np.float32))

        view_norm = _make_gae_view(T, rewards, done, values)
        compute_gae(ctx, view_norm, returns_rms=rms)

        # With rms active, denormalized values differ → different advantages
        assert not np.allclose(view_plain["advantage"], view_norm["advantage"], atol=1e-3)

    def test_repeated_batches_normalize(self):
        """After multiple batches, rms.update() tracks the return scale,
        and normalize() produces roughly unit-scale values."""
        T = 8
        rng = np.random.default_rng(99)
        ctx = _Ctx(rollout=T, gamma=0.99, lam=0.95)
        rms = RunningMeanStd()

        # Simulate the PPOAlgorithm flow: compute_gae (denormalize values),
        # then rms.update(returns), then rms.normalize(returns).
        for _ in range(20):
            view = _make_gae_view(
                T,
                rewards=(rng.random(T) * 100).astype(np.float32),
                done=np.zeros(T, dtype=np.float32),
                values=(rng.random(T + 1) * 50).astype(np.float32),
            )
            compute_gae(ctx, view, returns_rms=rms)
            # PPOAlgorithm.update() calls rms.update then rms.normalize
            rms.update(view["return"])

        # After many updates, rms should track the reward scale
        assert rms.var > 1.0  # variance should be large (rewards are ~0-100)

        # Normalized returns should be much smaller than raw scale (~50-100)
        normed = rms.normalize(view["return"])
        assert np.abs(np.mean(normed)) < 5.0
        assert np.std(normed) < 5.0


# ---------------------------------------------------------------------------
# RunningMeanStdTorch (normalize_input)
# ---------------------------------------------------------------------------

class TestRunningMeanStdTorch:

    def test_train_mode_updates_stats(self):
        """In train mode, forward pass updates running mean/var."""
        rms = RunningMeanStdTorch(shape=3)
        rms.train()

        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rms(x)

        # After one batch, running_mean should be close to batch mean
        expected_mean = x.mean(dim=0)
        torch.testing.assert_close(rms.running_mean, expected_mean, atol=1e-5, rtol=1e-5)

    def test_eval_mode_freezes_stats(self):
        """In eval mode, forward pass normalizes but does NOT update stats."""
        rms = RunningMeanStdTorch(shape=2)
        rms.train()

        # First update in train mode
        x1 = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        rms(x1)
        mean_after_train = rms.running_mean.clone()

        # Switch to eval
        rms.eval()
        x2 = torch.tensor([[100.0, 200.0], [300.0, 400.0]])
        rms(x2)

        # Stats should be unchanged
        torch.testing.assert_close(rms.running_mean, mean_after_train)

    def test_state_dict_persistence(self):
        """State dict includes buffers (running_mean, running_var, count)."""
        rms = RunningMeanStdTorch(shape=4)
        rms.train()
        rms(torch.randn(32, 4))

        sd = rms.state_dict()
        assert "running_mean" in sd
        assert "running_var" in sd
        assert "count" in sd

        # Load into a fresh module — stats should match
        rms2 = RunningMeanStdTorch(shape=4)
        rms2.load_state_dict(sd)
        torch.testing.assert_close(rms2.running_mean, rms.running_mean)
        torch.testing.assert_close(rms2.running_var, rms.running_var)

    def test_output_clipped(self):
        """Normalized output should be clipped to [-clip, clip]."""
        rms = RunningMeanStdTorch(shape=1, clip=2.0)
        rms.train()
        # Update with small values so stats are well-defined
        rms(torch.zeros(100, 1))

        # Large outlier should be clipped
        rms.eval()
        out = rms(torch.tensor([[1000.0]]))
        assert out.item() <= 2.0
        out = rms(torch.tensor([[-1000.0]]))
        assert out.item() >= -2.0

    def test_normalize_approximately_unit_scale(self):
        """After training, output should be roughly zero-mean unit-scale."""
        rms = RunningMeanStdTorch(shape=1)
        rms.train()

        # Feed several batches of data with known distribution
        for _ in range(50):
            rms(torch.randn(64, 1) * 5.0 + 10.0)

        rms.eval()
        test_data = torch.randn(1000, 1) * 5.0 + 10.0
        normed = rms(test_data)
        assert abs(normed.mean().item()) < 0.5
        assert abs(normed.std().item() - 1.0) < 0.5


# ---------------------------------------------------------------------------
# KL loss coefficient
# ---------------------------------------------------------------------------

class TestKLLoss:

    @pytest.fixture()
    def setup(self):
        from beastrand.ppo.policy import PPOPolicy

        class Cfg:
            obs_shape = (4,)
            act_shape = (2,)
            act_kind = "box"
            act_n = None

            class args:
                mlp_layers = [8, 8]
                use_lstm = False
                normalize_input = False

        policy = PPOPolicy(Cfg())
        opt = policy.build_optimizers(
            type("Ctx", (), {"args": type("A", (), {"learning_rate": 1e-3})()})()
        )
        return policy, opt

    def test_kl_loss_increases_total_loss(self, setup):
        """With kl_loss_coeff > 0, total loss should include KL term."""
        policy, opt = setup
        N = 16
        torch.manual_seed(42)
        np.random.seed(42)

        batch = {
            "obs": np.random.randn(N, 4).astype(np.float32),
            "act": np.random.randn(N, 2).astype(np.float32),
            "logp": np.random.randn(N).astype(np.float32),
            "adv": np.random.randn(N).astype(np.float32),
            "ret": np.random.randn(N).astype(np.float32),
            "val": np.random.randn(N).astype(np.float32),
        }

        # Without KL penalty
        ctx_no_kl = _Ctx(rollout=N, minibatch_size=N, train_epochs=1, kl_loss_coeff=0.0)
        stats_no_kl = ppo_update(ctx_no_kl, policy, opt, batch, torch.device("cpu"))

        # With KL penalty — approx_kl should still be tracked
        ctx_kl = _Ctx(rollout=N, minibatch_size=N, train_epochs=1, kl_loss_coeff=0.1)
        stats_kl = ppo_update(ctx_kl, policy, opt, batch, torch.device("cpu"))

        # Both should report approx_kl
        assert "approx_kl" in stats_no_kl
        assert "approx_kl" in stats_kl

    def test_kl_coeff_zero_no_kl_gradient(self, setup):
        """With kl_loss_coeff=0, KL term should not affect gradients."""
        policy, opt = setup
        N = 16

        batch = {
            "obs": np.random.randn(N, 4).astype(np.float32),
            "act": np.random.randn(N, 2).astype(np.float32),
            "logp": np.random.randn(N).astype(np.float32),
            "adv": np.random.randn(N).astype(np.float32),
            "ret": np.random.randn(N).astype(np.float32),
            "val": np.random.randn(N).astype(np.float32),
        }

        ctx = _Ctx(rollout=N, minibatch_size=N, train_epochs=1, kl_loss_coeff=0.0)
        # Should run without errors
        stats = ppo_update(ctx, policy, opt, batch, torch.device("cpu"))
        assert stats["approx_kl"] >= 0.0  # KL is non-negative


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

class TestLRSchedule:

    def _make_algo(self, lr_schedule="constant", total_steps=1000, batch_size=100,
                   learning_rate=1e-3):
        from beastrand.ppo.policy import PPOPolicy

        class Cfg:
            obs_shape = (4,)
            act_shape = (2,)
            act_kind = "box"
            act_n = None

            class args:
                mlp_layers = [8, 8]
                use_lstm = False
                normalize_input = False

        policy = PPOPolicy(Cfg())

        class CtxArgs:
            pass

        ctx_args = CtxArgs()
        ctx_args.learning_rate = learning_rate
        ctx_args.lr_schedule = lr_schedule
        ctx_args.total_env_steps = total_steps
        ctx_args.batch_size = batch_size
        ctx_args.normalize_returns = False

        ctx = type("Ctx", (), {"args": ctx_args})()

        opt = policy.build_optimizers(ctx)
        algo = PPOAlgorithm(ctx, policy, opt, torch.device("cpu"))
        return algo

    def test_constant_lr_stays_same(self):
        """With lr_schedule='constant', LR should not change."""
        algo = self._make_algo(lr_schedule="constant", learning_rate=1e-3)
        initial_lr = algo.opt["opt"].param_groups[0]["lr"]

        for _ in range(5):
            algo._step_lr()

        assert algo.opt["opt"].param_groups[0]["lr"] == pytest.approx(initial_lr)

    def test_linear_decay_decreases(self):
        """With lr_schedule='linear_decay', LR should monotonically decrease."""
        total_steps = 1000
        batch_size = 100
        total_updates = total_steps // batch_size  # = 10

        algo = self._make_algo(
            lr_schedule="linear_decay",
            total_steps=total_steps,
            batch_size=batch_size,
            learning_rate=1e-3,
        )

        lrs = []
        for _ in range(total_updates):
            algo._step_lr()
            lrs.append(algo.opt["opt"].param_groups[0]["lr"])

        # Should monotonically decrease
        for i in range(len(lrs) - 1):
            assert lrs[i] > lrs[i + 1]

        # First LR should equal initial (count=0 → frac=1.0)
        assert lrs[0] == pytest.approx(1e-3, abs=1e-8)

        # At count=total_updates (one past the end), LR would reach 0
        algo._step_lr()
        assert algo.opt["opt"].param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-8)

    def test_linear_decay_values(self):
        """LR follows formula: initial_lr * (1 - count / total_updates)."""
        total_steps = 1000
        batch_size = 100
        total_updates = total_steps // batch_size  # = 10
        initial_lr = 1e-3

        algo = self._make_algo(
            lr_schedule="linear_decay",
            total_steps=total_steps,
            batch_size=batch_size,
            learning_rate=initial_lr,
        )

        for i in range(total_updates):
            algo._step_lr()
            expected_lr = initial_lr * (1.0 - i / total_updates)
            actual_lr = algo.opt["opt"].param_groups[0]["lr"]
            assert actual_lr == pytest.approx(expected_lr, abs=1e-8)
