"""Tests for core PPO framework utilities: GAE, advantage normalization, ppo_update, to_torch."""

import numpy as np
import pytest
import torch

from modules.algos.ppo import compute_gae, normalize_advantages, ppo_update
from utils.tensor_utils import to_torch


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
        from modules.policy.ppo_policy import PPOPolicy

        class Cfg:
            obs_shape = (4,)
            act_shape = (2,)

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
            "clip_frac", "num_minibatches",
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
