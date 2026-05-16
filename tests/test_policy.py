import torch
import pytest

from beastrand.core.base_policy import BasePolicy
from beastrand.ppo.policy import PPOPolicy
from projects.ppo_lstm.policy import PPOLSTMPolicy
from projects.ppo_lstm.data_record import PPOLSTMDataRecord


class DummyCfg:
    def __init__(self, use_lstm: bool = False, obs_shape=(4,), normalize_input: bool = False):
        self.obs_shape = obs_shape
        self.act_shape = (2,)
        self.act_kind = "box"
        self.act_n = None
        args = type("Args", (), {})()
        args.mlp_layers = [8, 8]
        args.use_lstm = use_lstm
        args.lstm_hidden_size = 4
        args.normalize_input = normalize_input
        self.args = args


def test_ppo_policy_dict_api():
    cfg = DummyCfg()
    policy = PPOPolicy(cfg)
    obs = torch.zeros(1, 4)
    out = policy.act({"obs": obs})
    assert out["action"].shape == (1, 2)
    eval_out = policy.evaluate_actions({"obs": obs, "action": out["action"]})
    assert eval_out["value"].shape == (1,)
    v = policy.value({"obs": obs})
    assert v.shape == (1,)


def test_ppo_lstm_policy_state():
    cfg = DummyCfg(use_lstm=True)
    policy = PPOLSTMPolicy(cfg)
    obs = torch.zeros(1, 4)
    state = policy.initial_state(1)
    out = policy.act({"obs": obs, "rnn_state": state})
    assert "rnn_state" in out
    h, c = out["rnn_state"]
    assert h.shape[0] == 1 and c.shape[0] == 1


def test_ppo_lstm_data_record_batch():
    np = pytest.importorskip("numpy")

    class Ctx:
        class Args:
            lstm_hidden_size = 4

        args = Args()

    ctx = Ctx()
    T = 2
    obs_shape = (4,)
    act_shape = (2,)
    specs = PPOLSTMDataRecord.alloc_specs(ctx, T, obs_shape, act_shape)
    assert specs["rnn_state_h"][0] == (T + 1, ctx.args.lstm_hidden_size)
    assert specs["rnn_state_c"][0] == (T + 1, ctx.args.lstm_hidden_size)
    assert specs["mask"][0] == (T + 1,)
    view = {}
    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "float16": np.float16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
    }
    for field, (shape, dtype_str) in specs.items():
        view[field] = np.zeros(shape, dtype=dtype_map.get(dtype_str, np.float32))

    batch = PPOLSTMDataRecord.build_batch(ctx, view)
    assert batch["rnn_state_h"].shape == (T, ctx.args.lstm_hidden_size)
    assert batch["rnn_state_c"].shape == (T, ctx.args.lstm_hidden_size)
    assert batch["mask"].shape == (T,)


class _NormPolicy(BasePolicy):
    """Minimal BasePolicy subclass for exercising normalize_obs in isolation."""

    def act(self, inputs, deterministic=False):
        return {"action": self.normalize_obs(inputs["obs"])}


def test_normalize_obs_flat_updates_running_stats():
    cfg = DummyCfg(obs_shape=(4,), normalize_input=True)
    policy = _NormPolicy(cfg)
    policy.train()

    obs = torch.randn(32, 4) * 3.0 + 5.0
    out = policy.normalize_obs(obs)
    assert out.shape == obs.shape

    mean = policy.obs_normalizer.running_mean
    assert mean.shape == (4,)
    assert torch.allclose(mean, obs.mean(dim=0), atol=1e-4)


def test_normalize_obs_multidim_preserves_shape():
    cfg = DummyCfg(obs_shape=(3, 4, 4), normalize_input=True)
    policy = _NormPolicy(cfg)
    policy.train()

    obs = torch.randn(8, 3, 4, 4)
    out = policy.normalize_obs(obs)
    assert out.shape == obs.shape

    mean = policy.obs_normalizer.running_mean
    assert mean.shape == (3 * 4 * 4,)
    assert torch.allclose(mean, obs.reshape(8, -1).mean(dim=0), atol=1e-4)


class _ImagePolicy(BasePolicy):
    """Tiny image-observation policy used to exercise normalize_obs end-to-end
    on a (B, C, H, W) forward path. Mirrors what an Atari-style policy does:
    keep spatial shape through normalization, then feed Conv2d."""

    def __init__(self, cfg):
        super().__init__(cfg)
        c, _, _ = cfg.obs_shape
        self.conv = torch.nn.Conv2d(c, 4, kernel_size=3, padding=1)
        self.head = torch.nn.Linear(4 * cfg.obs_shape[1] * cfg.obs_shape[2], cfg.act_shape[0])

    def act(self, inputs, deterministic=False):
        x = self.normalize_obs(inputs["obs"])
        x = self.conv(x).flatten(1)
        return {"action": self.head(x)}


def test_normalize_input_end_to_end_multidim():
    cfg = DummyCfg(obs_shape=(3, 4, 4), normalize_input=True)
    policy = _ImagePolicy(cfg)
    policy.train()

    obs = torch.randn(8, 3, 4, 4) * 2.0 + 1.0
    out = policy.act({"obs": obs})
    assert out["action"].shape == (8, 2)

    assert policy.obs_normalizer.running_mean.shape == (3 * 4 * 4,)
    assert policy.obs_normalizer.count.item() >= 8


def test_normalize_obs_disabled_passthrough():
    cfg = DummyCfg(obs_shape=(4,), normalize_input=False)
    policy = _NormPolicy(cfg)
    assert policy.obs_normalizer is None
    obs = torch.randn(2, 4)
    assert torch.equal(policy.normalize_obs(obs), obs)


def test_ppo_policy_with_normalize_input():
    cfg = DummyCfg(obs_shape=(4,), normalize_input=True)
    policy = PPOPolicy(cfg)
    obs = torch.randn(4, 4)
    out = policy.act({"obs": obs})
    assert out["action"].shape == (4, 2)
    assert policy.obs_normalizer is not None
    assert policy.obs_normalizer.running_mean.shape == (4,)


def test_atari_preprocess_rescales_float32_pixels():
    """The rollout worker stores Atari obs as float32 [0, 255]. The policy
    must rescale to [0, 1] regardless of dtype — the prior dtype-gated branch
    silently fed raw 0-255 pixels into the CNN."""
    from projects.atari.policy import AtariPolicy

    class _AtariCfg:
        def __init__(self):
            self.obs_shape = (4, 84, 84)
            self.act_shape = ()
            self.act_kind = "discrete"
            self.act_n = 6
            args = type("Args", (), {})()
            args.normalize_input = False
            self.args = args

    policy = AtariPolicy(_AtariCfg())
    # Worker-style storage: float32 in [0, 255].
    obs = torch.full((1, 4, 84, 84), 255.0, dtype=torch.float32)
    out = policy._preprocess(obs)
    assert out.max().item() <= 1.0
    assert out.min().item() >= 0.0
    assert torch.allclose(out, torch.ones_like(out))

    # uint8 path still works (it's just a different upstream encoding).
    obs_u8 = torch.full((1, 4, 84, 84), 128, dtype=torch.uint8)
    out_u8 = policy._preprocess(obs_u8)
    assert torch.allclose(out_u8, torch.full_like(out_u8, 128.0 / 255.0))
