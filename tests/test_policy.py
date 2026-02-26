import torch
import pytest

from modules.policy.ppo_policy import PPOPolicy
from modules.policy.ppo_lstm_policy import PPOLSTMPolicy
from modules.dataset.data_record.ppo_lstm_data_record import PPOLSTMDataRecord


class DummyCfg:
    def __init__(self, use_lstm: bool = False):
        self.obs_shape = (4,)
        self.act_shape = (2,)
        args = type("Args", (), {})()
        args.mlp_layers = [8, 8]
        args.use_lstm = use_lstm
        args.lstm_hidden_size = 4
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
