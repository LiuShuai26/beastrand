"""Tests for ParameterServer / ParameterClient weight sharing."""

import multiprocessing as mp
import threading

import torch
import torch.nn as nn

from beastrand.utils.model_sharing import ParameterServer, ParameterClient


def _make_policy():
    """Simple 2-layer MLP for testing."""
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


class TestParameterServer:

    def test_update_bumps_version(self):
        policy = _make_policy()
        version = torch.tensor(0, dtype=torch.int64).share_memory_()
        ps = ParameterServer(policy, torch.device("cpu"), version)

        assert version.item() == 0
        ps.update(policy)
        assert version.item() == 1
        ps.update(policy)
        assert version.item() == 2

    def test_shared_state_matches_policy(self):
        policy = _make_policy()
        version = torch.tensor(0, dtype=torch.int64).share_memory_()
        ps = ParameterServer(policy, torch.device("cpu"), version)

        # Shared state should match policy after construction
        for name, param in policy.named_parameters():
            assert name in ps.shared_state
            torch.testing.assert_close(ps.shared_state[name], param.detach().cpu())

    def test_update_copies_new_weights(self):
        policy = _make_policy()
        version = torch.tensor(0, dtype=torch.int64).share_memory_()
        ps = ParameterServer(policy, torch.device("cpu"), version)

        # Modify policy weights
        with torch.no_grad():
            for p in policy.parameters():
                p.fill_(42.0)

        ps.update(policy)

        # Shared state should now be 42.0
        for name, param in policy.named_parameters():
            torch.testing.assert_close(ps.shared_state[name], param.detach().cpu())
            assert (ps.shared_state[name] == 42.0).all()


class TestParameterClient:

    def test_ensure_updated_loads_weights(self):
        policy = _make_policy()
        version = torch.tensor(0, dtype=torch.int64).share_memory_()
        ps = ParameterServer(policy, torch.device("cpu"), version)

        # Create a separate policy (client side)
        client_policy = _make_policy()
        pc = ParameterClient(ps)

        # Modify server-side policy and update
        with torch.no_grad():
            for p in policy.parameters():
                p.fill_(99.0)
        ps.update(policy)

        # Client should detect version change and load
        updated = pc.ensure_updated(client_policy)
        assert updated is True
        for name, param in client_policy.named_parameters():
            assert (param == 99.0).all()

    def test_ensure_updated_skips_when_current(self):
        policy = _make_policy()
        version = torch.tensor(0, dtype=torch.int64).share_memory_()
        ps = ParameterServer(policy, torch.device("cpu"), version)

        client_policy = _make_policy()
        pc = ParameterClient(ps)

        ps.update(policy)
        pc.ensure_updated(client_policy)

        # No new update — should return False
        updated = pc.ensure_updated(client_policy)
        assert updated is False

    def test_multiple_updates_tracked(self):
        policy = _make_policy()
        version = torch.tensor(0, dtype=torch.int64).share_memory_()
        ps = ParameterServer(policy, torch.device("cpu"), version)

        client_policy = _make_policy()
        pc = ParameterClient(ps)

        # Multiple server updates before client sync
        for val in [1.0, 2.0, 3.0]:
            with torch.no_grad():
                for p in policy.parameters():
                    p.fill_(val)
            ps.update(policy)

        # Client should get latest (3.0)
        pc.ensure_updated(client_policy)
        for name, param in client_policy.named_parameters():
            assert (param == 3.0).all()
        assert pc._local_version == 3

    def test_buffers_synced(self):
        """Buffers (e.g., obs normalizer stats) should also sync."""
        class PolicyWithBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)
                self.register_buffer("running_mean", torch.zeros(4))

            def forward(self, x):
                return self.linear(x)

        policy = PolicyWithBuffer()
        version = torch.tensor(0, dtype=torch.int64).share_memory_()
        ps = ParameterServer(policy, torch.device("cpu"), version)

        # Modify buffer
        policy.running_mean.fill_(7.0)
        ps.update(policy)

        client_policy = PolicyWithBuffer()
        pc = ParameterClient(ps)
        pc.ensure_updated(client_policy)

        torch.testing.assert_close(client_policy.running_mean, torch.full((4,), 7.0))


class TestConcurrency:

    def test_concurrent_update_and_read_no_tearing(self):
        """Simulate learner updating while inference server reads.

        With the lock, client should always see a consistent snapshot:
        all parameters set to the same value (not a mix of old and new).
        """
        # Use a larger model to increase the chance of catching tearing
        policy = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64),
        )
        version = torch.tensor(0, dtype=torch.int64).share_memory_()
        ps = ParameterServer(policy, torch.device("cpu"), version)

        client_policy = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64),
        )
        pc = ParameterClient(ps)

        num_updates = 200
        errors = []

        def writer():
            """Simulates learner: rapidly update weights with distinct values."""
            for i in range(1, num_updates + 1):
                with torch.no_grad():
                    for p in policy.parameters():
                        p.fill_(float(i))
                ps.update(policy)

        def reader():
            """Simulates inference server: read weights and check consistency."""
            for _ in range(num_updates * 2):
                if version.item() == 0:
                    continue  # writer hasn't started yet
                pc._local_version = -1  # force reload every time
                pc.ensure_updated(client_policy)

                # All parameters should have the same fill value (no tearing)
                vals = set()
                for p in client_policy.parameters():
                    vals.add(p.flatten()[0].item())
                if len(vals) > 1:
                    errors.append(f"Tearing detected: saw values {vals}")
                    return

        t_writer = threading.Thread(target=writer)
        t_reader = threading.Thread(target=reader)
        t_writer.start()
        t_reader.start()
        t_writer.join()
        t_reader.join()

        assert not errors, errors[0]

    def test_multiple_clients_concurrent(self):
        """Multiple inference servers reading concurrently from one ParameterServer."""
        policy = _make_policy()
        version = torch.tensor(0, dtype=torch.int64).share_memory_()
        ps = ParameterServer(policy, torch.device("cpu"), version)

        num_clients = 4
        num_rounds = 50
        errors = []

        def client_loop(client_id):
            local_policy = _make_policy()
            pc = ParameterClient(ps)
            for _ in range(num_rounds):
                pc.ensure_updated(local_policy)
                # Verify weights are valid (not NaN or torn)
                for p in local_policy.parameters():
                    if torch.isnan(p).any():
                        errors.append(f"Client {client_id}: NaN in weights")
                        return

        # Writer: update policy in parallel
        def writer():
            for i in range(1, num_rounds + 1):
                with torch.no_grad():
                    for p in policy.parameters():
                        p.fill_(float(i))
                ps.update(policy)

        threads = [threading.Thread(target=writer)]
        for i in range(num_clients):
            threads.append(threading.Thread(target=client_loop, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors[0]

    def test_version_monotonically_increases(self):
        """Version counter should only increase, never decrease or skip."""
        policy = _make_policy()
        version = torch.tensor(0, dtype=torch.int64).share_memory_()
        ps = ParameterServer(policy, torch.device("cpu"), version)

        observed_versions = []

        def writer():
            for _ in range(100):
                v = ps.update(policy)
                observed_versions.append(v)

        t = threading.Thread(target=writer)
        t.start()
        t.join()

        # Should be 1, 2, 3, ..., 100
        assert observed_versions == list(range(1, 101))


def _mp_writer(ps, num_updates):
    """Writer process: fill all params with sequential values."""
    # Reconstruct a local policy with same architecture
    policy = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
    # Load current shared state to get matching keys
    from beastrand.utils.model_sharing import _load_state_into_model
    _load_state_into_model(ps.shared_state, policy)

    for i in range(1, num_updates + 1):
        with torch.no_grad():
            for p in policy.parameters():
                p.fill_(float(i))
        ps.update(policy)


def _mp_version_child(ps, result):
    """Child process: update 5 times and record final version."""
    policy = _make_policy()
    from beastrand.utils.model_sharing import _load_state_into_model
    _load_state_into_model(ps.shared_state, policy)
    for _ in range(5):
        ps.update(policy)
    result.value = int(ps.policy_version.item())


def _mp_reader(ps, num_checks, error_flag):
    """Reader process: check for tearing across shared memory."""
    policy = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
    pc = ParameterClient(ps)

    for _ in range(num_checks):
        if ps.policy_version.item() == 0:
            continue
        pc._local_version = -1  # force reload
        pc.ensure_updated(policy)

        vals = set()
        for p in policy.parameters():
            vals.add(p.flatten()[0].item())
        if len(vals) > 1:
            error_flag.value = 1
            return


class TestCrossProcess:

    def test_mp_concurrent_no_tearing(self):
        """Cross-process: writer and reader via mp.Process with shared memory tensors."""
        # Build ParameterServer in the main process
        policy = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        version = torch.tensor(0, dtype=torch.int64).share_memory_()
        ps = ParameterServer(policy, torch.device("cpu"), version)

        num_updates = 200
        error_flag = mp.Value("i", 0)

        writer = mp.Process(target=_mp_writer, args=(ps, num_updates))
        reader = mp.Process(target=_mp_reader, args=(ps, num_updates * 2, error_flag))

        writer.start()
        reader.start()
        writer.join(timeout=15)
        reader.join(timeout=15)

        assert writer.exitcode == 0, f"Writer exited with {writer.exitcode}"
        assert reader.exitcode == 0, f"Reader exited with {reader.exitcode}"
        assert error_flag.value == 0, "Tearing detected across processes"
        assert version.item() == num_updates

    def test_mp_version_visible_across_processes(self):
        """Version counter updated in one process is visible in another."""
        policy = _make_policy()
        version = torch.tensor(0, dtype=torch.int64).share_memory_()
        ps = ParameterServer(policy, torch.device("cpu"), version)

        result = mp.Value("i", -1)

        p = mp.Process(target=_mp_version_child, args=(ps, result))
        p.start()
        p.join(timeout=10)

        assert p.exitcode == 0
        # Both parent and child should see version=5
        assert result.value == 5
        assert version.item() == 5
