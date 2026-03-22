"""
PackedSequence utilities for LSTM BPTT training.

Splits trajectories at episode boundaries into variable-length sub-sequences
and builds a PackedSequence for efficient batched LSTM processing.

Adapted from Sample Factory's rnn_utils.py (Alex Petrenko, MIT License).
"""

from typing import Tuple

import torch
from torch.nn.utils.rnn import PackedSequence, invert_permutation


def _build_pack_info_from_dones(
    dones: torch.Tensor, T: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build indexing info for PackedSequence from done flags.

    Splits a flat (N*T,) tensor into variable-length sub-sequences at every
    done=1 boundary and at every rollout boundary (every T steps).

    Returns:
        rollout_starts: (num_seqs,) start indices of each sub-sequence
        is_new_episode: (N*T,) rolled done flags (1 = first step of new episode)
        select_inds: (N*T,) reorder indices for PackedSequence data
        batch_sizes: (max_len,) PackedSequence batch_sizes
        sorted_indices: (num_seqs,) permutation for sorted order
    """
    num_samples = len(dones)

    # Mark boundaries: done=1 OR end of each rollout
    boundaries = dones.clone().detach()
    boundaries[T - 1 :: T] = 1
    boundaries = boundaries.nonzero(as_tuple=False).squeeze(dim=1) + 1

    first_len = boundaries[0].unsqueeze(0)
    if len(boundaries) <= 1:
        lengths = first_len
    else:
        lengths = boundaries[1:] - boundaries[:-1]
        lengths = torch.cat([first_len, lengths])

    starts = boundaries - lengths

    # Rolled dones: identify first frames of new episodes
    is_new_episode = dones.clone().detach().view(-1, T)
    is_new_episode = is_new_episode.roll(1, 1)
    is_new_episode[:, 0] = 0  # first step of each rollout is NOT a new episode flag
    is_new_episode = is_new_episode.view(-1)

    # Sort by length (descending) for PackedSequence
    lengths_sorted, sorted_indices = torch.sort(lengths, descending=True)
    cpu_lengths = lengths_sorted.to(device="cpu", non_blocking=True)

    starts_sorted = starts.index_select(0, sorted_indices)

    # Build select_inds and batch_sizes
    select_inds = torch.empty(num_samples, device=dones.device, dtype=torch.int64)
    max_length = int(cpu_lengths[0].item())
    batch_sizes = torch.empty(max_length, device="cpu", dtype=torch.int64)

    offset = 0
    prev_len = 0
    num_valid = lengths_sorted.size(0)

    for unique_len in torch.unique_consecutive(cpu_lengths).flip(0):
        valids = lengths_sorted[:num_valid] > prev_len
        num_valid = int(valids.float().sum().item())
        next_len = int(unique_len.item())

        batch_sizes[prev_len:next_len] = num_valid

        new_inds = (
            starts_sorted[:num_valid].view(1, num_valid)
            + torch.arange(prev_len, next_len, device=starts_sorted.device).view(next_len - prev_len, 1)
        ).view(-1)
        select_inds[offset : offset + new_inds.numel()] = new_inds
        offset += new_inds.numel()
        prev_len = next_len

    assert offset == num_samples
    return starts, is_new_episode, select_inds, batch_sizes, sorted_indices


def build_rnn_inputs(
    x: torch.Tensor,
    dones: torch.Tensor,
    rnn_states: torch.Tensor,
    T: int,
) -> Tuple[PackedSequence, torch.Tensor, torch.Tensor]:
    """Build PackedSequence input for LSTM from flat tensors.

    Args:
        x: (N*T, feat_dim) — encoder output (body output for all timesteps)
        dones: (N*T,) — done flags (1.0 at episode boundaries). Can be CPU tensor.
        rnn_states: (N*T, state_dim) — per-step rnn states (h+c concatenated)
        T: rollout length

    Returns:
        x_seq: PackedSequence for LSTM input
        rnn_h0: (num_seqs, state_dim) — initial states per sub-sequence,
                zeroed for new episode starts
        inv_inds: (N*T,) — indices to unpack LSTM output back to original order
    """
    starts, is_new_episode, select_inds, batch_sizes, sorted_indices = (
        _build_pack_info_from_dones(dones, T)
    )
    inv_inds = invert_permutation(select_inds)

    # Move indices to same device as x
    dev = x.device
    select_inds = select_inds.to(dev)
    inv_inds = inv_inds.to(dev)
    sorted_indices = sorted_indices.to(dev)
    starts = starts.to(dev)
    is_new_episode = is_new_episode.to(dev)

    # Build PackedSequence (data reordered by select_inds, sorted by length)
    x_seq = PackedSequence(x.index_select(0, select_inds), batch_sizes, sorted_indices)

    # Extract h0 for each sub-sequence; zero for new episode starts
    rnn_h0 = rnn_states.to(dev).index_select(0, starts)
    is_same_episode = (1 - is_new_episode.view(-1, 1)).index_select(0, starts)
    rnn_h0 = rnn_h0 * is_same_episode

    return x_seq, rnn_h0, inv_inds


def unpack_rnn_outputs(output_seq: PackedSequence, inv_inds: torch.Tensor) -> torch.Tensor:
    """Unpack LSTM PackedSequence output back to original (N*T, hidden) order."""
    return output_seq.data.index_select(0, inv_inds)
