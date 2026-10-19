# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug lag-major convolution using the runner's ragged recurrent-state primitive."""


import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.ragged_conv1d_jax import ragged_conv1d
from tpu_inference.layers.common.sharding import ShardingAxisName


def grug_short_conv_local(inputs, state, weight, starts, indices, distribution, lengths):
    """Return convolved tokens and updated oldest-to-newest history for one data shard."""
    query_lengths = starts[1:] - starts[:-1]
    valid = jnp.arange(indices.shape[0]) < distribution[2]
    has_history = lengths > query_lengths
    # The runner classifies one-token fresh prompts as decode. The existing
    # decode primitive does not mask initial state, so clear those reused slots.
    previous = state[indices]
    state = state.at[indices].set(jnp.where((valid & ~has_history)[:, None, None], 0, previous))
    return ragged_conv1d(
        inputs,
        state,
        weight.T[:, None, ::-1],
        None,
        starts,
        indices,
        distribution,
        has_history,
        kernel_size=weight.shape[0],
    )


def grug_short_conv(inputs, state, weight, starts, indices, distribution, lengths, mesh):
    """Apply independent request histories on the existing attention-data mesh."""
    data = ShardingAxisName.ATTN_DATA
    mapped = jax.shard_map(
        grug_short_conv_local,
        mesh=mesh,
        in_specs=(P(data, None), P(data, None, None), P(None, None), P(data), P(data), P(data), P(data)),
        out_specs=(P(data, None), P(data, None, None)),
        check_vma=False,
    )
    return mapped(inputs, state, weight, starts, indices, distribution, lengths)
