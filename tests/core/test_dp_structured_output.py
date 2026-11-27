# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import xgrammar
from jax.sharding import Mesh
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.sched.output import CachedRequestData, GrammarOutput, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

from tpu_inference.core.sched.dp_scheduler import DPScheduler, SchedulerCommand
from tpu_inference.runner.structured_decoding_manager import StructuredDecodingManager


VOCAB = ["a", "b", "c", "d"] + [f"<{i}>" for i in range(4, 32)]


class NativeGrammarManager(StructuredOutputManager):
    """Use native mask generation with a tiny, weight-free tokenizer."""

    def grammar_init(self, request):
        info = xgrammar.TokenizerInfo(VOCAB, vocab_type=xgrammar.VocabType.RAW,
                                     stop_token_ids=[31])
        compiler = xgrammar.GrammarCompiler(info, max_threads=1)
        compiled = compiler.compile_grammar(
            request.sampling_params.structured_outputs.grammar)
        request.structured_output_request.grammar = XgrammarGrammar(
            vocab_size=32, ctx=compiled,
            matcher=xgrammar.GrammarMatcher(compiled, max_rollback_tokens=2))
        self.backend = SimpleNamespace(
            allocate_token_bitmask=lambda batch: xgrammar.allocate_token_bitmask(batch, 32))


class FrameScheduler:
    """Deterministic frames around the real vLLM grammar implementation."""

    get_grammar_bitmask = Scheduler.get_grammar_bitmask

    def __init__(self, structured_output_manager, **kwargs):
        self.structured_output_manager = structured_output_manager
        self.requests = {}

    def add_request(self, request):
        self.requests[request.request_id] = request

    def schedule(self, request_ids):
        selected = [rid for rid in request_ids if rid in self.requests]
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=selected, resumed_req_ids=[], new_token_ids=[],
                all_token_ids={}, new_block_ids=[],
                num_computed_tokens=[1] * len(selected),
                num_output_tokens=[0] * len(selected)),
            num_scheduled_tokens=dict.fromkeys(selected, 1),
            total_num_scheduled_tokens=len(selected),
            scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
            num_common_prefix_blocks=[], finished_req_ids=set(),
            free_encoder_mm_hashes=[], has_structured_output_requests=bool(selected))

    def update_from_output(self, scheduler_output, model_runner_output):
        for rid in scheduler_output.num_scheduled_tokens:
            tokens = model_runner_output.sampled_token_ids[
                model_runner_output.req_id_to_index[rid]]
            grammar = self.requests[rid].structured_output_request.grammar
            if not grammar.accept_tokens(rid, tokens):
                raise ValueError(f"Grammar rejected sampled tokens for {rid}: {tokens}")
        return {}

    def shutdown(self):
        pass


@contextmanager
def grammar_scheduler(requests):
    torch.set_num_threads(1)
    config = SimpleNamespace(
        sharding_config=SimpleNamespace(total_dp_size=2),
        scheduler_config=SimpleNamespace(
            _original_scheduler_cls=FrameScheduler, max_num_seqs=8,
            max_num_batched_tokens=32, async_scheduling=True),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(distributed_executor_backend="external_launcher"),
        model_config=SimpleNamespace(skip_tokenizer_init=True, is_diffusion=False),
        structured_outputs_config=SimpleNamespace(enable_in_reasoning=True),
        num_speculative_tokens=0, additional_config={})
    scheduler = DPScheduler(config, SimpleNamespace(num_blocks=16),
                            NativeGrammarManager(config), 16)
    scheduler._batch_prefills = False
    try:
        for rank, rid, text in requests:
            request = Request(rid, [4], SamplingParams(
                max_tokens=4, structured_outputs=StructuredOutputsParams(
                    grammar=f'root ::= "{text}"')), None)
            request.structured_output_request.reasoning_ended = True
            scheduler.assigned_dp_rank[rid] = rank
            scheduler._send_command(rank, SchedulerCommand.ADD_REQUEST, request)
            scheduler._get_result(rank, SchedulerCommand.ADD_REQUEST)
        yield scheduler
    finally:
        scheduler.shutdown()


def allowed_tokens(mask):
    return [token for token in range(32)
            if (int(mask[token // 32]) >> (token % 32)) & 1]


def test_rank_masks_preserve_request_order_and_numpy_contract():
    with grammar_scheduler([(0, "zulu", "ab"), (1, "alpha", "cd")]) as scheduler:
        frame = scheduler.schedule(["zulu", "alpha"])
        output = scheduler.get_grammar_bitmask(frame)
        assert output.structured_output_request_ids == ["zulu", "alpha"]
        assert isinstance(output.grammar_bitmask, np.ndarray)
        assert output.grammar_bitmask.dtype == np.int32
        assert [allowed_tokens(row) for row in output.grammar_bitmask] == [[0], [2]]


def test_grammar_mask_uses_requested_frame_with_two_queued_steps():
    with grammar_scheduler([(0, "first", "ab"), (0, "later", "cd")]) as scheduler:
        first = scheduler.schedule(["first"])
        later = scheduler.schedule(["later"])
        first_mask = scheduler.get_grammar_bitmask(first)
        assert first_mask.structured_output_request_ids == ["first"]
        assert allowed_tokens(first_mask.grammar_bitmask[0]) == [0]
        scheduler.update_from_output(first, ModelRunnerOutput(
            req_ids=["first"], req_id_to_index={"first": 0},
            sampled_token_ids=[[0]], logprobs=None, prompt_logprobs_dict={}))
        later_mask = scheduler.get_grammar_bitmask(later)
        assert later_mask.structured_output_request_ids == ["later"]
        assert allowed_tokens(later_mask.grammar_bitmask[0]) == [2]
        next_frame = scheduler.schedule(["first"])
        advanced = scheduler.get_grammar_bitmask(next_frame)
        assert allowed_tokens(advanced.grammar_bitmask[0]) == [1]


def decoding_manager(request_ids):
    runner = SimpleNamespace(
        vocab_size=32, mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
        grammar_bitmask_cpu=np.zeros((len(request_ids), 1), dtype=np.int32),
        require_structured_out_cpu=np.zeros((len(request_ids), 1), dtype=bool),
        structured_decode_arange=np.arange(32, dtype=np.int32),
        input_batch=SimpleNamespace(
            req_id_to_index={rid: i for i, rid in enumerate(request_ids)}))
    return StructuredDecodingManager(runner)


@pytest.mark.parametrize("missing_request", [False, True])
def test_runner_preserves_mask_rows_when_ids_are_unsorted_or_absent(missing_request):
    # Each row permits a different token. Batch order differs from mask order.
    ids = ["zulu", "missing", "alpha"] if missing_request else ["zulu", "alpha"]
    values = [[1], [8], [4]] if missing_request else [[1], [4]]
    output = GrammarOutput(ids, np.array(values, dtype=np.int32))
    manager = decoding_manager(["alpha", "plain", "zulu"])
    logits = jnp.zeros((3, 32))
    required, masks, arange = manager.prepare_structured_decoding_input(logits, output)
    actual = np.asarray(manager.structured_decode_fn(required, masks, logits, arange))
    assert np.flatnonzero(np.isfinite(actual[0])).tolist() == [2]
    assert np.isfinite(actual[1]).all()
    assert np.flatnonzero(np.isfinite(actual[2])).tolist() == [0]


def test_rank_masks_sample_tokens_accepted_by_each_native_grammar():
    # Initialize worker processes before CPU JAX compilation.
    with grammar_scheduler([(0, "zulu", "ab"), (1, "alpha", "cd")]) as scheduler:
        manager = decoding_manager(["alpha", "zulu"])
        for expected in ([2, 0], [3, 1]):
            frame = scheduler.schedule(["zulu", "alpha"])
            output = scheduler.get_grammar_bitmask(frame)
            logits = jnp.zeros((2, 32))
            args = manager.prepare_structured_decoding_input(logits, output)
            constrained = manager.structured_decode_fn(args[0], args[1], logits, args[2])
            sampled = np.asarray(jnp.argmax(constrained, axis=-1)).tolist()
            assert sampled == expected
            scheduler.update_from_output(frame, ModelRunnerOutput(
                req_ids=["alpha", "zulu"], req_id_to_index={"alpha": 0, "zulu": 1},
                sampled_token_ids=[[token] for token in sampled], logprobs=None,
                prompt_logprobs_dict={}))
