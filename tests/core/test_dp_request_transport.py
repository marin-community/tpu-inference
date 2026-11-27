# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import os
import unittest
from concurrent.futures import Future
from types import SimpleNamespace

import cloudpickle
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from tpu_inference.core.sched.dp_scheduler import (DPScheduler, SchedulerCommand,
                                                 _scheduler_worker_process)


def structured_request():
    params = SamplingParams(
        max_tokens=7,
        temperature=0.3,
        structured_outputs=StructuredOutputsParams(json_object=True))
    request = Request("structured", [1, 2, 3], params, None)
    request.structured_output_request.reasoning_ended = True
    request.structured_output_request.reasoning_parser_kwargs = {"key": "value"}
    request.structured_output_request.grammar = Future()
    return request


class LocalGrammarManager:

    def __init__(self, fail):
        self.fail = fail

    def grammar_init(self, request):
        future = Future()
        if self.fail:
            future.set_exception(ValueError("invalid grammar"))
        else:
            future.set_result(os.getpid())
        request.structured_output_request.grammar = future


class RecordingScheduler:

    def __init__(self, **kwargs):
        self.grammar_compile_error_reqs = set()

    def add_request(self, request):
        self.request = request

    def make_stats(self, spec_decoding_stats, kv_connector_stats):
        request = self.request
        promoted = Scheduler._try_promote_blocked_waiting_request(self, request)
        return {
            "promoted": promoted,
            "grammar_failed": request.request_id in self.grammar_compile_error_reqs,
            "grammar": request.structured_output_request.grammar,
            "constraints": request.structured_output_request.params,
            "prompt": request.prompt_token_ids,
            "sampling": request.sampling_params,
            "reasoning_ended": request.structured_output_request.reasoning_ended,
            "reasoning_kwargs": request.structured_output_request.reasoning_parser_kwargs,
        }

    def shutdown(self):
        pass


class TestDPRequestTransport(unittest.TestCase):

    def test_pending_and_completed_grammar_cross_request_commands(self):
        for command in (SchedulerCommand.ADD_REQUEST,
                        SchedulerCommand.PROBE_COMPUTED_BLOCKS):
            for completed in (False, True):
                with self.subTest(command=command, completed=completed):
                    request = structured_request()
                    future = request.structured_output_request._grammar
                    if completed:
                        future.set_result(None)
                    with self.assertRaises(TypeError):
                        cloudpickle.dumps((command, request))
                    sender, receiver = multiprocessing.Pipe()
                    try:
                        scheduler = object.__new__(DPScheduler)
                        scheduler.input_conns = [sender]
                        scheduler._send_command(0, command, request)
                        received_command, received = cloudpickle.loads(
                            receiver.recv_bytes())
                        self.assertEqual(received_command, command)
                        self.assertEqual(received.sampling_params,
                                         request.sampling_params)
                        self.assertEqual(received.prompt_token_ids, [1, 2, 3])
                        self.assertIsNone(received.structured_output_request._grammar)
                        self.assertIs(request.structured_output_request._grammar,
                                      future)
                    finally:
                        sender.close()
                        receiver.close()

    def worker_result(self, fail):
        context = multiprocessing.get_context("fork")
        input_parent, input_child = context.Pipe()
        output_parent, output_child = context.Pipe()
        process = context.Process(
            target=_scheduler_worker_process,
            args=(0, input_child, output_child,
                  SimpleNamespace(additional_config={}), None,
                  LocalGrammarManager(fail), 16, 16, None, False, False,
                  RecordingScheduler))
        process.start()
        request = structured_request()
        try:
            sender = object.__new__(DPScheduler)
            sender.input_conns = [input_parent]
            sender._send_command(0, SchedulerCommand.ADD_REQUEST, request)
            self.assertTrue(output_parent.poll(15), "worker did not admit request")
            self.assertIsNone(cloudpickle.loads(output_parent.recv_bytes()))
            sender._send_command(0, SchedulerCommand.MAKE_STATS, (None, None))
            self.assertTrue(output_parent.poll(15), "worker did not report grammar")
            result = cloudpickle.loads(output_parent.recv_bytes())
            self.assertEqual(result["sampling"], request.sampling_params)
            self.assertEqual(result["constraints"],
                             request.structured_output_request.params)
            self.assertEqual(result["prompt"], [1, 2, 3])
            self.assertTrue(result["reasoning_ended"])
            self.assertEqual(result["reasoning_kwargs"], {"key": "value"})
            self.assertFalse(request.structured_output_request._grammar.done())
            return result, process.pid
        finally:
            if process.is_alive():
                input_parent.send_bytes(cloudpickle.dumps(
                    (SchedulerCommand.SHUTDOWN, None)))
                process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
            for connection in (input_parent, input_child, output_parent,
                               output_child):
                connection.close()

    def test_worker_recompiles_grammar_and_admits_constrained_request(self):
        result, worker_pid = self.worker_result(False)
        self.assertTrue(result["promoted"])
        self.assertFalse(result["grammar_failed"])
        self.assertEqual(result["grammar"], worker_pid)

    def test_worker_grammar_failure_stays_blocked(self):
        result, _ = self.worker_result(True)
        self.assertFalse(result["promoted"])
        self.assertTrue(result["grammar_failed"])
        self.assertIsInstance(result["grammar"], ValueError)


if __name__ == "__main__":
    unittest.main()
