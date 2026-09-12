# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import socket
import subprocess
import time
import warnings
from collections.abc import AsyncGenerator, Iterable, Mapping
from copy import copy
from typing import Any

import torch

import vllm.envs as envs
from vllm import TokensPrompt
from vllm.config import VllmConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.entrypoints.serve.elastic_ep.middleware import set_scaling_elastic_ep
from vllm.exceptions import (
    GracefulHTTPError,
    MaxQueuedTokensError,
    QueueOverflowError,
    VLLMClientError,
    VLLMValidationError,
)
from vllm.inputs import EngineInput, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import STREAM_FINISHED, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.tracing import init_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.usage.usage_lib import UsageContext
from vllm.utils.async_utils import cancel_task_threadsafe
from vllm.utils.collection_utils import as_list
from vllm.v1.engine import EngineCoreRequest, PauseMode
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest, FaultToleranceResult
from vllm.v1.metrics.loggers import (
    StatLoggerFactory,
    StatLoggerManager,
    load_stat_logger_plugin_factories,
)
from vllm.v1.metrics.prometheus import shutdown_prometheus
from vllm.v1.metrics.stats import IterationStats

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# vllm-sm75 overlay: deep-sleep respawn tuning.
#
# Respawning an engine is a full cold start (model load + CUDA graph capture),
# and GPU memory is released asynchronously when the previous engine exits.
# A single failed attempt used to wedge the API server forever, so attempts
# are now retried (after reaping orphan processes) and the gate stays armed
# for later requests instead of marking the engine permanently dead.
#
# Timeouts are deliberately generous: a cold start reads the whole model from
# disk, which on a mechanical HDD with a large model can take many minutes.
# Every value is overridable via env.
# ---------------------------------------------------------------------------
# Attempts per request batch before failing fast.
VLLM_SM75_RESPAWN_MAX_ATTEMPTS = int(
    os.environ.get("VLLM_SM75_RESPAWN_MAX_ATTEMPTS", "3")
)
# Backoff between attempts, in seconds.
VLLM_SM75_RESPAWN_BACKOFF_S = float(
    os.environ.get("VLLM_SM75_RESPAWN_BACKOFF_S", "5")
)
# Upper bound for waiting for GPU memory to be released before each retry.
VLLM_SM75_RESPAWN_GPU_WAIT_S = float(
    os.environ.get("VLLM_SM75_RESPAWN_GPU_WAIT_S", "120")
)
# Wall-clock cap for a single respawn attempt (model load + ready wait).  The
# model-load step (asyncio.to_thread) has no built-in timeout and can stall on
# a slow disk; this bounds it so one stuck load cannot hang the gate forever.
VLLM_SM75_RESPAWN_ATTEMPT_TIMEOUT_S = float(
    os.environ.get("VLLM_SM75_RESPAWN_ATTEMPT_TIMEOUT_S", "3600")
)
# Wall-clock cap for the whole retry loop.  A backstop so a bad respawn cycle
# cannot hold the gate (and every request behind it) for an unbounded time.
VLLM_SM75_RESPAWN_TOTAL_BUDGET_S = float(
    os.environ.get("VLLM_SM75_RESPAWN_TOTAL_BUDGET_S", "7200")
)


class InputStreamError(Exception):
    """Wrapper for errors from the input stream generator.

    This is used to propagate errors from the user's input generator
    without wrapping them in EngineGenerateError.
    """

    def __init__(self, cause: Exception):
        self.cause = cause
        super().__init__(str(cause))


class AsyncLLM(EngineClient):
    """An asynchronous wrapper for the vLLM engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        """
        Create an AsyncLLM.

        Args:
            vllm_config: global configuration.
            executor_class: an Executor impl, e.g. MultiprocExecutor.
            log_stats: Whether to log stats.
            usage_context: Usage context of the LLM.
            mm_registry: Multi-modal registry.
            log_requests: Whether to log requests.
            start_engine_loop: Whether to start the engine loop.
            stat_loggers: customized stat loggers for the engine.
                If not provided, default stat loggers will be used.
                PLEASE BE AWARE THAT STAT LOGGER IS NOT STABLE
                IN V1, AND ITS BASE CLASS INTERFACE MIGHT CHANGE.

        Returns:
            None
        """
        # Ensure we can serialize custom transformer configs
        maybe_register_config_serialize_by_value()

        self.vllm_config = vllm_config
        self._elastic_ep_lock = asyncio.Lock()
        # vllm-sm75 overlay: serializes deep-sleep respawns so concurrent
        # requests arriving while the engine is down trigger exactly one
        # respawn and all wait for it.
        self._deep_sleep_respawn_lock = asyncio.Lock()
        # Shared in-flight respawn task.  The respawn must NOT run inside a
        # request's coroutine: when a client times out and disconnects, uvicorn
        # cancels that coroutine, which aborted the cold start halfway and left
        # orphaned engine processes holding GPU memory.  Those orphans then
        # starved the next respawn ("free memory ... is less than desired GPU
        # memory utilization"), which is what made concurrent wake-ups fail.
        # Running it as a detached task + awaiting it under asyncio.shield()
        # keeps it alive across client disconnects and lets later requests
        # join the cold start that is already in progress.
        self._deep_sleep_respawn_task: asyncio.Task | None = None
        # Set when the last respawn attempt failed: the gate stays armed so a
        # later request retries instead of the server staying wedged forever.
        self._deep_sleep_respawn_pending = False
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.observability_config = vllm_config.observability_config

        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        if tracing_endpoint is not None:
            init_tracer("vllm.llm_engine", tracing_endpoint)

        self.log_requests = log_requests

        custom_stat_loggers = list(stat_loggers or [])
        custom_stat_loggers.extend(load_stat_logger_plugin_factories())

        has_custom_loggers = bool(custom_stat_loggers)
        self.log_stats = log_stats or has_custom_loggers
        if not log_stats and has_custom_loggers:
            logger.info(
                "AsyncLLM created with log_stats=False, "
                "but custom stat loggers were found; "
                "enabling logging without default stat loggers."
            )

        self.renderer = renderer = renderer_from_config(self.vllm_config)

        # Convert EngineInput --> EngineCoreRequest.
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # Converts EngineCoreOutputs --> RequestOutput.
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

        # Loggers.
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=custom_stat_loggers,
                enable_default_loggers=log_stats,
                client_count=client_count,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        self._client_count = client_count

        self.output_handler: asyncio.Task | None = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

        if (
            vllm_config.profiler_config.profiler == "torch"
            and not vllm_config.profiler_config.ignore_frontend
        ):
            profiler_dir = vllm_config.profiler_config.torch_profiler_dir
            logger.info(
                "Torch profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501
                profiler_dir,
            )
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_llm"
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=vllm_config.profiler_config.torch_profiler_with_stack,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    profiler_dir,
                    worker_name=worker_name,
                    use_gzip=vllm_config.profiler_config.torch_profiler_use_gzip,
                ),
            )
        else:
            self.profiler = None

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncLLM":
        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def __del__(self):
        self.shutdown()

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown, cleaning up the background proc and IPC."""
        shutdown_prometheus()

        if renderer := getattr(self, "renderer", None):
            renderer.shutdown()

        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown(timeout=timeout)

        handler = getattr(self, "output_handler", None)
        if handler is not None:
            cancel_task_threadsafe(handler)

    def get_num_unfinished_requests(self) -> int:
        return self.output_processor.get_num_unfinished_requests()

    def get_num_queued_tokens(self) -> int:
        return self.output_processor.get_num_queued_tokens()

    def check_admission(self, n: int = 1, request_id: str | None = None) -> None:
        """Reject the request if it would exceed queue limits.

        Both limits return HTTP 503 (Service Unavailable) so that load
        balancers and client SDKs retry on a different instance.

        - ``max_num_queued_reqs``: hard cap on the number of unfinished requests
          (waiting + running).  A request with ``n > 1`` counts as ``n`` slots.
        - ``max_num_queued_tokens``: TTFT QoS — cap on the total prompt
          tokens of requests still in prefill.

        Note: ``get_num_queued_tokens`` uses ``prompt_len`` for all prefilling requests.
        Chunked prefill progress and prefix-cache hits are not subtracted because the
        scheduler's ``num_computed_tokens`` and ``num_cached_tokens`` are only
        propagated to the API server after prefill completes. The overestimation is
        conservative — earlier rejection, preserving TTFT targets.

        Args:
            n: Number of sequences the request will occupy.
            request_id: Request id, used for logging only.

        Raises:
            QueueOverflowError: If ``max_num_queued_reqs`` would be exceeded.
            MaxQueuedTokensError: If ``max_num_queued_tokens`` would be exceeded.
        """
        max_num_reqs = self.scheduler_config.max_num_queued_reqs
        if max_num_reqs is not None:
            current = self.get_num_unfinished_requests()
            if current + n > max_num_reqs:
                logger.info(
                    "Request queue full - rejecting request %s "
                    "(current=%d, n=%d, max=%d).",
                    request_id,
                    current,
                    n,
                    max_num_reqs,
                )
                raise QueueOverflowError()

        max_queued_tokens = self.scheduler_config.max_num_queued_tokens
        if max_queued_tokens is not None:
            current_tokens = self.get_num_queued_tokens()
            if current_tokens >= max_queued_tokens:
                logger.info(
                    "Max queued tokens reached - rejecting request %s "
                    "(current_tokens=%d, max=%d).",
                    request_id,
                    current_tokens,
                    max_queued_tokens,
                )
                raise MaxQueuedTokensError()

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            self._supported_tasks = await self.engine_core.get_supported_tasks_async()

        return self._supported_tasks

    async def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest
        | PromptType
        | EngineInput
        | AsyncGenerator[StreamingInput, None],
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        session_id: str | None = None,
        prompt_text: str | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""

        if self.errored:
            raise EngineDeadError()

        # vllm-sm75 overlay: deep-sleep gate.  If the engine exited for deep
        # sleep (auto-sleep "exit"), respawn it transparently before
        # submitting; this request's latency includes the cold start.  The
        # pending flag also re-enters the gate after a failed attempt so the
        # request retries instead of being served by a missing engine.
        if (
            self.engine_core.resources.engine_sleeping
            or self._deep_sleep_respawn_pending
        ):
            await self._await_deep_sleep_respawn()

        is_pooling = isinstance(params, PoolingParams)

        if (
            self.vllm_config.cache_config.kv_sharing_fast_prefill
            and not is_pooling
            and params.prompt_logprobs
        ):
            raise VLLMValidationError(
                "--kv-sharing-fast-prefill produces incorrect logprobs for "
                "prompt tokens, please disable it when the requests need "
                "prompt logprobs"
            )

        if isinstance(params, SamplingParams) and params.n > 1:
            # TODO (NickLucche) Batch check admission check for all n requests
            self.check_admission(params.n, request_id)

        if isinstance(prompt, AsyncGenerator):
            if reasoning_ended is not None or reasoning_parser_kwargs is not None:
                raise NotImplementedError

            # Streaming input case.
            return await self._add_streaming_input_request(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
                data_parallel_rank,
                session_id,
            )

        # Convert Input --> Request.
        if isinstance(prompt, EngineCoreRequest):
            logger.warning_once(
                "Passing EngineCoreRequest to AsyncLLM.generate() and .add_requests() "
                "is deprecated and will be removed in the future. You should "
                "instead pass the outputs of Renderer.render_cmpl() or "
                "Renderer.render_chat()."
            )

            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "AsyncLLM.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            if isinstance(prompt, dict) and "type" in prompt:
                # Rendered EngineInput; no blocking preprocessing needed.
                request = self.input_processor.process_inputs(
                    request_id,
                    prompt,
                    params,
                    supported_tasks=await self.get_supported_tasks(),
                    arrival_time=arrival_time,
                    lora_request=lora_request,
                    tokenization_kwargs=tokenization_kwargs,
                    trace_headers=trace_headers,
                    priority=priority,
                    data_parallel_rank=data_parallel_rank,
                    session_id=session_id,
                )
            else:
                # Raw prompts require tokenization and possibly multimodal
                # processing, which must not block the event loop.
                request = await self.input_processor.process_inputs_async(
                    request_id,
                    prompt,
                    params,
                    supported_tasks=await self.get_supported_tasks(),
                    arrival_time=arrival_time,
                    lora_request=lora_request,
                    tokenization_kwargs=tokenization_kwargs,
                    trace_headers=trace_headers,
                    priority=priority,
                    data_parallel_rank=data_parallel_rank,
                    session_id=session_id,
                )
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

        if reasoning_ended is not None:
            request.reasoning_ended = reasoning_ended
        if reasoning_parser_kwargs is not None:
            request.reasoning_parser_kwargs = reasoning_parser_kwargs

        self.input_processor.assign_request_id(request)

        # We start the output_handler on the first call to add_request() so
        # we can call __init__ before the event loop, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        self._run_output_handler()

        # Create a new output collector for the request.
        queue = RequestOutputCollector(params.output_kind, request.request_id)

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        if is_pooling or params.n == 1:
            await self._add_request(request, prompt_text, None, 0, queue)
            return queue

        parent_params = params
        assert isinstance(parent_params, SamplingParams)

        # Fan out child requests (for n>1).
        parent_request = ParentRequest(request)
        for idx in range(parent_params.n):
            request_id, child_params = parent_request.get_child_info(idx)
            child_request = request if idx == parent_params.n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params
            await self._add_request(
                child_request, prompt_text, parent_request, idx, queue
            )
        return queue

    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ):
        if parent_req is None and not self.output_processor.has_request(
            request.request_id
        ):
            self.check_admission(request_id=request.request_id)

        # Register locally before the first await so concurrent tasks see this request.
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)

    async def _add_streaming_input_request(
        self,
        request_id: str,
        input_stream: AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        session_id: str | None = None,
    ) -> RequestOutputCollector:
        self._validate_streaming_input_sampling_params(sampling_params)

        inputs = dict(
            supported_tasks=await self.get_supported_tasks(),
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            session_id=session_id,
        )

        if not sampling_params.skip_clone:
            sampling_params = sampling_params.clone()
            sampling_params.skip_clone = True

        # Create request for validation, also used as the finished signal
        # once the input stream is closed.
        final_req = self.input_processor.process_inputs(
            request_id=request_id,
            prompt=TokensPrompt(prompt_token_ids=[0]),
            params=sampling_params,
            **inputs,  # type: ignore[arg-type]
        )
        self.input_processor.assign_request_id(final_req)
        internal_req_id = final_req.request_id

        queue = RequestOutputCollector(sampling_params.output_kind, internal_req_id)

        async def handle_inputs():
            cancelled = False
            try:
                async for input_chunk in input_stream:
                    sp = input_chunk.sampling_params
                    if sp:
                        self._validate_streaming_input_sampling_params(sp)
                    else:
                        sp = sampling_params
                    # TODO(nick): Avoid re-validating reused sampling parameters
                    req = self.input_processor.process_inputs(
                        request_id=internal_req_id,
                        prompt=input_chunk.prompt,
                        params=sp,
                        resumable=True,
                        **inputs,  # type: ignore[arg-type]
                    )
                    req.external_req_id = request_id
                    if req.prompt_embeds is not None:
                        raise VLLMValidationError(
                            "prompt_embeds not supported for streaming inputs"
                        )
                    prompt_text, _, _ = extract_prompt_components(
                        self.model_config, input_chunk.prompt
                    )
                    await self._add_request(req, prompt_text, None, 0, queue)
            except (asyncio.CancelledError, GeneratorExit):
                cancelled = True
            except Exception as error:
                # Wrap in InputStreamError so generate() can propagate it
                # without wrapping in EngineGenerateError.
                queue.put(InputStreamError(error))
            finally:
                queue._input_stream_task = None
                if not cancelled:
                    # Send empty final request to indicate that inputs have
                    # finished. Don't send if cancelled (session was aborted).
                    await self._add_request(final_req, None, None, 0, queue)

        # Ensure output handler is running.
        self._run_output_handler()

        queue._input_stream_task = asyncio.create_task(handle_inputs())
        return queue

    @staticmethod
    def _validate_streaming_input_sampling_params(
        params: SamplingParams | PoolingParams,
    ):
        if (
            not isinstance(params, SamplingParams)
            or params.n > 1
            or params.output_kind == RequestOutputKind.FINAL_ONLY
            or params.stop
        ):
            raise VLLMValidationError(
                "Input streaming not currently supported "
                "for pooling models, n > 1, request_kind = FINAL_ONLY "
                "or with stop strings."
            )

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    async def generate(
        self,
        prompt: EngineCoreRequest
        | PromptType
        | EngineInput
        | AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        session_id: str | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        q: RequestOutputCollector | None = None
        try:
            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                session_id=session_id,
                prompt_text=prompt_text,
                reasoning_ended=reasoning_ended,
                reasoning_parser_kwargs=reasoning_parser_kwargs,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()

                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                assert isinstance(out, RequestOutput)
                finished = out.finished
                if out is not STREAM_FINISHED:
                    yield out

        # If the request is disconnected by the client, generate()
        # is cancelled or the generator is garbage collected. So,
        # we abort the request if we end up here.
        except (asyncio.CancelledError, GeneratorExit):
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error or admission control rejection.
        except (VLLMClientError, GracefulHTTPError) as e:
            if self.log_requests:
                logger.info("Request %s failed (bad request): %s.", request_id, e)
            raise

        # Error from input stream generator - propagate directly.
        except InputStreamError as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed (input error): %s.", request_id, e)
            raise e.cause from e

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                try:
                    s = f"{e.__class__.__name__}: {e}"
                except Exception as e2:
                    s = (
                        f"{e.__class__.__name__}: "
                        "error during printing an exception of class"
                        + e2.__class__.__name__
                    )
                logger.info("Request %s failed due to %s.", request_id, s)
            raise EngineGenerateError() from e
        finally:
            if q is not None:
                q.close()

    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        if self.output_handler is not None:
            return

        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        # We use a mutable list for logger_manager so that it can be updated
        # during elastic EP scaling (see scale_elastic_ep) without creating
        # a circular reference via self.
        self._logger_ref = [self.logger_manager]
        logger_ref = self._logger_ref
        renderer = self.renderer
        # P0 multi-modal sender ("shadow") cache; None for text-only models.
        mm_processor_cache = renderer.mm_processor_cache
        chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE

        async def output_handler():
            try:
                while True:
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    outputs = await engine_core.get_output_async()
                    num_outputs = len(outputs.outputs)

                    iteration_stats = (
                        IterationStats() if (log_stats and num_outputs) else None
                    )

                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    engine_core_outputs = outputs.outputs
                    for start in range(0, num_outputs, chunk_size):
                        end = start + chunk_size
                        outputs_slice = engine_core_outputs[start:end]
                        # 2) Process EngineCoreOutputs.
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats
                        )
                        # NOTE: RequestOutputs are pushed to their queues.
                        assert not processed_outputs.request_outputs

                        # 2b) Recover from P0/P1 cache drift: the engine flags hashes
                        # it couldn't find (mm_cache_miss_hashes); drop them from the
                        # P0 shadow so the client's retry resends the data and
                        # repopulates P1. Hot-path no-op (field is None otherwise).
                        if mm_processor_cache is not None:
                            for eco in outputs_slice:
                                if eco.mm_cache_miss_hashes:
                                    for mm_hash in eco.mm_cache_miss_hashes:
                                        mm_processor_cache.invalidate(mm_hash)

                        # Allow other asyncio tasks to run between chunks
                        if end < num_outputs:
                            await asyncio.sleep(0)

                        # 3) Abort any reqs that finished due to stop strings.
                        if processed_outputs.reqs_to_abort:
                            await engine_core.abort_requests_async(
                                processed_outputs.reqs_to_abort
                            )

                    output_processor.update_scheduler_stats(outputs.scheduler_stats)

                    # 4) Logging.
                    # TODO(rob): make into a coroutine and launch it in
                    # background thread once Prometheus overhead is non-trivial.
                    if logger_ref[0]:
                        logger_ref[0].record(
                            engine_idx=outputs.engine_index,
                            scheduler_stats=outputs.scheduler_stats,
                            iteration_stats=iteration_stats,
                            mm_cache_stats=renderer.stat_mm_cache(),
                        )
            except Exception as e:
                logger.exception("AsyncLLM output_handler failed.")
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(output_handler())

    async def abort(
        self, request_id: str | Iterable[str], internal: bool = False
    ) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""

        request_ids = (
            (request_id,) if isinstance(request_id, str) else as_list(request_id)
        )
        all_request_ids = self.output_processor.abort_requests(request_ids, internal)
        await self.engine_core.abort_requests_async(all_request_ids)

        if self.log_requests:
            logger.info("Aborted request(s) %s.", ",".join(request_ids))

    async def notify_kv_transfer_request_rejected(
        self,
        request_id: str,
        kv_transfer_params: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> None:
        """Submit a pre-aborted request so the connector's request_finished
        hook runs to free any pre-admission KV-transfer resources (e.g. NIXL
        prefill blocks pinned on the P node)."""
        request = EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=[0],
            mm_features=None,
            sampling_params=SamplingParams(
                max_tokens=1,
                extra_args={"kv_transfer_params": dict(kv_transfer_params)},
            ),
            pooling_params=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=data_parallel_rank,
            abort_immediately=True,
        )
        await self.engine_core.add_request_async(request)

    async def pause_generation(
        self,
        *,
        mode: PauseMode = "abort",
        wait_for_inflight_requests: bool | None = None,
        clear_cache: bool = True,
    ) -> None:
        """
        Pause generation to allow model weight updates.

        All mode handling (abort / wait / keep) and cache clearing is done
        in the engine. New generation/encoding requests will not be scheduled
        until resume is called.

        Args:
            mode: How to handle in-flight requests:
                - ``"abort"``: Abort all in-flight requests immediately
                  (default).
                - ``"wait"``: Wait for in-flight requests to complete.
                - ``"keep"``: Freeze requests in queue; they resume on
                  :meth:`resume_generation`.
            wait_for_inflight_requests: DEPRECATED: use mode argument.
            clear_cache: Whether to clear KV cache and prefix cache after
                draining. Set to ``False`` to preserve cache for faster resume.
        """
        if wait_for_inflight_requests:
            warnings.warn(
                "The `wait_for_inflight_requests` parameter in "
                "`AsyncLLM.pause_generation()` is deprecated. "
                "Please use `mode` argument instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "wait"
        if clear_cache:
            await self.renderer.clear_mm_cache_async()
        await self.engine_core.pause_scheduler_async(mode=mode, clear_cache=clear_cache)
        # Small sleep to help ensure that final outputs from any in-flight requests are
        # returned prior to this method returning. These outputs come out of the engine
        # prior to the wait-for-idle completion event, but involve additional async
        # tasks in output processing.
        # Note that this is not required for correctness, just more intuitive ordering
        # of events from caller's pov.
        await asyncio.sleep(0.02)

    async def resume_generation(self) -> None:
        """Resume generation after :meth:`pause_generation`."""
        await self.engine_core.resume_scheduler_async()

    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        return await self.engine_core.is_scheduler_paused_async()

    # vllm-sm75 overlay: runtime speculative-decoding on/off. Uses the generic
    # UTILITY channel (no dedicated core_client method needed); the engine-core
    # methods are looked up by name on the other side.
    async def set_speculative_decoding(self, enabled: bool) -> None:
        await self.engine_core.call_utility_async(
            "set_speculative_decoding", enabled
        )

    async def is_speculative_decoding_enabled(self) -> bool:
        return await self.engine_core.call_utility_async(
            "is_speculative_decoding_enabled"
        )

    async def is_speculative_decoding_configured(self) -> bool:
        return await self.engine_core.call_utility_async(
            "is_speculative_decoding_configured"
        )

    async def encode(
        self,
        prompt: PromptType | EngineInput,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
        reasoning_ended: bool | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        q: RequestOutputCollector | None = None
        try:
            q = await self.add_request(
                request_id,
                prompt,
                pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                reasoning_ended=reasoning_ended,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()
                assert isinstance(out, PoolingRequestOutput)
                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = out.finished
                yield out

        # If the request is disconnected by the client, generate()
        # is cancelled. So, we abort the request if we end up here.
        except asyncio.CancelledError:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error or admission control rejection.
        except (VLLMClientError, GracefulHTTPError):
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e
        finally:
            if q is not None:
                q.close()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    async def is_tracing_enabled(self) -> bool:
        return self.observability_config.otlp_traces_endpoint is not None

    async def do_log_stats(self) -> None:
        if self.logger_manager:
            self.logger_manager.log()

    async def check_health(self) -> None:
        logger.debug("Called check_health.")
        if self.errored:
            raise self.dead_error

    async def start_profile(self, profile_prefix: str | None = None) -> None:
        coros = [self.engine_core.profile_async(True, profile_prefix)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.start))
        await asyncio.gather(*coros)

    async def stop_profile(self) -> None:
        coros = [self.engine_core.profile_async(False)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.stop))
        await asyncio.gather(*coros)

    async def reset_mm_cache(self) -> None:
        await self.renderer.clear_mm_cache_async()
        await self.engine_core.reset_mm_cache_async()

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return await self.engine_core.reset_prefix_cache_async(
            reset_running_requests, reset_connector
        )

    async def reset_encoder_cache(self) -> None:
        await self.engine_core.reset_encoder_cache_async()

    async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        if level >= 1:
            await self.renderer.clear_mm_cache_async()
        await self.engine_core.sleep_async(level, mode)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        await self.engine_core.wake_up_async(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    async def checkpoint_prepare(self) -> None:
        await self.collective_rpc("checkpoint_prepare")

    async def checkpoint_restore(self) -> None:
        await self.collective_rpc("checkpoint_restore")

    async def is_sleeping(self) -> bool:
        return await self.engine_core.is_sleeping_async()

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return await self.engine_core.add_lora_async(lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return await self.engine_core.remove_lora_async(lora_id)

    async def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return await self.engine_core.list_loras_async()

    async def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return await self.engine_core.pin_lora_async(lora_id)

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """
        Perform a collective RPC call to the given path.
        """
        return await self.engine_core.collective_rpc_async(
            method, timeout, args, kwargs
        )

    async def wait_for_requests_to_drain(self, drain_timeout: int = 300):
        """Wait for all requests to be drained."""
        start_time = time.time()
        while time.time() - start_time < drain_timeout:
            if not self.engine_core.dp_engines_running():
                logger.info("Engines are idle, requests have been drained")
                return

            logger.info("Engines are still running, waiting for requests to drain...")
            await asyncio.sleep(1)  # Wait 1 second before checking again

        raise TimeoutError(
            f"Timeout reached after {drain_timeout} seconds "
            "waiting for requests to drain."
        )

    async def _drain_requests_for_elastic_ep(self, drain_timeout: int) -> None:
        try:
            logger.info(
                "VLLM_ELASTIC_EP_DRAIN_REQUESTS is set, "
                "waiting for requests to drain before scaling"
            )
            await self.wait_for_requests_to_drain(drain_timeout)
        except BaseException:
            set_scaling_elastic_ep(False)
            raise

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ):
        """Scale the elastic EP data parallel size."""
        async with self._elastic_ep_lock:
            await self._scale_elastic_ep(new_data_parallel_size, drain_timeout)

    async def _scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int
    ) -> None:
        old_data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        if old_data_parallel_size == new_data_parallel_size:
            logger.info(
                "Data parallel size is already %s, skipping scale",
                new_data_parallel_size,
            )
            return

        await self.engine_core.prepare_elastic_ep(new_data_parallel_size)

        # recreate stat loggers
        if new_data_parallel_size > old_data_parallel_size and self.log_stats:
            # TODO(rob): fix this after talking with Ray team.
            # This resets all the prometheus metrics since we
            # unregister during initialization. Need to understand
            # the intended behavior here better.
            self.logger_manager = StatLoggerManager(
                vllm_config=self.vllm_config,
                engine_idxs=list(range(new_data_parallel_size)),
                custom_stat_loggers=None,
            )
            # Update the mutable ref so output_handler picks up the
            # new logger without creating a circular reference via self.
            if hasattr(self, "_logger_ref"):
                self._logger_ref[0] = self.logger_manager
            self.logger_manager.log_engine_initialized()

        set_scaling_elastic_ep(True)
        if envs.VLLM_ELASTIC_EP_DRAIN_REQUESTS:
            await self._drain_requests_for_elastic_ep(drain_timeout)

        await self.engine_core.commit_elastic_ep()
        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        set_scaling_elastic_ep(False)

    async def handle_fault(
        self, fault_tolerance_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        """send fault tolerance instruction to the engine"""
        return await self.engine_core.handle_fault(fault_tolerance_request)

    async def get_status(self):
        return await self.engine_core.get_status()

    @property
    def is_running(self) -> bool:
        # Is None before the loop is started.
        return self.output_handler is None or not self.output_handler.done()

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return self.engine_core.resources.engine_dead or not self.is_running

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()

    async def _await_deep_sleep_respawn(self) -> None:
        # vllm-sm75 overlay: concurrency-safe deep-sleep respawn gate.
        #
        # Requests arriving while the engine is asleep share ONE respawn: the
        # first one ("leader") performs it, the rest join the in-flight task
        # and reuse the outcome.  The respawn runs as a detached task awaited
        # under asyncio.shield() so a client disconnect (which cancels the
        # request coroutine) cannot abort the cold start halfway and leave
        # orphaned engine processes holding GPU memory.
        #
        # A failed attempt no longer marks the engine permanently dead (which
        # wedged the API server until the container was restarted): the gate
        # stays armed so the next request retries once GPU memory is free.
        resources = self.engine_core.resources

        # Fast path: engine is awake and no respawn is owed.
        if not resources.engine_sleeping and not self._deep_sleep_respawn_pending:
            return

        async with self._deep_sleep_respawn_lock:
            # Re-check under the lock -- an earlier request may have already
            # finished the work while we waited for the lock.
            if not resources.engine_sleeping and not self._deep_sleep_respawn_pending:
                return
            task = self._deep_sleep_respawn_task
            if task is None or task.done():
                # Start (or restart, if a prior attempt's task is finished) a
                # detached respawn.  It outlives this request so a client
                # disconnect cannot abort the cold start.  Checking
                # task.done() -- not just `is None` -- means a finished attempt
                # is always replaced by a fresh one instead of being re-awaited
                # (e.g. when the only waiter was cancelled before it completed).
                task = asyncio.create_task(self._respawn_engine_with_retries())
                self._deep_sleep_respawn_task = task
                # Consume the exception if every waiter goes away, to avoid
                # "Task exception was never retrieved" noise.
                task.add_done_callback(
                    lambda t: t.cancelled() or t.exception()
                )
                logger.info(
                    "[deep-sleep] engine exited for deep sleep; respawning it "
                    "(cold start; this request's latency includes it)"
                )
            else:
                logger.info(
                    "[deep-sleep] respawn already in progress; joining it "
                    "instead of starting a second engine"
                )

        try:
            # shield(): if this request is cancelled (client timeout), the
            # shared respawn keeps running for the other waiters.
            await asyncio.shield(task)
        finally:
            # Whoever observes it finished clears the shared slot so a later
            # request can start a fresh attempt (i.e. retry after a failure).
            if task.done() and self._deep_sleep_respawn_task is task:
                self._deep_sleep_respawn_task = None

    async def _respawn_engine_with_retries(self) -> None:
        """Run the deep-sleep respawn, retrying after reaping orphan engines.

        A failed respawn can leave worker processes alive that still hold GPU
        memory but are unreachable (the handshake never completed).  They must
        be reaped and their memory released before the next attempt, otherwise
        every retry fails with "free memory on device ... is less than desired
        GPU memory utilization".

        The FIRST attempt is the common case (the engine exited cleanly for
        deep sleep, so there are no orphans and its GPU memory is released
        asynchronously); it therefore goes straight to the respawn without the
        reap / GPU-memory wait that only matter after a failure.
        """
        last_exc: BaseException | None = None
        # Backstop so a bad respawn cycle cannot hold the gate (and every
        # request behind it) for an unbounded time.
        deadline = time.monotonic() + VLLM_SM75_RESPAWN_TOTAL_BUDGET_S

        for attempt in range(1, VLLM_SM75_RESPAWN_MAX_ATTEMPTS + 1):
            if time.monotonic() >= deadline:
                logger.warning(
                    "[deep-sleep] respawn total budget (%.0fs) exhausted "
                    "before attempt %d/%d; deferring to a later request",
                    VLLM_SM75_RESPAWN_TOTAL_BUDGET_S,
                    attempt,
                    VLLM_SM75_RESPAWN_MAX_ATTEMPTS,
                )
                break

            # Only retry attempts need to clean up: a clean deep-sleep exit
            # left no orphans and its GPU memory is being released async.
            if attempt > 1:
                await self._reap_orphan_engine()
                if not await self._wait_for_gpu_memory():
                    logger.warning(
                        "[deep-sleep] GPU memory still occupied before respawn "
                        "attempt %d/%d; attempting anyway",
                        attempt,
                        VLLM_SM75_RESPAWN_MAX_ATTEMPTS,
                    )

            t0 = time.monotonic()
            try:
                # wait_for() bounds a single attempt: the model-load step
                # (asyncio.to_thread) has no built-in timeout and can stall on
                # a slow disk, which would otherwise hang the gate forever.
                await asyncio.wait_for(
                    self.engine_core.respawn_engine(),
                    timeout=VLLM_SM75_RESPAWN_ATTEMPT_TIMEOUT_S,
                )
            except Exception as e:  # noqa: BLE001 - retried / reported below
                last_exc = e
                logger.warning(
                    "[deep-sleep] engine respawn attempt %d/%d failed after "
                    "%.1fs: %s",
                    attempt,
                    VLLM_SM75_RESPAWN_MAX_ATTEMPTS,
                    time.monotonic() - t0,
                    e,
                )
                if attempt < VLLM_SM75_RESPAWN_MAX_ATTEMPTS:
                    await asyncio.sleep(VLLM_SM75_RESPAWN_BACKOFF_S)
                continue

            logger.info(
                "[deep-sleep] engine respawned in %.1fs (attempt %d)",
                time.monotonic() - t0,
                attempt,
            )
            self._deep_sleep_respawn_pending = False
            return

        # All attempts failed (or the total budget ran out).  Deliberately
        # leave `engine_sleeping` set: the liveness monitor treats an exit
        # while sleeping as intentional (it keeps the client alive), whereas
        # clearing it would make the monitor treat the failure as a crash and
        # shut the client down for good.  The pending flag lets a later
        # request retry.  Note: while the engine is down the liveness monitor
        # has nothing to watch (respawn_engine restarts it on success), so the
        # brief monitoring gap is harmless.
        self._deep_sleep_respawn_pending = True
        logger.error(
            "[deep-sleep] engine respawn failed after %d attempts; the next "
            "request will retry",
            VLLM_SM75_RESPAWN_MAX_ATTEMPTS,
        )
        raise EngineDeadError() from last_exc

    async def _reap_orphan_engine(self) -> None:
        """Shut down engine processes left behind by a failed respawn.

        Those processes hold GPU memory but are unreachable because the API
        server never completed the handshake with them.

        NOTE: when a prior attempt left a *live but unreachable* engine, the
        liveness monitor thread is still blocked in
        ``monitor_engine_liveness()``.  ``shutdown()`` terminates that process,
        which is what unblocks the monitor; the two operate on the same process
        lifecycle (the manager serializes terminate/join), so this is safe.
        """
        def _shutdown() -> None:
            manager = getattr(self.engine_core.resources, "engine_manager", None)
            if manager is None:
                return
            try:
                manager.shutdown(timeout=30)
            except Exception as e:  # noqa: BLE001 - best effort cleanup
                logger.warning(
                    "[deep-sleep] orphan engine cleanup failed: %s", e
                )

        await asyncio.to_thread(_shutdown)

    async def _wait_for_gpu_memory(self) -> bool:
        """Wait until every visible GPU has enough free memory for the
        configured ``gpu_memory_utilization``.

        Returns True when it is safe to respawn (enough free, or the free
        memory cannot be determined, in which case we do not gate), and False
        only when GPUs are still occupied after the wait timeout.
        """
        try:
            utilization = self.vllm_config.cache_config.gpu_memory_utilization
        except Exception:  # noqa: BLE001 - fall back to a sane default
            utilization = 0.9

        def _enough_free() -> bool | None:
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.free,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except Exception:  # noqa: BLE001 - cannot query; do not block
                return None
            if proc.returncode != 0:
                return None
            enough = True
            for line in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    free_mib, total_mib = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                if free_mib < total_mib * utilization:
                    enough = False
            return enough

        deadline = time.monotonic() + VLLM_SM75_RESPAWN_GPU_WAIT_S
        query_warned = False
        while True:
            result = await asyncio.to_thread(_enough_free)
            if result is None:
                # Cannot query nvidia-smi: do not gate the respawn on memory.
                if not query_warned:
                    logger.warning(
                        "[deep-sleep] cannot query nvidia-smi; proceeding "
                        "without a GPU-memory gate"
                    )
                    query_warned = True
                return True
            if result:
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(2)

    async def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest
    ) -> None:
        """
        Initialize weight transfer for RL training.

        Args:
            request: Weight transfer initialization request with backend-specific info
        """
        await self.collective_rpc(
            "init_weight_transfer_engine", kwargs={"init_info": request.init_info}
        )

    async def start_weight_update(self) -> None:
        """Start a new weight update."""
        await self.collective_rpc("start_weight_update")

    async def start_draft_weight_update(self) -> None:
        """Start a new weight update targeting the speculative draft model."""
        await self.collective_rpc("start_draft_weight_update")

    async def update_weights(self, request: WeightTransferUpdateRequest) -> None:
        """
        Batched weight update for RL training.

        Args:
            request: Weight update request with backend-specific update info
        """
        await self.collective_rpc(
            "update_weights", kwargs={"update_info": request.update_info}
        )

    async def finish_weight_update(self, weight_version: str | None = None) -> None:
        """Finish the weight update and set its version if provided."""
        await self.collective_rpc("finish_weight_update")
        if weight_version is not None:
            await self.update_weight_version(weight_version)

    async def update_weight_version(self, new_version: str) -> None:
        """Set the weight version without updating weights."""
        await self.engine_core.set_weight_version_async(new_version)

    async def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        return await self.engine_core.get_weight_version_async()
