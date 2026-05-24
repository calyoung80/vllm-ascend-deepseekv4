# SPDX-License-Identifier: Apache-2.0
import copy
import inspect
import os
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.sequence import IntermediateTensors
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_model_parallel_group,
    patch_tensor_parallel_group,
)
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger


def _mtp_meta_chain_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_MTP_META_CHAIN_TRACE", "0") == "1"


def _mtp_meta_chain_summary(metas) -> str:
    if metas is None:
        return "len=-1 first=-1 last=-1 layers=-1"
    if not isinstance(metas, list):
        return f"type={type(metas).__name__}"
    meta_len = len(metas)
    if meta_len == 0:
        return "len=0 first=-1 last=-1 layers=-1"
    first = metas[0]
    last = metas[-1]
    layer_count = len(first) if isinstance(first, dict) else -1
    return (
        f"len={meta_len} first={id(first)} last={id(last)} layers={layer_count}"
    )
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    compute_new_slot_mapping,
    extend_all_queries_by_N,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper, update_full_graph_params
from vllm_ascend.ops.triton.spec_decode.utils import prepare_inputs_padded_kernel
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
from vllm_ascend.utils import enable_sp, lmhead_tp_enable, shared_expert_dp_enabled

# Currently we will fix block size to a small one since `num_reqs` can't be too large
_PREPARE_INPUTS_BLOCK_SIZE = 4


@dataclass
class TailLiveState:
    step0_draft_token_ids: torch.Tensor
    live_token_indices_to_sample: torch.Tensor
    live_positions: torch.Tensor
    live_hidden_states: torch.Tensor
    state_batch_size: int
    sample_batch_size: int


# TODO: Remove it when the bug of fx-graph is solved
# patch vllm_config to be in CompilationMode.NONE temporarily
@contextmanager
def _maybe_eager_context(vllm_config):
    raw_compilation_config_mode = vllm_config.compilation_config.mode
    vllm_config.compilation_config.mode = CompilationMode.NONE
    try:
        yield
    finally:
        vllm_config.compilation_config.mode = raw_compilation_config_mode


# split hidden states along dimension of sequence
def split_inputs_tp_to_sp(hidden_states, out):
    # tp and sp share the same group
    group = get_tp_group()

    world_size = group.world_size
    rank = group.rank

    num_tokens = hidden_states.shape[0]
    # the size per rank after padded
    padded_num_tokens_per_rank = (num_tokens + world_size - 1) // world_size
    # compute the start and end of slice
    start = padded_num_tokens_per_rank * rank
    end = padded_num_tokens_per_rank * (rank + 1)

    # copy only hidden_states in current rank
    hidden_states_curr_rank = hidden_states[start:end]
    out[: hidden_states_curr_rank.shape[0]] = hidden_states_curr_rank
    return out[:padded_num_tokens_per_rank]


class _FusedModelWithMTP:
    """Wraps the main model forward together with ALL MTP steps.

    Used as the ``runnable`` of :class:`ACLGraphWrapper` so that both the
    main-model forward **and** all N MTP speculative steps are captured
    into a single ACLGraph.  On replay the entire fused graph runs in one
    launch on the main stream — no separate graph or stream-sync for MTP.

    Attribute access is transparently delegated to ``raw_model`` so that
    call-sites like ``self.model.compute_logits(...)`` keep working.
    """

    def __init__(self, raw_model: nn.Module, drafter: "SpecDecodeBaseProposer"):
        self.raw_model = raw_model
        self.drafter = drafter
        num_spec_tokens = drafter.num_speculative_tokens
        max_num_reqs = drafter.runner.max_num_reqs
        max_num_tokens = drafter.runner.max_num_tokens
        device = drafter.device
        self.logits_indices_buf = torch.zeros(
            max_num_reqs * (1 + num_spec_tokens), dtype=torch.int64, device=device)
        self.draft_token_ids_buf = torch.zeros(
            (max_num_reqs, num_spec_tokens), dtype=torch.int64, device=device)
        self.main_next_token_ids_buf = torch.zeros(
            (max_num_reqs,), dtype=torch.int64, device=device)
        self.mtp_last_hidden_states_buf = torch.zeros(
            (max_num_tokens, drafter.hidden_size),
            dtype=drafter.dtype, device=device)
        self._state_debug = os.getenv("VLLM_ASCEND_MTP_FUSED_STATE_DEBUG", "0") == "1"
        self._state_debug_interval = int(os.getenv("VLLM_ASCEND_MTP_FUSED_STATE_DEBUG_INTERVAL", "50"))
        self._state_debug_counter = 0
        self._force_state_debug_once = False

    def __getattr__(self, key: str):
        return getattr(self.raw_model, key)

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.raw_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        forward_context = get_forward_context()
        is_capturing = getattr(forward_context, 'capturing', False)
        
        num_tokens_dbg = input_ids.shape[0]
        num_spec_dbg = self.drafter.num_speculative_tokens
        num_actual_tokens_dbg = int(getattr(forward_context, "num_actual_tokens", num_tokens_dbg))
        batch_size_dbg = max(num_actual_tokens_dbg // (num_spec_dbg + 1), 1)
        cudagraph_mode_dbg = str(getattr(forward_context, 'cudagraph_runtime_mode', 'NONE'))
        
        # Capture-safe entry log: only print shapes in capture mode, no CPU sync
        wrapper_force_debug_once = bool(getattr(self.drafter, "_force_state_debug_once", False))
        emit_wrapper_debug = (self._state_debug or wrapper_force_debug_once)
        if is_capturing and self._state_debug:
            logger.info(
                "[MTP_FUSED_DEBUG] B2_wrapper_entry_capture is_capturing=True cudagraph_mode=%s num_tokens=%d batch_size=%d draft_buf_shape=%s li_buf_shape=%s",
                cudagraph_mode_dbg,
                num_tokens_dbg,
                batch_size_dbg,
                tuple(self.draft_token_ids_buf.shape),
                tuple(self.logits_indices_buf.shape),
            )
        
        # Only run fused draft generation during graph capture/replay setup.
        # Profile/warmup paths may invoke this runnable outside capture, where
        # the draft-step shape assumptions do not hold. Prefill still skips
        # draft execution because hidden_states is IntermediateTensors there.
        if is_capturing and not isinstance(hidden_states, IntermediateTensors):
            raw_hidden = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
            if getattr(forward_context, 'flash_comm_v1_enabled', False):
                from vllm.distributed import tensor_model_parallel_all_gather
                raw_hidden = tensor_model_parallel_all_gather(raw_hidden, 0)
                pad_size = getattr(forward_context, 'pad_size', 0)
                if pad_size > 0:
                    raw_hidden = raw_hidden[:-pad_size, :]
            num_tokens = input_ids.shape[0]
            num_spec = self.drafter.num_speculative_tokens
            num_actual_tokens = int(getattr(forward_context, "num_actual_tokens", num_tokens))
            batch_size = max(num_actual_tokens // (num_spec + 1), 1)
            if emit_wrapper_debug and (not is_capturing) and batch_size == 1:
                try:
                    li_dbg = self.logits_indices_buf[: min(4, self.logits_indices_buf.shape[0])].detach().to("cpu").tolist()
                    logger.info(
                        "[MTP_FUSED_DEBUG] fused_wrapper_step0_indices num_tokens=%d batch_size=%d li_head=%s",
                        num_tokens,
                        batch_size,
                        li_dbg,
                    )
                except Exception as e:
                    logger.warning("[MTP_FUSED_DEBUG] fused_wrapper_step0_indices failed: %r", e)
            step0_logits_indices = self.logits_indices_buf[:batch_size].clone().to(
                dtype=torch.long)
            sample_hs = raw_hidden[step0_logits_indices]
            main_logits = self.raw_model.compute_logits(sample_hs)
            next_token_ids = main_logits.argmax(dim=-1)
            self.main_next_token_ids_buf[:batch_size].copy_(next_token_ids[:batch_size])
            all_draft_ids = self.drafter.propose_all_in_graph(
                hidden_states=raw_hidden,
                input_ids=input_ids,
                positions=positions,
                logits_indices=self.logits_indices_buf,
                step0_logits_indices=step0_logits_indices,
                next_token_ids=next_token_ids,
                num_tokens=num_tokens,
            )
            num_reqs = all_draft_ids.shape[0]
            if emit_wrapper_debug and (not is_capturing) and num_reqs == 1:
                try:
                    draft_head_dbg = all_draft_ids[:1, : min(4, all_draft_ids.shape[1])].detach().to("cpu").tolist()
                    buf_head_pre_dbg = self.draft_token_ids_buf[:1, : min(4, self.draft_token_ids_buf.shape[1])].detach().to("cpu").tolist()
                    next_head_dbg = next_token_ids[: min(4, next_token_ids.shape[0])].detach().to("cpu").tolist()
                    logger.info(
                        "[MTP_FUSED_DEBUG] fused_wrapper_draft_write_pre next_head=%s all_draft_head=%s draft_buf_pre=%s",
                        next_head_dbg,
                        draft_head_dbg,
                        buf_head_pre_dbg,
                    )
                except Exception as e:
                    logger.warning("[MTP_FUSED_DEBUG] fused_wrapper_draft_write_pre failed: %r", e)
            self.draft_token_ids_buf[:num_reqs, :all_draft_ids.shape[1]].copy_(
                all_draft_ids)
            if emit_wrapper_debug and (not is_capturing) and num_reqs == 1:
                try:
                    buf_head_post_dbg = self.draft_token_ids_buf[:1, : min(4, self.draft_token_ids_buf.shape[1])].detach().to("cpu").tolist()
                    logger.info(
                        "[MTP_FUSED_DEBUG] fused_wrapper_draft_write_post draft_buf_post=%s",
                        buf_head_post_dbg,
                    )
                except Exception as e:
                    logger.warning("[MTP_FUSED_DEBUG] fused_wrapper_draft_write_post failed: %r", e)
        else:
            # Prefill phase: hidden_states is IntermediateTensors, skip draft execution.
            if self._state_debug:
                logger.info(
                    "[MTP_FUSED_DEBUG] fused_wrapper_prefill_skip hidden_states_type=IntermediateTensors num_tokens=%d batch_size=%d",
                    num_tokens_dbg,
                    batch_size_dbg,
                )
        return hidden_states


class SpecDecodeBaseProposer(EagleProposer):
    _runnable: ACLGraphWrapper | Callable

    def __init__(self, vllm_config: VllmConfig, device: torch.device, pass_hidden_states_to_model: bool, runner=None):
        super().__init__(vllm_config, device, runner)

        self.use_async_scheduling = self.vllm_config.scheduler_config.async_scheduling
        self.use_compress = hasattr(self.vllm_config.model_config.hf_config, "compress_ratios")
        self.pass_hidden_states_to_model = pass_hidden_states_to_model
        self.decode_threshold = 1 + self.num_speculative_tokens
        self.query_start_loc = self.runner._make_buffer(self.runner.max_num_reqs + 2, dtype=torch.int32)
        self.arange_cpu = torch.arange(self.arange.shape[0], device="cpu", dtype=torch.int32)
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

        self.enable_shared_expert_dp = shared_expert_dp_enabled()

        self.pcp_size = self.runner.pcp_size
        self.dcp_size = self.runner.dcp_size
        self.pcp_rank = self.runner.pcp_rank
        self.dcp_rank = self.runner.dcp_rank

        self.full_indices = range(
            self.runner.max_num_tokens * self.pcp_size * self.dcp_size
            + self.pcp_size * self.dcp_size * self.runner.max_num_reqs
        )

        self.use_sparse = hasattr(vllm_config.model_config.hf_text_config, "index_topk")
        # NOTE:
        # `draft_tensor_parallel_size` does not take effect for Eagle:
        # the draft model uses the same TP size as the target model in practice.
        # so we applied this patch to set tp=1 of draft model separately.
        # Due to verification of `_verify_and_get_draft_tp` in vllm,
        # the value of `draft_tensor_parallel_size` here will either be 1 separately
        # or the same as target model.
        # TODO(zhaomingyu13): If we want to adapt to the case where draft model tp
        # is not 1 and differs from target model, this part should be rewritten.
        if vllm_config.parallel_config.tensor_parallel_size != self.speculative_config.draft_tensor_parallel_size:
            tp_group = init_model_parallel_group(
                [[get_world_group().rank]],
                get_world_group().rank,
                torch.distributed.get_backend(get_world_group().device_group),
                use_message_queue_broadcaster=True,
                group_name="tp",
            )
            self.tp_group_context = patch_tensor_parallel_group(tp_group)
        else:
            self.tp_group_context = nullcontext()

        self.use_cuda_graph = self.runner._use_aclgraph() and not self.speculative_config.enforce_eager
        if self.method == "mtp":
            self.use_cuda_graph = (
                self.use_cuda_graph
                and not self.use_async_scheduling
                and not self.speculative_config.disable_padded_drafter_batch
                and not self.use_compress
            )

        # TODO: Remove it when the bug of fx-graph is solved
        self.maybe_eager_context: AbstractContextManager[Any] = nullcontext()
        if not self.use_cuda_graph and enable_sp(vllm_config):
            self.maybe_eager_context = _maybe_eager_context(vllm_config)

        self.token_indices_to_sample = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens, dtype=torch.int32, device=device
        )
        slot_mapping_lens = self.runner.max_num_tokens + 2 * self.pcp_size * self.runner.max_num_reqs
        self.slot_mapping_group = [
            torch.zeros(slot_mapping_lens, dtype=torch.int32, device=device, pin_memory=self.runner.pin_memory)
            for _ in range(self.num_speculative_tokens)
        ]

        self._runnable = self._run_merged_draft
        self._tail_live_runnable = self._run_compact_tail_live_draft
        self.query_lens = torch.ones(
            self.vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.int32,
            device=self.device,
        )
        self.is_multimodal_model = self.vllm_config.model_config.is_multimodal_model
        if self.uses_mrope:
            self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1), dtype=torch.int32, device=device)
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions = torch.zeros(
                (self.uses_xdrope_dim, self.max_num_tokens + 1),
                dtype=torch.int32,
                device=device,
            )
        else:
            # RoPE need (max_num_tokens,)
            self.positions = torch.zeros(self.max_num_tokens, dtype=torch.int32, device=device)

        self.token_arange_np = np.arange(self.max_num_tokens + 1)

        # Experimental compact tail-live graph is currently unstable on the
        # deployed stack (rotary dim0 mismatch). Force-disable to keep runtime
        # correctness while preserving code paths for future re-enable.
        _tail_live_graph_requested = (
            os.getenv("VLLM_ASCEND_MTP_TAIL_LIVE_GRAPH", "0") == "1")
        if _tail_live_graph_requested:
            logger.warning(
                "[MTP_FUSED_DEBUG] VLLM_ASCEND_MTP_TAIL_LIVE_GRAPH is temporarily disabled due to rotary dim0 mismatch"
            )
        self._tail_live_graph_enabled = False
        self._tail_live_graph_path_logged = False

        # NPU aclnnIndexPutImpl in fused MTP path requires int64 self tensor.
        if self.method == "mtp" and hasattr(self, "input_ids") and self.input_ids.dtype != torch.int64:
            self.input_ids = self.input_ids.to(torch.int64)

    def _get_model(self) -> nn.Module:
        """
        Default method to call get_model(). Can be overridden by subclasses which
        need to customize model loading.
        """
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("eagle_head"):
            model = get_model(
                vllm_config=self.vllm_config,
                model_config=self.vllm_config.speculative_config.draft_model_config,
            )
        return model

    def load_model(self, model: nn.Module) -> None:
        target_attn_layer_names = set(get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys())

        with self.maybe_eager_context:
            self.model = self._get_model()

        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        all_indexer_layer_names = set(get_layers_from_vllm_config(self.vllm_config, DeepseekV32IndexerCache).keys())
        
        # Filter to only layers that have KV cache specs.
        self._draft_attn_layer_names = {
            name
            for name in (set(all_attn_layers.keys()) - target_attn_layer_names)
            if all_attn_layers[name].get_kv_cache_spec(self.vllm_config) is not None
        } - all_indexer_layer_names

        self.attn_layer_names = list(sorted(self._draft_attn_layer_names))
        draft_attn_layers_dict = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        self.kernel_block_size = (
            draft_attn_layers_dict[self.attn_layer_names[0]].get_attn_backend().get_supported_kernel_block_sizes()[0]
        )

        self.piece_all_attn_layer_name = []
        for _ in range(self.num_speculative_tokens):
            self.piece_all_attn_layer_name.append([name for name in self.attn_layer_names])

        if supports_multimodal(model):
            # handle multimodality
            if self.get_model_name(model) in [
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
            ]:
                self.model.config.image_token_index = model.config.image_token_id
            elif self.get_model_name(model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = model.config.vision_config.image_token_id
            elif self.get_model_name(model) == "KimiK25ForConditionalGeneration":
                self.model.config.image_token_index = model.config.media_placeholder_token_id
            else:
                self.model.config.image_token_index = model.config.image_token_index
            target_language_model = model.get_language_model()
        else:
            target_language_model = model

        # share embed_tokens with the target model if needed
        self._maybe_share_embeddings(target_language_model)
        self._maybe_share_topk_indices(target_language_model)
        self._maybe_share_lm_head(model)

        if self.parallel_drafting and self.pass_hidden_states_to_model:
            assert self.parallel_drafting_hidden_state_tensor is not None
            self.parallel_drafting_hidden_state_tensor.copy_(
                self.model.combine_hidden_states(self.model.mask_hidden.view(3 * self.hidden_size))
                if self.eagle3_use_aux_hidden_state
                else self.model.mask_hidden.view(self.hidden_size)
            )

    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        """
        Some draft models may not have their own embedding layers, and some may
        have a duplicate copy of the target model's embedding layers. In these cases,
        we share the target model's embedding layers with the draft model to save
        memory.
        """
        if get_pp_group().world_size == 1:
            if hasattr(target_language_model.model, "embed_tokens"):
                target_embed_tokens = target_language_model.model.embed_tokens
            elif hasattr(target_language_model.model, "embedding"):
                target_embed_tokens = target_language_model.model.embedding
            else:
                raise AttributeError("Target model does not have 'embed_tokens' or 'embedding' attribute")
            # If pp>1, the weights of mtp and the main model's embedding are not on the same device.
            # check if mtp model use main model's embedding and LMhead
            share_embeddings = False
            if hasattr(self.model, "has_own_embed_tokens"):
                # EAGLE model
                if not self.model.has_own_embed_tokens:
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model without its own embed_tokens in the"
                        " checkpoint. Sharing target model embedding weights with the"
                        " draft model."
                    )
                elif (
                    isinstance(target_embed_tokens.weight, torch.Tensor)
                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                    # TODO: Offload to CPU for comparison to avoid extra NPU memory
                    # usage in CI testing environments with limited NPU memory
                    and torch.equal(
                        target_embed_tokens.weight.cpu(),
                        self.model.model.embed_tokens.weight.cpu(),
                    )
                ):
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model with embed_tokens identical to the target"
                        " model. Sharing target model embedding weights with the draft"
                        " model."
                    )
                else:
                    logger.info(
                        "Detected EAGLE model with distinct embed_tokens weights. "
                        "Keeping separate embedding weights from the target model."
                    )
            else:
                # MTP model
                share_embeddings = not self.use_compress
                if share_embeddings:
                    logger.info(
                        "Detected MTP model. "
                        "Sharing target model embedding weights with the draft model."
                    )

            if share_embeddings:
                if hasattr(self.model.model, "embed_tokens"):
                    del self.model.model.embed_tokens
                self.model.model.embed_tokens = target_embed_tokens
        else:
            logger.info(
                "Since PP > 1 or other reasons the model head loaded its own vocab embedding"
                " weights instead of sharing them with the target model."
            )

    # share lm_head with the target model if needed
    def _maybe_share_lm_head(self, model: nn.Module) -> None:
        # some model definition do not define lm_head explicitly
        # and reuse embed_tokens for lm_head, e.g., CohereForCausalLM
        if self.method == "eagle" and hasattr(model, "lm_head"):
            logger.info("Loading EAGLE LM head weights from the target model.")
            if supports_multimodal(model):
                self.model.lm_head = model.get_language_model().lm_head
            else:
                self.model.lm_head = model.lm_head

        if self.method == "mtp" and self.vllm_config.model_config.is_deepseek_mla:
            for _, layer_module in self.model.model.layers.items():
                if torch.equal(layer_module.shared_head.head.weight, model.lm_head.weight):
                    layer_module.shared_head.head = model.lm_head

        if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs() and self.use_cuda_graph:
            self.update_stream = torch.npu.Stream()
            if self.method == "mtp":
                if not getattr(self, 'fused_with_main_graph', False):
                    self.model = ACLGraphWrapper(self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)
            else:
                self._runnable = ACLGraphWrapper(
                    self._run_merged_draft, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
                )

    def _maybe_share_topk_indices(self, target_language_model: nn.Module) -> None:
        if hasattr(target_language_model.model, "topk_indices_buffer"):
            if hasattr(self.model.model, "topk_indices_buffer"):
                del self.model.model.topk_indices_buffer
            self.model.model.topk_indices_buffer = (
                target_language_model.model.topk_indices_buffer
            )
            logger.info(
                "Detecting MTP model with topk_indices_buffer."
                "Sharing target model topk_indices_buffer with the draft model."
            )

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model, ACLGraphWrapper):
            return self.model.unwrap()
        return self.model

    def propose_all_in_graph(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        step0_logits_indices: torch.Tensor | None,
        next_token_ids: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Run ALL MTP steps inside the main model's ACLGraph.

        Called by :class:`_FusedModelWithMTP` so that main model forward
        plus all N MTP steps are recorded/replayed as a **single** graph.
        Returns ``draft_token_ids`` of shape ``(batch_size, num_speculative_tokens)``.
        """
        forward_context = get_forward_context()
        num_actual_tokens = int(getattr(forward_context, "num_actual_tokens", num_tokens))
        batch_size = max(num_actual_tokens // (self.num_speculative_tokens + 1), 1)
        raw_model = self.get_model()

        # Defensive init for debug attrs: some deployments may load proposer
        # instances that predate debug fields.
        if not hasattr(self, "_state_debug"):
            self._state_debug = os.getenv("VLLM_ASCEND_MTP_FUSED_STATE_DEBUG", "0") == "1"
        if not hasattr(self, "_state_debug_interval"):
            self._state_debug_interval = int(os.getenv("VLLM_ASCEND_MTP_FUSED_STATE_DEBUG_INTERVAL", "50"))
        if not hasattr(self, "_state_debug_counter"):
            self._state_debug_counter = 0
        if not hasattr(self, "_force_state_debug_once"):
            self._force_state_debug_once = False

        self._state_debug_counter += 1
        do_state_debug = (
            self._force_state_debug_once
            or (
                self._state_debug
                and self._state_debug_interval > 0
                and self._state_debug_counter % self._state_debug_interval == 0
            )
        )

        # Never do debug scalar/tensor reads in graph-capturing phase.
        # Ascend forbids stream synchronize on captured stream.
        in_capture = bool(getattr(forward_context, "capturing", False))
        if in_capture:
            if self._state_debug and self._state_debug_interval > 0 and self._state_debug_counter % self._state_debug_interval == 0:
                logger.info(
                    "[MTP_FUSED_DEBUG] graph_step0_capture_mode num_tokens=%d batch_size=%d logits_shape=%s input_dtype=%s next_dtype=%s",
                    num_tokens,
                    batch_size,
                    tuple(logits_indices.shape),
                    str(input_ids.dtype),
                    str(next_token_ids.dtype),
                )
            do_state_debug = False

        if self._state_debug and self._state_debug_interval > 0 and self._state_debug_counter % self._state_debug_interval == 0:
            logger.info(
                "[MTP_FUSED_DEBUG] graph_entry_ctx in_capture=%s counter=%d interval=%d num_tokens=%d batch_size=%d logits_shape=%s input_shape=%s positions_shape=%s next_shape=%s",
                str(in_capture),
                self._state_debug_counter,
                self._state_debug_interval,
                num_tokens,
                batch_size,
                tuple(logits_indices.shape),
                tuple(input_ids.shape),
                tuple(positions.shape),
                tuple(next_token_ids.shape),
            )

        # Flush one capture-safe snapshot at first non-capture opportunity.
        if (not in_capture) and hasattr(self, "_capture_step0_snapshot"):
            try:
                snap = self._capture_step0_snapshot
                logger.info(
                    "[MTP_FUSED_DEBUG] graph_step0_capture_snapshot num_tokens=%d batch_size=%d li_head=%s pre_head=%s post_head=%s",
                    int(snap["num_tokens"]),
                    int(snap["batch_size"]),
                    snap["li_head"].detach().to("cpu").tolist(),
                    snap["pre_head"].detach().to("cpu").tolist(),
                    snap["post_head"].detach().to("cpu").tolist(),
                )
            except Exception as _e:
                logger.warning("[MTP_FUSED_DEBUG] graph_step0_capture_snapshot flush failed: %s", repr(_e))
            finally:
                delattr(self, "_capture_step0_snapshot")

        if step0_logits_indices is None:
            step0_logits_indices = logits_indices[:batch_size].clone().to(
                dtype=torch.long)

        li_head_dbg = logits_indices[: min(8, logits_indices.shape[0])]
        pre_in_head_dbg = self.input_ids[: min(8, num_tokens)].detach().to("cpu") if do_state_debug else None
        pre_in_li0_dbg = None
        li0_dbg = -1
        if do_state_debug:
            li0_dbg = int(logits_indices[0].item()) if logits_indices.numel() > 0 else -1
            if li0_dbg >= 0 and li0_dbg < self.input_ids.shape[0]:
                pre_in_li0_dbg = int(self.input_ids[li0_dbg].item())

        # Assemble step-0 input ids in local tensor to avoid shared-buffer state pollution.
        work_input_ids = self.input_ids[:num_tokens].clone()
        shifted_input_ids = input_ids[1:num_tokens].clone()
        work_input_ids[: num_tokens - 1] = shifted_input_ids

        # Prefer index_copy_ over advanced indexing assignment in graph mode.
        # This makes index write semantics explicit and more stable for capture/replay.
        li = step0_logits_indices.contiguous()
        if num_tokens > 0:
            li = li.clamp_(0, num_tokens - 1)
        next_ids_cast = next_token_ids[:batch_size].to(work_input_ids.dtype)
        work_input_ids.index_copy_(0, li, next_ids_cast)

        self._set_positions(num_tokens, positions[:num_tokens])
        self.hidden_states[:num_tokens].copy_(hidden_states[:num_tokens])

        # Capture-safe probe: keep tensors on device, no scalar reads or CPU copies.
        if in_capture and self._state_debug:
            head_n = min(8, num_tokens)
            li_n = min(8, logits_indices.shape[0])
            self._capture_step0_snapshot = {
                "num_tokens": num_tokens,
                "batch_size": batch_size,
                "li_head": logits_indices[:li_n].clone(),
                "pre_head": self.input_ids[:head_n].clone(),
                "post_head": work_input_ids[:head_n].clone(),
            }
            logger.info(
                "[MTP_FUSED_DEBUG] graph_step0_capture_meta num_tokens=%d batch_size=%d li_slice_shape=%s input_shape=%s work_shape=%s input_stride=%s work_stride=%s input_off=%d work_off=%d input_ptr=%d work_ptr=%d li_dtype=%s input_dtype=%s work_dtype=%s",
                num_tokens,
                batch_size,
                tuple(logits_indices[:batch_size].shape),
                tuple(self.input_ids[:num_tokens].shape),
                tuple(work_input_ids.shape),
                tuple(self.input_ids.stride()),
                tuple(work_input_ids.stride()),
                int(self.input_ids.storage_offset()),
                int(work_input_ids.storage_offset()),
                int(self.input_ids.data_ptr()),
                int(work_input_ids.data_ptr()),
                str(logits_indices.dtype),
                str(self.input_ids.dtype),
                str(work_input_ids.dtype),
            )

        if do_state_debug:
            post_in_head_dbg = work_input_ids[: min(8, num_tokens)].detach().to("cpu")
            post_in_li0_dbg = -1
            if li0_dbg >= 0 and li0_dbg < work_input_ids.shape[0]:
                post_in_li0_dbg = int(work_input_ids[li0_dbg].item())
            next0_dbg = int(next_token_ids[0].item()) if next_token_ids.numel() > 0 else -1
            write_ok = (post_in_li0_dbg == next0_dbg) if li0_dbg >= 0 else False
            logger.info(
                "[MTP_FUSED_DEBUG] graph_step0_write_check li0=%d post_li0=%d next0=%d ok=%s",
                li0_dbg,
                post_in_li0_dbg,
                next0_dbg,
                str(write_ok),
            )
            logger.info(
                "[MTP_FUSED_DEBUG] graph_step0_prepost num_tokens=%d batch_size=%d li0=%d li_head=%s pre_li0=%s post_li0=%d next0=%d pre_head=%s post_head=%s",
                num_tokens,
                batch_size,
                li0_dbg,
                li_head_dbg.detach().to("cpu").tolist(),
                str(pre_in_li0_dbg),
                post_in_li0_dbg,
                next0_dbg,
                pre_in_head_dbg.tolist() if pre_in_head_dbg is not None else [],
                post_in_head_dbg.tolist(),
            )

        model_input_ids = work_input_ids
        model_positions = self._get_positions(num_tokens)
        model_hidden_states = self.hidden_states[:num_tokens]
        model_hidden_states, model_positions = self.maybe_pad_and_reduce(
            model_hidden_states, model_positions)

        model_kwargs: dict[str, torch.Tensor] = {
            "input_ids": model_input_ids,
            "positions": model_positions,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = model_hidden_states
            if self.method == "mtp":
                model_kwargs["positions"] = model_positions

        ret_hidden_states = raw_model(**model_kwargs)
        if not self.model_returns_tuple():
            last_hidden_states = ret_hidden_states
            hidden_states_out = last_hidden_states
        else:
            last_hidden_states, hidden_states_out = ret_hidden_states

        last_hidden_states, model_positions, hidden_states_out = (
            self.maybe_all_gather_and_unpad(
                last_hidden_states, model_positions, hidden_states_out))

        sample_hs = last_hidden_states[step0_logits_indices]
        if do_state_debug:
            hs_shape = tuple(sample_hs.shape)
            hs_l2 = float(sample_hs.float().pow(2).mean().sqrt().item()) if sample_hs.numel() > 0 else 0.0
            li_runtime_dbg = step0_logits_indices[: min(4, batch_size)].detach().to("cpu").tolist()
            logger.info(
                "[MTP_FUSED_DEBUG] graph_step0_sample_hs li_head=%s runtime_li_head=%s hs_shape=%s hs_l2=%.6f",
                li_head_dbg.detach().to("cpu").tolist(),
                li_runtime_dbg,
                hs_shape,
                hs_l2,
            )
            self._force_state_debug_once = False
        logits = raw_model.compute_logits(sample_hs)
        draft_token_ids = logits.argmax(dim=-1)
        forward_context = get_forward_context()
        draft_attn_metadatas = getattr(
            forward_context, 'draft_attn_metadatas', None)
        capture_step_snapshots = [] if (in_capture and _mtp_meta_chain_enabled()) else None
        if capture_step_snapshots is not None:
            capture_step_snapshots.append({
                "step": 0,
                "input_head": work_input_ids[: min(4, work_input_ids.shape[0])].clone(),
                "pos_head": positions[: min(4, positions.shape[0])].clone(),
                "sample_idx_head": step0_logits_indices[: min(4, step0_logits_indices.shape[0])].clone(),
                "draft_head": draft_token_ids[: min(4, draft_token_ids.shape[0])].clone(),
                "meta_len": len(draft_attn_metadatas) if draft_attn_metadatas is not None else -1,
                "selected_meta_id": id(draft_attn_metadatas[0]) if draft_attn_metadatas and len(draft_attn_metadatas) > 0 else -1,
            })
        if do_state_debug:
            try:
                vocab_n = min(4, logits.shape[-1]) if logits.ndim > 1 else 0
                draft_head_dbg = draft_token_ids[: min(4, draft_token_ids.shape[0])].detach().to("cpu").tolist()
                next_head_dbg = next_token_ids[: min(4, next_token_ids.shape[0])].detach().to("cpu").tolist() if isinstance(next_token_ids, torch.Tensor) else []
                logits_topk_dbg = []
                logits_topv_dbg = []
                if logits.ndim == 2 and logits.shape[0] > 0 and vocab_n > 0:
                    topv, topi = torch.topk(logits[:1], k=vocab_n, dim=-1)
                    logits_topk_dbg = topi[0].detach().to("cpu").tolist()
                    logits_topv_dbg = [float(x) for x in topv[0].detach().to("cpu").tolist()]
                logger.info(
                    "[MTP_FUSED_DEBUG] graph_step0_logits_probe next_head=%s draft_head=%s top_idx=%s top_val=%s",
                    next_head_dbg,
                    draft_head_dbg,
                    logits_topk_dbg,
                    logits_topv_dbg,
                )
            except Exception as _e:
                logger.warning("[MTP_FUSED_DEBUG] graph_step0_logits_probe failed: %s", repr(_e))

        # Sentinel: detect suspicious zero-heavy draft outputs in fused path.
        # Only run in non-capture debug mode to avoid stream-sync risks.
        if do_state_debug:
            try:
                zero_cnt = int((draft_token_ids == 0).sum().item()) if draft_token_ids.numel() > 0 else 0
                all_zero = (draft_token_ids.numel() > 0 and zero_cnt == int(draft_token_ids.numel()))
                # Trigger on all-zero or very high zero ratio with non-trivial batch.
                if all_zero or (draft_token_ids.numel() >= 8 and zero_cnt * 100 >= int(draft_token_ids.numel()) * 80):
                    li_dbg = li_head_dbg.detach().to("cpu").tolist()
                    in_head_dbg = self.input_ids[: min(16, num_tokens)].detach().to("cpu").tolist()
                    next_head_dbg = next_token_ids[: min(8, next_token_ids.shape[0])].detach().to("cpu").tolist() if isinstance(next_token_ids, torch.Tensor) else []
                    draft_head_dbg = draft_token_ids[: min(8, draft_token_ids.shape[0])].detach().to("cpu").tolist()
                    logger.warning(
                        "[MTP_FUSED_DEBUG] draft_zero_sentinel num_tokens=%d batch_size=%d zeros=%d total=%d all_zero=%s li_head=%s next_head=%s draft_head=%s input_ids_head16=%s",
                        num_tokens,
                        batch_size,
                        zero_cnt,
                        int(draft_token_ids.numel()),
                        str(all_zero),
                        li_dbg,
                        next_head_dbg,
                        draft_head_dbg,
                        in_head_dbg,
                    )
            except Exception as _e:
                logger.warning("[MTP_FUSED_DEBUG] draft_zero_sentinel failed: %s", repr(_e))

        if _mtp_meta_chain_enabled():
            logger.info(
                "[MTP_META_CHAIN] proposer_entry %s",
                _mtp_meta_chain_summary(draft_attn_metadatas),
            )

        has_meta_dbg = bool(draft_attn_metadatas)
        meta_len_dbg = len(draft_attn_metadatas) if draft_attn_metadatas is not None else -1
        # Cache runtime metadata status for out-of-context shadow diagnostics.
        self._last_graph_meta_status = {
            "has_meta": has_meta_dbg,
            "meta_len": meta_len_dbg,
            "total_steps": self.num_speculative_tokens,
            "active_meta_id": id(getattr(forward_context, "attn_metadata", None)),
            "selected_meta_id": id(draft_attn_metadatas[0]) if has_meta_dbg else -1,
            "selected_meta_type": type(draft_attn_metadatas[0]).__name__ if has_meta_dbg else "None",
        }

        if self._state_debug:
            logger.info(
                "[MTP_FUSED_DEBUG] graph_step_meta_status has_meta=%s meta_len=%d total_steps=%d",
                str(draft_attn_metadatas is not None),
                meta_len_dbg,
                self.num_speculative_tokens,
            )
            if not draft_attn_metadatas:
                logger.warning(
                    "[MTP_FUSED_DEBUG] graph_step_meta_missing total_steps=%d draft_attn_metadatas=%s",
                    self.num_speculative_tokens,
                    "None_or_empty",
                )

        if self.num_speculative_tokens == 1:
            if draft_attn_metadatas and len(draft_attn_metadatas) > 0:
                forward_context.attn_metadata = draft_attn_metadatas[0]
                self._last_graph_meta_status = {
                    "has_meta": True,
                    "meta_len": len(draft_attn_metadatas),
                    "total_steps": self.num_speculative_tokens,
                    "active_meta_id": id(getattr(forward_context, "attn_metadata", None)),
                    "selected_meta_id": id(draft_attn_metadatas[0]),
                    "selected_meta_type": type(getattr(forward_context, "attn_metadata", None)).__name__,
                }
                if self._state_debug:
                    logger.info(
                        "[MTP_FUSED_DEBUG] graph_step_meta_switch step=%d total_steps=%d meta_len=%d active_meta_id=%d selected_meta_id=%d active_meta_type=%s",
                        0,
                        self.num_speculative_tokens,
                        len(draft_attn_metadatas),
                        id(getattr(forward_context, "attn_metadata", None)),
                        id(draft_attn_metadatas[0]),
                        type(getattr(forward_context, "attn_metadata", None)).__name__,
                    )
            return draft_token_ids.view(-1, 1)

        tail_state = self._build_fused_tail_runtime_state(
            step0_draft_token_ids=draft_token_ids,
            step0_logits_indices=step0_logits_indices,
            hidden_states_out=hidden_states_out,
            live_batch_size=batch_size,
        )

        # Ensure draft step-0 uses drafter metadata instead of main-model metadata.
        if draft_attn_metadatas and len(draft_attn_metadatas) > 0:
            forward_context.attn_metadata = draft_attn_metadatas[0]
            if self._state_debug:
                logger.info(
                    "[MTP_FUSED_DEBUG] graph_step_meta_switch step=%d total_steps=%d meta_len=%d active_meta_id=%d selected_meta_id=%d active_meta_type=%s",
                    0,
                    self.num_speculative_tokens,
                    len(draft_attn_metadatas),
                    id(getattr(forward_context, "attn_metadata", None)),
                    id(draft_attn_metadatas[0]),
                    type(getattr(forward_context, "attn_metadata", None)).__name__,
                )

        draft_token_ids_tensor, capture_step_snapshots = (
            self._run_fused_tail_steps(
                raw_model=raw_model,
                forward_context=forward_context,
                draft_attn_metadatas=draft_attn_metadatas,
                tail_state=tail_state,
                num_tokens=num_tokens,
                work_input_ids=work_input_ids,
                do_state_debug=do_state_debug,
                capture_step_snapshots=capture_step_snapshots,
            ))

        if capture_step_snapshots is not None:
            self._capture_multistep_snapshot = capture_step_snapshots
        return draft_token_ids_tensor.swapaxes(0, 1)

    def _build_fused_tail_state(
        self,
        step0_draft_token_ids: torch.Tensor,
        step0_logits_indices: torch.Tensor,
        hidden_states_out: torch.Tensor,
    ) -> TailLiveState:
        # Preserve the current fused behavior: later steps inherit all lanes
        # addressed by step0_logits_indices, not just the business-level live
        # request count. A later tail live-lane graph will intentionally
        # replace this with compact live-lane state construction.
        step_batch_size = int(step0_logits_indices.shape[0])
        return self._make_tail_live_state(
            step0_draft_token_ids=step0_draft_token_ids,
            live_token_indices_to_sample=self.arange[:step_batch_size],
            live_positions=self.positions[step0_logits_indices],
            live_hidden_states=hidden_states_out[step0_logits_indices],
            state_batch_size=step_batch_size,
            sample_batch_size=step_batch_size,
        )

    def _build_fused_tail_runtime_state(
        self,
        step0_draft_token_ids: torch.Tensor,
        step0_logits_indices: torch.Tensor,
        hidden_states_out: torch.Tensor,
        live_batch_size: int,
    ) -> TailLiveState:
        if getattr(self, "_tail_live_graph_enabled", False):
            if not getattr(self, "_tail_live_graph_path_logged", False):
                logger.info(
                    "[MTP_FUSED_DEBUG] fused tail runtime selects compact live-lane state"
                )
                self._tail_live_graph_path_logged = True
            return self._build_compact_tail_live_state(
                step0_draft_token_ids=step0_draft_token_ids,
                step0_logits_indices=step0_logits_indices,
                hidden_states_out=hidden_states_out,
                live_batch_size=live_batch_size,
            )
        return self._build_fused_tail_state(
            step0_draft_token_ids=step0_draft_token_ids,
            step0_logits_indices=step0_logits_indices,
            hidden_states_out=hidden_states_out,
        )

    def _build_compact_tail_live_state(
        self,
        step0_draft_token_ids: torch.Tensor,
        step0_logits_indices: torch.Tensor,
        hidden_states_out: torch.Tensor,
        live_batch_size: int,
    ) -> TailLiveState:
        """Build a compact live-lane tail state for future tail-graph use.

        Unlike `_build_fused_tail_state`, which preserves the current graph's
        full companion-lane width, this helper intentionally narrows the tail
        state to the business-level live request lanes. It is not wired into
        the runtime path yet; it serves as the structural entry point for the
        eventual `tail live-lane graph` refactor.
        """
        live_batch_size = max(int(live_batch_size), 1)
        compact_indices = step0_logits_indices[:live_batch_size]
        return self._make_tail_live_state(
            step0_draft_token_ids=step0_draft_token_ids[:live_batch_size],
            live_token_indices_to_sample=self.arange[:live_batch_size],
            live_positions=self.positions[compact_indices],
            live_hidden_states=hidden_states_out[compact_indices],
            state_batch_size=live_batch_size,
            sample_batch_size=live_batch_size,
        )

    def _make_tail_live_state(
        self,
        step0_draft_token_ids: torch.Tensor,
        live_token_indices_to_sample: torch.Tensor,
        live_positions: torch.Tensor,
        live_hidden_states: torch.Tensor,
        state_batch_size: int,
        sample_batch_size: int,
    ) -> TailLiveState:
        return TailLiveState(
            step0_draft_token_ids=step0_draft_token_ids,
            live_token_indices_to_sample=live_token_indices_to_sample,
            live_positions=live_positions,
            live_hidden_states=live_hidden_states,
            state_batch_size=int(state_batch_size),
            sample_batch_size=int(sample_batch_size),
        )

    def _run_fused_tail_steps(
        self,
        raw_model: nn.Module,
        forward_context,
        draft_attn_metadatas,
        tail_state: TailLiveState,
        num_tokens: int,
        work_input_ids: torch.Tensor,
        do_state_debug: bool,
        capture_step_snapshots: list[dict[str, Any]] | None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]] | None]:
        draft_token_ids_tensor = torch.zeros(
            (self.num_speculative_tokens, *tail_state.step0_draft_token_ids.shape),
            dtype=tail_state.step0_draft_token_ids.dtype,
            device=self.device,
        )
        draft_token_ids_tensor[0] = tail_state.step0_draft_token_ids
        step_batch_size = tail_state.state_batch_size
        step_positions = tail_state.live_positions
        step_hidden_states = tail_state.live_hidden_states
        token_indices_to_sample = tail_state.live_token_indices_to_sample[:tail_state.sample_batch_size]
        for draft_step in range(self.num_speculative_tokens - 1):
            if self._state_debug and self._state_debug_interval > 0 and self._state_debug_counter % self._state_debug_interval == 0:
                logger.info(
                    "[MTP_FUSED_DEBUG] graph_step_meta_probe step=%d has_meta=%s meta_len=%d",
                    draft_step + 1,
                    str(draft_attn_metadatas is not None),
                    len(draft_attn_metadatas) if draft_attn_metadatas is not None else -1,
                )
            step_input_ids = draft_token_ids_tensor[draft_step]
            step_positions = step_positions + 1

            exceeds_max_model_len = (
                step_positions >= self.vllm_config.model_config.max_model_len)
            clamped_positions = torch.where(exceeds_max_model_len, 0,
                                            step_positions)
            if do_state_debug:
                try:
                    step_input_dbg = step_input_ids[: min(4, step_input_ids.shape[0])].detach().to("cpu").tolist()
                    step_pos_dbg = clamped_positions[: min(4, clamped_positions.shape[0])].detach().to("cpu").tolist()
                    step_idx_dbg = token_indices_to_sample[: min(4, token_indices_to_sample.shape[0])].detach().to("cpu").tolist()
                    logger.info(
                        "[MTP_FUSED_DEBUG] graph_step%d_input input_head=%s pos_head=%s sample_idx_head=%s",
                        draft_step + 1,
                        step_input_dbg,
                        step_pos_dbg,
                        step_idx_dbg,
                    )
                except Exception as _e:
                    logger.warning("[MTP_FUSED_DEBUG] graph_step%d_input failed: %s", draft_step + 1, repr(_e))

            work_input_ids = work_input_ids.clone()
            work_input_ids[:step_batch_size] = step_input_ids
            self._set_positions(step_batch_size, clamped_positions)
            self.hidden_states[:step_batch_size] = step_hidden_states

            model_input_ids = work_input_ids
            model_positions = self._get_positions(num_tokens)
            model_hidden_states = self.hidden_states[:num_tokens]
            model_hidden_states, model_positions = self.maybe_pad_and_reduce(
                model_hidden_states, model_positions)

            if draft_attn_metadatas and draft_step + 1 < len(
                    draft_attn_metadatas):
                forward_context.attn_metadata = draft_attn_metadatas[
                    draft_step + 1]
                self._last_graph_meta_status = {
                    "has_meta": True,
                    "meta_len": len(draft_attn_metadatas),
                    "total_steps": self.num_speculative_tokens,
                    "active_meta_id": id(getattr(forward_context, "attn_metadata", None)),
                    "selected_meta_id": id(draft_attn_metadatas[draft_step + 1]),
                    "selected_meta_type": type(getattr(forward_context, "attn_metadata", None)).__name__,
                }
                if self._state_debug:
                    logger.info(
                        "[MTP_FUSED_DEBUG] graph_step_meta_switch step=%d total_steps=%d meta_len=%d active_meta_id=%d selected_meta_id=%d active_meta_type=%s",
                        draft_step + 1,
                        self.num_speculative_tokens,
                        len(draft_attn_metadatas),
                        id(getattr(forward_context, "attn_metadata", None)),
                        id(draft_attn_metadatas[draft_step + 1]),
                        type(getattr(forward_context, "attn_metadata", None)).__name__,
                    )
            elif self._state_debug and self._state_debug_interval > 0 and self._state_debug_counter % self._state_debug_interval == 0:
                logger.warning(
                    "[MTP_FUSED_DEBUG] graph_step_meta_unavailable step=%d reason=%s meta_len=%d",
                    draft_step + 1,
                    "none_or_short",
                    len(draft_attn_metadatas) if draft_attn_metadatas is not None else -1,
                )

            model_kwargs = {
                "input_ids": model_input_ids,
                "positions": model_positions,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = model_hidden_states

            ret_hidden_states = raw_model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states_out = last_hidden_states
            else:
                last_hidden_states, hidden_states_out = ret_hidden_states

            last_hidden_states, model_positions, hidden_states_out = (
                self.maybe_all_gather_and_unpad(
                    last_hidden_states, model_positions, hidden_states_out))

            sample_hs = last_hidden_states[token_indices_to_sample]
            logits = raw_model.compute_logits(sample_hs)
            draft_token_ids = logits.argmax(dim=-1)
            if do_state_debug:
                try:
                    vocab_n = min(4, logits.shape[-1]) if logits.ndim > 1 else 0
                    draft_head_dbg = draft_token_ids[: min(4, draft_token_ids.shape[0])].detach().to("cpu").tolist()
                    step_pos_dbg = clamped_positions[: min(4, clamped_positions.shape[0])].detach().to("cpu").tolist()
                    hs_l2 = float(sample_hs.float().pow(2).mean().sqrt().item()) if sample_hs.numel() > 0 else 0.0
                    logits_topk_dbg = []
                    logits_topv_dbg = []
                    if logits.ndim == 2 and logits.shape[0] > 0 and vocab_n > 0:
                        topv, topi = torch.topk(logits[:1], k=vocab_n, dim=-1)
                        logits_topk_dbg = topi[0].detach().to("cpu").tolist()
                        logits_topv_dbg = [float(x) for x in topv[0].detach().to("cpu").tolist()]
                    logger.info(
                        "[MTP_FUSED_DEBUG] graph_step%d_logits_probe pos_head=%s draft_head=%s hs_l2=%.6f top_idx=%s top_val=%s",
                        draft_step + 1,
                        step_pos_dbg,
                        draft_head_dbg,
                        hs_l2,
                        logits_topk_dbg,
                        logits_topv_dbg,
                    )
                except Exception as _e:
                    logger.warning("[MTP_FUSED_DEBUG] graph_step%d_logits_probe failed: %s", draft_step + 1, repr(_e))
            if capture_step_snapshots is not None:
                capture_step_snapshots.append({
                    "step": draft_step + 1,
                    "input_head": step_input_ids[: min(4, step_input_ids.shape[0])].clone(),
                    "pos_head": clamped_positions[: min(4, clamped_positions.shape[0])].clone(),
                    "sample_idx_head": token_indices_to_sample[: min(4, token_indices_to_sample.shape[0])].clone(),
                    "draft_head": draft_token_ids[: min(4, draft_token_ids.shape[0])].clone(),
                    "meta_len": len(draft_attn_metadatas) if draft_attn_metadatas is not None else -1,
                    "selected_meta_id": id(draft_attn_metadatas[draft_step + 1]) if draft_attn_metadatas and draft_step + 1 < len(draft_attn_metadatas) else -1,
                })
            draft_token_ids_tensor[draft_step + 1] = draft_token_ids
            step_hidden_states = hidden_states_out[:step_batch_size]

        return draft_token_ids_tensor, capture_step_snapshots

    def propose_tail_live_graph(
        self,
        tail_state: TailLiveState,
        num_tokens: int,
        num_input_tokens: int,
        multi_steps_attn_metadata,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dedicated entry point for the future tail live-lane graph.

        This is intentionally not wired into the runtime path yet. The goal is
        to provide a stable call boundary so later refactors can swap the
        current companion-lane later-step rollout for a compact live-lane graph
        without reshaping callers again.
        """
        return self._run_merged_tail_steps(
            tail_state=tail_state,
            num_tokens=num_tokens,
            num_input_tokens=num_input_tokens,
            inputs_embeds=inputs_embeds,
            multi_steps_attn_metadata=multi_steps_attn_metadata,
            shadow_step_snapshots=None,
        )

    def shallow_copy_metadata(self, attn_metadata):
        # Currently, new objects will be assigned to the lists in attn_metadata
        # when update. So we can use the shallow copy.
        return copy.copy(attn_metadata)

    def _freeze_draft_step_attn_metadata(self, attn_metadata):
        decode_metadata = getattr(attn_metadata, "decode", None)
        if decode_metadata is not None:
            if decode_metadata.sas_metadata is not None:
                decode_metadata.sas_metadata = decode_metadata.sas_metadata.clone()
        return attn_metadata

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        in_graph_capturing: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ):
        (
            num_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_tokens, is_draft_model=True)

        multi_steps_attn_metadata = []
        requested_aclgraph_runtime_mode = aclgraph_runtime_mode
        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
        # Build per-step draft metadata from the requested fused runtime mode
        # even when the drafter itself does not dispatch through aclgraph.
        if requested_aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.draft_attn_groups) > 0:
            num_computed_tokens_cpu = self.runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]

            # num_reqs is already the padded version
            self.query_start_loc.cpu[: num_reqs + 1].copy_(self.runner.query_start_loc.cpu[: num_reqs + 1])
            self.query_start_loc.copy_to_gpu()

            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],
                seq_lens_cpu=self.runner.seq_lens.cpu,
                seq_lens=self.runner.seq_lens.gpu[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                num_input_tokens=num_tokens,
                max_query_len=self.num_speculative_tokens + 1,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
                block_table_tensor=self.runner.input_batch.block_table[0].get_device_tensor()[:num_reqs],
                # This is used to hold a position.
                slot_mapping=self.runner.input_batch.block_table[0].slot_mapping.gpu,
                positions=self.runner.positions.gpu,
                positions_cpu=self.runner.positions.cpu,
                attn_state=self.runner.attn_state,
                decode_token_per_req=self.runner.decode_token_per_req,
                max_seq_len=0,
            )
            if self.pcp_size * self.dcp_size > 1:
                # update long_seq related params and flatten block_table
                common_attn_metadata.prefill_context_parallel_metadata = self.runner.pcp_manager.long_seq_metadata
                common_attn_metadata.block_table_tensor = self.runner.input_batch.block_table[0].get_device_tensor()[
                    : num_reqs * self.decode_threshold
                ]

            assert len(self.draft_attn_groups) > 0
            builder = self.draft_attn_groups[0].get_metadata_builder()
            build_for_graph_capture_params = inspect.signature(builder.build_for_graph_capture).parameters
            supports_builder_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in build_for_graph_capture_params.values()
            )
            extra_attn_metadata_args = {}
            if supports_builder_kwargs:
                extra_attn_metadata_args = dict(
                    prefill_ratio_to_sas_metadata=dict(),
                    decode_ratio_to_sas_metadata=dict(),
                    common_ratio_to_sas_metadata=dict(),
                    block_size=self.draft_attn_groups[0].kv_cache_spec.block_size,
                )
            # update the tensor's address for each step.
            for draft_step in range(self.num_speculative_tokens):
                common_attn_metadata = self.shallow_copy_metadata(common_attn_metadata)
                # Set the real slot_mapping.
                common_attn_metadata.slot_mapping = self.slot_mapping_group[draft_step]
                attn_metadata_eagle = builder.build_for_graph_capture(
                    common_attn_metadata,
                    AscendAttentionState.SpecDecoding if self.method == "mtp" else AscendAttentionState.ChunkedPrefill,
                    **extra_attn_metadata_args,
                )
                per_layer_attn_metadata = dict()
                for layer_name in self.attn_layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata_eagle
                multi_steps_attn_metadata.append(per_layer_attn_metadata)

        self._last_dummy_attn_metadata = multi_steps_attn_metadata
        if _mtp_meta_chain_enabled():
            logger.info(
                "[MTP_META_CHAIN] dummy_run_saved total_steps=%d %s",
                self.num_speculative_tokens,
                _mtp_meta_chain_summary(self._last_dummy_attn_metadata),
            )

        model_positions = self._get_positions(num_tokens)

        batch_size = max(
            num_tokens // (self.num_speculative_tokens + 1), 1
        )  # if not is_profile else self.runner.max_num_reqs
        if is_profile:
            batch_size = min(batch_size, self.runner.max_num_reqs)

        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = (None, None)
            inputs_embeds = self.model.embed_input_ids(
                self.input_ids[:num_tokens], multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed
            )
            self.inputs_embeds[:num_tokens] = inputs_embeds
            inputs_embeds = self.inputs_embeds[:num_tokens]
        else:
            inputs_embeds = None

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=0,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
        ):
            # Reset MOE layer index before first model call
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0

            self._runnable(
                num_input_tokens=num_tokens,
                batch_size=batch_size,
                token_indices_to_sample=self.token_indices_to_sample[: batch_size * self.extra_slots_per_request],
                # The target_position's address is same as the model_positions's
                target_positions=model_positions,
                inputs_embeds=inputs_embeds,
                multi_steps_attn_metadata=multi_steps_attn_metadata,
                num_tokens=num_tokens,
            )
            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                self._update_full_graph_params(forward_context, num_tokens, multi_steps_attn_metadata)

    def _propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        target_model_batch_desc: BatchDescriptor,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
        scheduler_output: SchedulerOutput = None,
        num_scheduled_tokens: int = 0,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = common_attn_metadata.batch_size()

        if token_indices_to_sample is None:
            token_indices_to_sample = common_attn_metadata.query_start_loc[1:] - 1

        if self.method == "eagle3":
            assert isinstance(self.get_model(), Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        num_tokens, token_indices_to_sample, common_attn_metadata, long_seq_args = self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
        )
        if self.pcp_size * self.dcp_size > 1:
            assert long_seq_args is not None
            query_lens_d, ori_token_indices_to_sample = long_seq_args
        assert self.runner is not None
        if self.use_cuda_graph and num_tokens <= self.runner.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.runner.cudagraph_dispatcher._bs_to_padded_graph_size[num_tokens]
        else:
            num_input_tokens = num_tokens

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_input_tokens, is_draft_model=True)

        has_lora = len(self.runner.input_batch.lora_id_to_lora_request) > 0
        if self.use_cuda_graph:
            aclgraph_runtime_mode, batch_descriptor = self.runner.cudagraph_dispatcher.dispatch(
                num_tokens=num_input_tokens, uniform_decode=target_model_batch_desc.uniform, has_lora=has_lora
            )
        else:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
            batch_descriptor = None

        if aclgraph_runtime_mode == CUDAGraphMode.FULL:
            # TODO: Due to the inconsistency between the proposer `dispatcher` and model runner, this padding
            # should have been done in model runner but not. For example, at prefill stage, target model
            # is run in eager mode currently, which means `_pad_query_start_loc_for_fia` is not called,
            # while draft model is run in graph model, which means we should pad the `query_start_loc`.
            # Need to be fixed in the future.
            num_reqs_padded = self.runner._pad_query_start_loc_for_fia(
                num_input_tokens, common_attn_metadata.num_reqs, common_attn_metadata.num_reqs
            )
            common_attn_metadata.num_reqs = num_reqs_padded
            common_attn_metadata.query_start_loc = self.runner.query_start_loc.gpu[: num_reqs_padded + 1]
            common_attn_metadata.query_start_loc_cpu = self.runner.query_start_loc.cpu[: num_reqs_padded + 1]
            common_attn_metadata.block_table_tensor = self._pad_tensor(
                common_attn_metadata.block_table_tensor, num_reqs_padded
            )
            common_attn_metadata.seq_lens = self.runner.seq_lens.gpu[:num_reqs_padded]
            common_attn_metadata.seq_lens_cpu = self.runner.seq_lens.cpu[:num_reqs_padded]

        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)
            inputs_embeds = self.model.embed_input_ids(
                self.input_ids[:num_tokens], multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed
            )
            self.inputs_embeds[:num_tokens] = inputs_embeds
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            inputs_embeds = None

        active_tail_runnable = self._tail_live_runnable if getattr(
            self, "_tail_live_graph_enabled", False
        ) else self._runnable
        compact_live_lanes = active_tail_runnable is self._tail_live_runnable
        # Current compact tail-live path is not shape-stable with mRoPE models.
        # Guard it off to avoid runtime rotary kernel crashes.
        if compact_live_lanes and self.uses_mrope:
            if not getattr(self, "_tail_live_graph_mrope_disabled_logged", False):
                logger.warning(
                    "[MTP_FUSED_DEBUG] disable tail_live_graph for mRoPE model; fallback to stable tail runnable"
                )
                self._tail_live_graph_mrope_disabled_logged = True
            active_tail_runnable = self._runnable
            compact_live_lanes = False
        token_indices_to_sample_len = token_indices_to_sample.shape[0]
        tail_batch_size = token_indices_to_sample_len if compact_live_lanes else batch_size
        # For compact tail-live lanes, later-step metadata/input widths must
        # follow the live lane count instead of graph-padded input width.
        tail_num_input_tokens = tail_batch_size if compact_live_lanes else num_input_tokens

        # Update slot_mapping for different speculative.
        # NOTE: Currently, we only remake the slot_mapping, because it's the
        # only tensor which will be used in current FIA.
        # Strictly speaking, `query_start_loc`, `seq_lens` should also have
        # their memory allocated separately for each step just like `slot_mapping`.
        slot_mapping_lens = common_attn_metadata.slot_mapping.shape[0]
        self.slot_mapping_group[0][:slot_mapping_lens].copy_(common_attn_metadata.slot_mapping[:slot_mapping_lens])
        self.slot_mapping_group[0][slot_mapping_lens:].fill_(-1)
        common_attn_metadata.slot_mapping = self.slot_mapping_group[0]
        common_attn_metadata.num_input_tokens = tail_num_input_tokens
        # FIXME(woosuk): The below two ops cause synchronization. Optimize.
        assert len(self.draft_attn_groups) > 0
        builder = self.draft_attn_groups[0].get_metadata_builder()
        extra_attn_metadata_args = dict(
                    prefill_ratio_to_sas_metadata=dict(),
                    decode_ratio_to_sas_metadata=dict(),
                    common_ratio_to_sas_metadata=dict(),
                    block_size=self.draft_attn_groups[0].kv_cache_spec.block_size)
        attn_metadata = builder.build(0, common_attn_metadata, self.runner.get_model(), **extra_attn_metadata_args)
        attn_metadata = self._freeze_draft_step_attn_metadata(attn_metadata)

        multi_steps_attn_metadata, attn_metadata_i = self._build_multi_steps_attn_metadata(
            common_attn_metadata=common_attn_metadata,
            token_indices_to_sample=token_indices_to_sample,
            attn_metadata=attn_metadata,
            batch_size=tail_batch_size,
            num_input_tokens=tail_num_input_tokens,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            compact_live_lanes=compact_live_lanes,
            query_lens_d=query_lens_d if self.pcp_size * self.dcp_size > 1 else None,
            ori_token_indices_to_sample=ori_token_indices_to_sample if self.pcp_size * self.dcp_size > 1 else None,
            num_decode_reqs=num_decode_reqs,
        )

        self.token_indices_to_sample[:token_indices_to_sample_len].copy_(token_indices_to_sample)

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0],
            self.vllm_config,
            num_tokens=tail_num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_tokens,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
        ):
            # Reset MOE layer index for forward pass
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0

            draft_token_ids = active_tail_runnable(
                num_input_tokens=tail_num_input_tokens,
                batch_size=tail_batch_size,
                token_indices_to_sample=self.token_indices_to_sample[:token_indices_to_sample_len],
                target_positions=target_positions,
                inputs_embeds=inputs_embeds,
                multi_steps_attn_metadata=multi_steps_attn_metadata,
                num_tokens=num_tokens,
                is_prefill=attn_metadata_i.num_prefills,
            )

            forward_context = get_forward_context()
            if (
                forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and active_tail_runnable is self._runnable
            ):
                self._update_full_graph_params(forward_context, num_input_tokens, multi_steps_attn_metadata)
        return draft_token_ids

    def _build_multi_steps_attn_metadata(
        self,
        common_attn_metadata,
        token_indices_to_sample: torch.Tensor,
        attn_metadata,
        batch_size: int,
        num_input_tokens: int,
        aclgraph_runtime_mode: CUDAGraphMode,
        compact_live_lanes: bool,
        query_lens_d=None,
        ori_token_indices_to_sample=None,
        num_decode_reqs: int = 0,
    ) -> tuple[list[dict[str, Any]], Any]:
        if self.uses_mrope:
            used_update_positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            used_update_positions = self.positions[token_indices_to_sample]
        per_layer_attn_metadata = dict()
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        multi_steps_attn_metadata = [per_layer_attn_metadata]

        attn_metadata_i = per_layer_attn_metadata[self.attn_layer_names[0]]
        common_attn_metadata.block_table_tensor = common_attn_metadata.block_table_tensor.clone()

        if self.pcp_size * self.dcp_size > 1:
            if self.num_speculative_tokens > 1 and not attn_metadata_i.num_prefills:
                num_reject_tokens = (
                    torch.tensor(self.runner.pcp_manager.cu_num_tokens_pcp_full, dtype=torch.int32).to(self.device)
                    - ori_token_indices_to_sample
                    - 1
                )
                num_accept_tokens = query_lens_d.to(self.device) - num_reject_tokens
                ori_seq_len = attn_metadata_i.seq_lens_cpu[:batch_size].clone()
                mtp_slot_mapping = self.runner.pcp_manager.mtp_slot_pad

                slot_idx_base = (
                    torch.cat(
                        [
                            torch.tensor([0], dtype=torch.int32, device=self.device),
                            (torch.cumsum(query_lens_d, dim=0)[:-1] * self.pcp_size).to(self.device),
                        ]
                    )
                    + torch.arange(num_decode_reqs, device=self.device)
                    * (self.num_speculative_tokens - 1)
                    * self.pcp_size
                    + (num_accept_tokens - 1) * self.pcp_size
                )
                slot_indices_list = []
                for req_id in range(num_decode_reqs):
                    slot_indices_list.append(
                        torch.arange(slot_idx_base[req_id], slot_idx_base[req_id] + self.pcp_size, device=self.device)
                    )
                slot_indices = torch.cat(slot_indices_list, dim=0)

                block_indices = torch.cat(
                    [torch.tensor([0], dtype=torch.int32), torch.cumsum(query_lens_d, dim=0)[:-1]]
                )
                common_attn_metadata.block_table_tensor[:batch_size] = common_attn_metadata.block_table_tensor[
                    block_indices
                ]
                common_attn_metadata.block_table_tensor = common_attn_metadata.block_table_tensor[:batch_size]

                if not self.parallel_drafting:
                    for draft_step in range(1, self.num_speculative_tokens):
                        per_layer_attn_metadata = dict()
                        for attn_group in self.draft_attn_groups:
                            common_attn_metadata, attn_metadata = self.attn_update_stack_num_spec_norm(
                                draft_step,
                                attn_metadata,
                                common_attn_metadata,
                                batch_size,
                                num_input_tokens,
                                used_update_positions,
                                aclgraph_runtime_mode,
                                ori_seq_len,
                                slot_indices,
                                mtp_slot_mapping,
                                attn_group=attn_group,
                                compact_live_lanes=compact_live_lanes,
                            )
                            attn_metadata = self._freeze_draft_step_attn_metadata(attn_metadata)
                            for layer_name in self.attn_layer_names:
                                per_layer_attn_metadata[layer_name] = attn_metadata
                        multi_steps_attn_metadata.append(per_layer_attn_metadata)
        else:
            if not self.parallel_drafting:
                for draft_step in range(1, self.num_speculative_tokens):
                    per_layer_attn_metadata = dict()
                    for attn_group in self.draft_attn_groups:
                        common_attn_metadata, attn_metadata = self.attn_update_stack_num_spec_norm(
                            draft_step,
                            attn_metadata,
                            common_attn_metadata,
                            batch_size,
                            num_input_tokens,
                            used_update_positions,
                            aclgraph_runtime_mode,
                            attn_group=attn_group,
                            compact_live_lanes=compact_live_lanes,
                        )
                        attn_metadata = self._freeze_draft_step_attn_metadata(attn_metadata)
                        for layer_name in self.attn_layer_names:
                            per_layer_attn_metadata[layer_name] = attn_metadata
                    multi_steps_attn_metadata.append(per_layer_attn_metadata)

        return multi_steps_attn_metadata, attn_metadata_i

    def _run_merged_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
    ) -> torch.Tensor:
        draft_token_ids, hidden_states, token_indices_to_sample, shadow_step_snapshots = (
            self._run_merged_step0(
                num_input_tokens=num_input_tokens,
                token_indices_to_sample=token_indices_to_sample,
                inputs_embeds=inputs_embeds,
                num_tokens=num_tokens,
            ))

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        if self.pcp_size * self.dcp_size > 1 and is_prefill:
            draft_token_ids_list = []
            for _ in range(self.num_speculative_tokens):
                draft_token_ids_list.append(draft_token_ids)
            return torch.stack(draft_token_ids_list, dim=1)

        tail_state = self._build_tail_live_state(
            step0_draft_token_ids=draft_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            hidden_states=hidden_states,
            batch_size=batch_size,
        )
        draft_token_ids = self._run_merged_tail_steps(
            tail_state=tail_state,
            num_tokens=num_tokens,
            num_input_tokens=num_input_tokens,
            inputs_embeds=inputs_embeds,
            multi_steps_attn_metadata=multi_steps_attn_metadata,
            shadow_step_snapshots=shadow_step_snapshots,
        )
        if shadow_step_snapshots is not None:
            self._shadow_multistep_snapshot = shadow_step_snapshots
        return draft_token_ids

    def _run_compact_tail_live_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
    ) -> torch.Tensor:
        draft_token_ids, hidden_states, token_indices_to_sample, shadow_step_snapshots = (
            self._run_merged_step0(
                num_input_tokens=num_input_tokens,
                token_indices_to_sample=token_indices_to_sample,
                inputs_embeds=inputs_embeds,
                num_tokens=num_tokens,
            ))

        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        if self.pcp_size * self.dcp_size > 1 and is_prefill:
            draft_token_ids_list = []
            for _ in range(self.num_speculative_tokens):
                draft_token_ids_list.append(draft_token_ids)
            return torch.stack(draft_token_ids_list, dim=1)

        tail_state = self._build_tail_live_state(
            step0_draft_token_ids=draft_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            hidden_states=hidden_states,
            batch_size=batch_size,
        )
        draft_token_ids = self.propose_tail_live_graph(
            tail_state=tail_state,
            num_tokens=num_tokens,
            num_input_tokens=num_input_tokens,
            multi_steps_attn_metadata=multi_steps_attn_metadata,
            inputs_embeds=inputs_embeds,
        )
        if shadow_step_snapshots is not None:
            self._shadow_multistep_snapshot = shadow_step_snapshots
        return draft_token_ids

    def _run_merged_step0(
        self,
        num_input_tokens: int,
        token_indices_to_sample: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]] | None]:
        model_input_ids = self.input_ids[:num_input_tokens]
        model_positions = self._get_positions(num_input_tokens)

        model_kwargs = {
            "input_ids": model_input_ids,
            "positions": model_positions,
            "inputs_embeds": inputs_embeds,
        }

        if self.pass_hidden_states_to_model:
            model_hidden_states = self.hidden_states[:num_input_tokens]
            model_hidden_states, model_positions = self.maybe_pad_and_reduce(model_hidden_states, model_positions)
            model_kwargs["hidden_states"] = model_hidden_states
            if self.method == "mtp":
                model_kwargs["positions"] = model_positions

        ret_hidden_states = self.model(**model_kwargs)
        if not self.model_returns_tuple():
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states

        last_hidden_states, model_positions, hidden_states = self.maybe_all_gather_and_unpad(
            last_hidden_states, model_positions, hidden_states
        )

        num_indices = token_indices_to_sample.shape[0]
        if self.pcp_size > 1:
            hidden_states = hidden_states[:num_tokens]
            hidden_states = get_pcp_group().all_gather(hidden_states, 0)
            hidden_states = torch.index_select(
                hidden_states, 0, self.runner.pcp_manager.pcp_allgather_restore_idx.gpu[: hidden_states.shape[0]]
            )
            if self.method == "mtp":
                last_hidden_states = hidden_states
            else:
                last_hidden_states = last_hidden_states[:num_tokens]
                last_hidden_states = get_pcp_group().all_gather(last_hidden_states, 0)
                last_hidden_states = torch.index_select(
                    last_hidden_states,
                    0,
                    self.runner.pcp_manager.pcp_allgather_restore_idx.gpu[: last_hidden_states.shape[0]],
                )

        if lmhead_tp_enable():
            max_num_reqs_across_dp = (
                self.vllm_config.scheduler_config.max_num_seqs * self.runner.uniform_decode_query_len
            )
            token_indices_to_sample = nn.functional.pad(
                token_indices_to_sample, (0, max_num_reqs_across_dp - num_indices)
            )

        shadow_step_snapshots = [] if _mtp_meta_chain_enabled() else None

        sample_hidden_states = last_hidden_states[token_indices_to_sample]
        logits = self.model.compute_logits(sample_hidden_states)

        if lmhead_tp_enable() and num_indices < logits.shape[0]:
            logits = logits[:num_indices]
            token_indices_to_sample = token_indices_to_sample[:num_indices]

        draft_token_ids = logits.argmax(dim=-1)
        if shadow_step_snapshots is not None:
            shadow_step_snapshots.append({
                "step": 0,
                "input_head": model_input_ids[: min(4, model_input_ids.shape[0])].clone(),
                "pos_head": model_positions[: min(4, model_positions.shape[0])].clone(),
                "sample_idx_head": token_indices_to_sample[: min(4, token_indices_to_sample.shape[0])].clone(),
                "draft_head": draft_token_ids[: min(4, draft_token_ids.shape[0])].clone(),
            })
        return draft_token_ids, hidden_states, token_indices_to_sample, shadow_step_snapshots

    def _build_tail_live_state(
        self,
        step0_draft_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        hidden_states: torch.Tensor,
        batch_size: int,
    ) -> TailLiveState:
        # Compact tail state dimensions must follow live sample lanes, not the
        # caller-provided batch size, otherwise positions/hidden_states dim0
        # can diverge and break rotary kernels.
        live_batch_size = int(token_indices_to_sample.shape[0])
        if self.uses_mrope:
            live_positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            live_positions = self.positions[token_indices_to_sample]
        live_hidden_states = hidden_states[token_indices_to_sample]
        step0_draft_token_ids = step0_draft_token_ids[:live_batch_size]
        return self._make_tail_live_state(
            step0_draft_token_ids=step0_draft_token_ids,
            live_token_indices_to_sample=token_indices_to_sample,
            live_positions=live_positions,
            live_hidden_states=live_hidden_states,
            state_batch_size=live_batch_size,
            sample_batch_size=live_batch_size,
        )

    def _run_merged_tail_steps(
        self,
        tail_state: TailLiveState,
        num_tokens: int,
        num_input_tokens: int,
        inputs_embeds: torch.Tensor | None,
        multi_steps_attn_metadata,
        shadow_step_snapshots: list[dict[str, Any]] | None,
    ) -> torch.Tensor:
        draft_token_ids_tensor = torch.zeros(
            (self.num_speculative_tokens, *tail_state.step0_draft_token_ids.shape),
            dtype=tail_state.step0_draft_token_ids.dtype,
            device=self.device,
        )
        draft_token_ids_tensor[0] = tail_state.step0_draft_token_ids
        positions = tail_state.live_positions
        hidden_states = tail_state.live_hidden_states
        batch_size = tail_state.state_batch_size
        token_indices_to_sample = self.arange[:tail_state.sample_batch_size]

        # For compact tail live-lane execution, the later-step model input must
        # follow live state width; reusing num_input_tokens can desync shapes
        # between positions/metadata/hidden_states on rotary kernels.
        use_compact_tail_live = (
            getattr(self, "_tail_live_graph_enabled", False)
            and batch_size < num_input_tokens
        )
        input_batch_size = (
            batch_size
            if use_compact_tail_live
            else (num_input_tokens if (self.method == "mtp" or self.use_cuda_graph) else batch_size)
        )

        forward_context = get_forward_context()
        _EXTRA_CTX.num_tokens = input_batch_size
        _EXTRA_CTX.num_accept_tokens = batch_size

        for draft_step in range(self.num_speculative_tokens - 1):
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0

            input_ids = draft_token_ids_tensor[draft_step]
            positions += 1

            if self.uses_mrope:
                exceeds_max_model_len = positions[0] >= self.vllm_config.model_config.max_model_len
                clamped_positions = torch.where(
                    exceeds_max_model_len.unsqueeze(0), torch.zeros_like(positions), positions
                )
            else:
                exceeds_max_model_len = positions >= self.vllm_config.model_config.max_model_len
                clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            model_input_ids = self.input_ids[:input_batch_size]
            model_positions = self._get_positions(input_batch_size)
            model_hidden_states = self.hidden_states[:input_batch_size]
            model_hidden_states, model_positions = self.maybe_pad_and_reduce(
                model_hidden_states, model_positions)

            forward_context.attn_metadata = (
                multi_steps_attn_metadata[draft_step + 1] if multi_steps_attn_metadata else None
            )

            model_kwargs = {
                "input_ids": model_input_ids,
                "positions": model_positions,
                "inputs_embeds": inputs_embeds,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = model_hidden_states

            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

            last_hidden_states, model_positions, hidden_states = self.maybe_all_gather_and_unpad(
                last_hidden_states, model_positions, hidden_states)

            num_indices = token_indices_to_sample.shape[0]
            if lmhead_tp_enable():
                max_num_reqs_across_dp = (
                    self.vllm_config.scheduler_config.max_num_seqs * self.runner.uniform_decode_query_len
                )
                token_indices_to_sample = nn.functional.pad(
                    token_indices_to_sample,
                    (0, max_num_reqs_across_dp - num_indices),
                )

            sample_hidden_states = last_hidden_states[token_indices_to_sample]
            logits = self.model.compute_logits(sample_hidden_states)

            if lmhead_tp_enable() and num_indices < logits.shape[0]:
                logits = logits[:num_indices]
                token_indices_to_sample = token_indices_to_sample[:num_indices]

            hidden_states = hidden_states[:batch_size]
            draft_token_ids = logits.argmax(dim=-1)
            if shadow_step_snapshots is not None:
                shadow_step_snapshots.append({
                    "step": draft_step + 1,
                    "input_head": input_ids[: min(4, input_ids.shape[0])].clone(),
                    "pos_head": clamped_positions[: min(4, clamped_positions.shape[0])].clone(),
                    "sample_idx_head": token_indices_to_sample[: min(4, token_indices_to_sample.shape[0])].clone(),
                    "draft_head": draft_token_ids[: min(4, draft_token_ids.shape[0])].clone(),
                })
            draft_token_ids_tensor[draft_step + 1] = draft_token_ids

        return draft_token_ids_tensor.swapaxes(0, 1)

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        if not self.needs_extra_input_slots:
            # Default EAGLE pathway: no reshaping of input tensors needed.
            # Simply rotate the input ids and leave the positions unchanged,
            # Inserting the next token ids at the last slot in each request.
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]
            # Shift the input ids by one token.
            # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]
            # Replace the last token with the next token.
            # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
            # Keep index/value dtype aligned with input_ids on NPU index_put.
            if token_indices_to_sample.dtype != self.input_ids.dtype:
                token_indices_to_sample = token_indices_to_sample.to(self.input_ids.dtype)
            if next_token_ids.dtype != self.input_ids.dtype:
                next_token_ids = next_token_ids.to(self.input_ids.dtype)
            self.input_ids[token_indices_to_sample] = next_token_ids

            assert self.runner is not None
            # update pcp related params
            ori_token_indices_to_sample = None
            query_lens_d = None
            if self.pcp_size * self.dcp_size > 1:
                assert long_seq_metadata is not None
                cad.prefill_context_parallel_metadata = long_seq_metadata
                ori_token_indices_to_sample = token_indices_to_sample.clone()
                query_lens_d = self.runner.query_lens[:num_decode_reqs]
            if self.pcp_size > 1:
                # 1. preprocess decode/prefill input_ids & target_hidden_states
                # decode input_ids: keep unchanged
                # decode target_hidden_states: remove padding
                # prefill input_ids: add padding and pcp split
                # prefill target_hidden_states: pcp split
                assert query_lens_d is not None
                num_tokens_d = query_lens_d.sum().item()
                num_tokens_d_padded = num_tokens_d * self.pcp_size
                input_ids_d = self.input_ids[:num_tokens_d]
                input_ids_p = self.input_ids[num_tokens_d:num_tokens]
                target_hidden_states_d_padded = target_hidden_states[:num_tokens_d_padded]
                if num_tokens_d:
                    # remove padding (from pcp all-gather) in decode part
                    mask_start_loc = torch.cat(
                        [torch.tensor([0], dtype=torch.int32), torch.cumsum(query_lens_d * self.pcp_size, dim=0)[:-1]]
                    )
                    mask_len = query_lens_d
                    mask = []
                    for req_id in range(num_decode_reqs):
                        assert None not in (mask_start_loc, mask_len)
                        mask += list(range(mask_start_loc[req_id], mask_start_loc[req_id] + mask_len[req_id]))
                    target_hidden_states_d = target_hidden_states_d_padded[mask]
                else:
                    target_hidden_states_d = target_hidden_states_d_padded
                target_hidden_states_p = target_hidden_states[num_tokens_d_padded:]
                req_scheduled_tokens_p = {}
                for i, req_id in enumerate(self.runner.input_batch.req_ids):
                    if i >= num_decode_reqs:
                        req_scheduled_tokens_p[req_id] = req_scheduled_tokens[req_id]
                (num_tokens_p, input_ids_p, target_hidden_states_p, max_query_len_p, seq_lens_p, cu_num_tokens_p) = (
                    self._split_pcp_input(req_scheduled_tokens_p, input_ids_p, target_hidden_states_p)
                )
                num_tokens = num_tokens_d + num_tokens_p
                target_positions = target_positions[:num_tokens]
                self.input_ids[:num_tokens].copy_(torch.cat([input_ids_d, input_ids_p], dim=0))
                target_hidden_states = torch.cat([target_hidden_states_d, target_hidden_states_p], dim=0)
                # 2. update sample_indices according to main model
                if num_decode_reqs:
                    token_indices_to_sample[:num_decode_reqs] = self.runner.logits_indices[
                        token_indices_to_sample[:num_decode_reqs]
                    ]
                if num_prefill_reqs:
                    token_indices_to_sample[-num_prefill_reqs:] = self.runner.logits_indices[-num_prefill_reqs:]
                    # 3. update attn_metadata params that may be influenced by pcp
                    cad.num_actual_tokens = num_tokens
                    cad.max_query_len = max(self.decode_threshold, max_query_len_p)
                    cad.seq_lens[-num_prefill_reqs:] = seq_lens_p
                    cad.seq_lens_cpu[-num_prefill_reqs:] = seq_lens_p
                    query_start_loc_p = cu_num_tokens_p[1:] + cad.query_start_loc[num_decode_reqs].item()
                    cad.query_start_loc[-num_prefill_reqs:] = query_start_loc_p
                    cad.query_start_loc_cpu[-num_prefill_reqs:] = query_start_loc_p

            # copy inputs to buffer for cudagraph
            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]

            self._set_positions(num_tokens, target_positions)
            self.hidden_states[:num_tokens] = target_hidden_states

            return num_tokens, token_indices_to_sample, cad, (query_lens_d, ori_token_indices_to_sample)
        else:
            assert self.is_rejected_token_mask is not None
            assert self.is_masked_token_mask is not None
            # 1.
            # Call the CopyAndExpandEagleInputs AscendC operator to copy
            # input_ids and positions into the correct slots in the
            # preallocated buffers self.input_ids, self.positions.
            batch_size = cad.batch_size()
            total_num_input_tokens = target_token_ids.shape[0]
            total_num_output_tokens = total_num_input_tokens + (self.net_num_new_slots_per_request * batch_size)

            query_start_loc = cad.query_start_loc
            query_end_loc = cad.query_start_loc[1:] - 1
            if num_rejected_tokens_gpu is not None:
                query_end_loc = query_end_loc - num_rejected_tokens_gpu

            (
                out_input_ids,
                out_positions,
                out_is_rejected_token_mask,
                out_is_masked_token_mask,
                token_indices_to_sample,
                out_hidden_state_mapping,
            ) = torch.ops._C_ascend.npu_copy_and_expand_eagle_inputs(
                target_token_ids,
                target_positions.to(torch.int32),
                next_token_ids,
                query_start_loc,
                query_end_loc,
                0,  # padding_token_id
                self.parallel_drafting_token_id,
                self.extra_slots_per_request,
                self.pass_hidden_states_to_model,
                total_num_output_tokens,
            )

            # Copy returned tensors into pre-allocated buffers
            self.input_ids[:total_num_output_tokens].copy_(out_input_ids)
            self.positions[:total_num_output_tokens].copy_(out_positions)
            self.is_rejected_token_mask[:total_num_output_tokens].copy_(out_is_rejected_token_mask)
            self.is_masked_token_mask[:total_num_output_tokens].copy_(out_is_masked_token_mask)
            if self.pass_hidden_states_to_model:
                assert self.parallel_drafting_hidden_state_tensor is not None
                self.hidden_states[out_hidden_state_mapping] = target_hidden_states
                # Use torch.where to avoid DtoH sync from boolean indexing
                mask = self.is_masked_token_mask[:total_num_output_tokens]
                torch.where(
                    mask.unsqueeze(1),  # type: ignore
                    self.parallel_drafting_hidden_state_tensor,
                    self.hidden_states[:total_num_output_tokens],
                    out=self.hidden_states[:total_num_output_tokens],
                )

            # 2.
            # Recompute the slot mapping based on the new positions and
            # rejection mask.
            # Use the first draft attention group's kv_cache_spec for block_size
            # (all draft layers share the same kv-cache group)
            assert len(self.draft_attn_groups) > 0
            block_size = self.draft_attn_groups[0].kv_cache_spec.block_size

            new_slot_mapping = compute_new_slot_mapping(
                cad=cad,
                new_positions=self.positions[:total_num_output_tokens],
                is_rejected_token_mask=self.is_rejected_token_mask[:total_num_output_tokens],
                block_size=block_size,
                num_new_tokens=self.net_num_new_slots_per_request,
                max_model_len=self.max_model_len,
            )

            # 3. Update the common attention metadata with the new (meta)data
            new_cad = extend_all_queries_by_N(
                cad,
                N=self.net_num_new_slots_per_request,
                arange=self.arange,
                new_slot_mapping=new_slot_mapping,
            )

            return total_num_output_tokens, token_indices_to_sample, new_cad, None

    def model_returns_tuple(self) -> bool:
        return self.method not in ("mtp", "draft_model")

    def attn_update_stack_num_spec_norm(
        self,
        # `draft_step` must start from `1`, no `0`
        draft_step,
        old_attn_metadata,
        old_common_metadata,
        batch_size,
        input_batch_size,
        used_update_positions,
        aclgraph_runtime_mode,
        ori_seq_len=None,
        slot_indices=None,
        mtp_slot_mapping=None,
        attn_group=None,
        compact_live_lanes: bool = False,
    ):
        assert draft_step > 0
        assert attn_group is not None, "vllm-ascend v0.17.0rc1 requires attn_group"
        common_attn_metadata = self.shallow_copy_metadata(old_common_metadata)

        if draft_step == 1:
            if aclgraph_runtime_mode == CUDAGraphMode.FULL and not compact_live_lanes:
                common_attn_metadata.num_reqs = input_batch_size
                common_attn_metadata.block_table_tensor = self._pad_tensor(
                    common_attn_metadata.block_table_tensor, input_batch_size
                )
                common_attn_metadata.seq_lens = self._pad_tensor(common_attn_metadata.seq_lens, input_batch_size)
                common_attn_metadata.seq_lens_cpu = self._pad_tensor(
                    common_attn_metadata.seq_lens_cpu, input_batch_size
                )
                common_attn_metadata.num_computed_tokens_cpu = self._pad_tensor(
                    common_attn_metadata.num_computed_tokens_cpu, input_batch_size
                )
                common_attn_metadata.query_start_loc = self.arange[: input_batch_size + 1]
                common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
                    self.token_arange_np[: input_batch_size + 1]
                ).clone()
            else:
                common_attn_metadata.num_reqs = batch_size
                if compact_live_lanes:
                    common_attn_metadata.block_table_tensor = common_attn_metadata.block_table_tensor[:batch_size]
                    common_attn_metadata.seq_lens = common_attn_metadata.seq_lens[:batch_size]
                    common_attn_metadata.seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:batch_size]
                    common_attn_metadata.num_computed_tokens_cpu = common_attn_metadata.num_computed_tokens_cpu[
                        :batch_size
                    ]
                common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
                common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
                    self.token_arange_np[: batch_size + 1]
                ).clone()

            common_attn_metadata.num_actual_tokens = batch_size
            common_attn_metadata.max_query_len = 1
            common_attn_metadata.decode_token_per_req = 1
            common_attn_metadata.attn_state = (
                AscendAttentionState.SpecDecoding if self.method == "mtp" else AscendAttentionState.ChunkedPrefill
            )
            common_attn_metadata.graph_pad_size = -1
            common_attn_metadata.num_input_tokens = batch_size if compact_live_lanes else input_batch_size

        # The loop part
        used_update_positions += 1

        # Clone the data so that when calculating the data at position 2 and position 3
        # in the merged graph, it does not affect position 1
        # FIXME(lilinsiman)
        common_attn_metadata.seq_lens = common_attn_metadata.seq_lens.clone()
        common_attn_metadata.seq_lens_cpu = common_attn_metadata.seq_lens_cpu.clone()
        common_attn_metadata.num_computed_tokens_cpu = common_attn_metadata.num_computed_tokens_cpu.clone()
        common_attn_metadata.positions = common_attn_metadata.positions.clone()

        # NOTE(woosuk): We should handle the case where the draft model
        # generates tokens beyond the max model length. Since it is complex
        # to remove such requests from the batch, we keep them in the batch
        # but adjust the position ids and slot mappings to avoid the
        # out-of-range access during the model execution. The draft tokens
        # generated with this adjustment should be ignored.
        if self.uses_mrope:
            exceeds_max_model_len = used_update_positions[0] >= self.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            clamped_positions = torch.where(
                exceeds_max_model_len.unsqueeze(0), torch.zeros_like(used_update_positions), used_update_positions
            )
        else:
            exceeds_max_model_len = used_update_positions >= self.max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0, used_update_positions)

        # For data integrity when async scheduling, we shouldn't use in place
        # operations in case they are modified in next step's `prepare_input`
        # of main model.
        # Increment the sequence lengths.
        common_attn_metadata.seq_lens[:batch_size] += 1
        # For the requests that exceed the max model length, we set the
        # sequence length to 1 to minimize their overheads in attention.
        common_attn_metadata.seq_lens[:batch_size].masked_fill_(exceeds_max_model_len, 1)

        common_attn_metadata.seq_lens_cpu[:batch_size] = common_attn_metadata.seq_lens_cpu[:batch_size] + 1
        exceeds_mask = common_attn_metadata.seq_lens_cpu[:batch_size] >= self.max_model_len
        common_attn_metadata.seq_lens_cpu[:batch_size].masked_fill_(exceeds_mask, 1)
        common_attn_metadata.num_computed_tokens_cpu[:batch_size] += 1
        if self.uses_mrope:
            common_attn_metadata.positions[:batch_size].copy_(clamped_positions[0])
        else:
            common_attn_metadata.positions[:batch_size].copy_(clamped_positions)

        if self.pcp_size * self.dcp_size > 1:
            num_computed_tokens_of_pcp_dcp = self.runner.pcp_manager._get_cp_local_seq_lens(
                ori_seq_len + draft_step + 1,
                self.pcp_size,
                self.dcp_size,
                self.runner.parallel_config.cp_kv_cache_interleave_size,
            )
            cp_seq_len = num_computed_tokens_of_pcp_dcp[:, self.pcp_rank, self.dcp_rank]
            # update slot_mapping
            slot_indices += self.pcp_size
            slot_mapping = mtp_slot_mapping[slot_indices]
            self.slot_mapping_group[draft_step][: batch_size * self.pcp_size] = slot_mapping
            common_attn_metadata.slot_mapping = self.slot_mapping_group[draft_step]
        else:
            # NOTE: In vllm, `block_size = attn_metadata_builder.kv_cache_spec.block_size`.
            # However, in vllm-ascend, the above value can be multiple of `kernel_block_size`,
            # which is not correct for computing `slot_mapping` below.
            block_size = self.kernel_block_size
            if not isinstance(block_size, int):
                block_size = 128

            # Compute the slot mapping.
            if self.uses_mrope:
                block_numbers = clamped_positions[0] // block_size
            else:
                block_numbers = clamped_positions // block_size
            block_ids = old_common_metadata.block_table_tensor.gather(dim=1, index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            if self.uses_mrope:
                slot_mapping = block_ids * block_size + clamped_positions[0] % block_size
            else:
                slot_mapping = block_ids * block_size + clamped_positions % block_size

            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)
            self.slot_mapping_group[draft_step][: slot_mapping.shape[0]].copy_(slot_mapping.to(torch.int32))
            self.slot_mapping_group[draft_step][slot_mapping.shape[0] :].fill_(PADDING_SLOT_ID)
            # Set the address of the attn_metadata.slot_mapping to the self.slot_mapping_group[idx]
            common_attn_metadata.slot_mapping = self.slot_mapping_group[draft_step]

        attn_metadata_builder = attn_group.get_metadata_builder()

        extra_attn_metadata_args = dict(
                    prefill_ratio_to_sas_metadata=dict(),
                    decode_ratio_to_sas_metadata=dict(),
                    common_ratio_to_sas_metadata=dict(),
                    block_size=self.draft_attn_groups[0].kv_cache_spec.block_size)

        attn_metadata = attn_metadata_builder.build(
            0,
            common_attn_metadata,
            self.runner.get_model(),
            **extra_attn_metadata_args,
        )

        if self.pcp_size * self.dcp_size > 1:
            if self.vllm_config.model_config.use_mla:
                if getattr(attn_metadata, "decode", None):
                    attn_metadata.decode.cp_seq_len = cp_seq_len
            else:
                attn_metadata.decode_meta.num_computed_tokens_of_pcp_dcp = num_computed_tokens_of_pcp_dcp

        return common_attn_metadata, attn_metadata

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_indices: torch.Tensor,
        num_discarded_requests: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead.
        It also accounts for the rejected tokens in `sampled_token_ids`.
        This function must use device functions to operate on the inputs, and
        should not introduce any blocking CPU-GPU synchronization.
        """
        # TODO(Ben): Combine this into a custom fused kernel

        # Precompute get_token_id for when there is no valid next token
        num_reqs = gpu_input_batch.num_reqs
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(common_attn_metadata.seq_lens_cpu[i].item())
                for i in range(num_reqs)
            ]
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)

        # Mask out the sampled tokens indices that should not be sampled.
        discard_sampled_tokens_req_indices = discard_request_indices[:num_discarded_requests]

        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        valid_sampled_token_ids_gpu.index_fill_(0, discard_sampled_tokens_req_indices, -1)

        # Generate a mask for all valid tokens within those requests
        valid_mask = (valid_sampled_token_ids_gpu != -1) & (valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size)

        # Count the number of valid tokens in each request
        valid_sampled_tokens_count = valid_mask.sum(dim=1)

        # Get the rightmost valid index per row
        last_valid_indices = valid_sampled_tokens_count - 1
        last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)

        # Get last valid token from each row
        # (assume undefined state where there is no valid token)
        selected_tokens = torch.gather(valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)).squeeze(1)

        # Use last token if valid, pre-computed backup if not
        batch_size = valid_sampled_token_ids_gpu.shape[0]
        next_token_ids = torch.where(
            last_valid_indices != -1,
            selected_tokens,
            self.backup_next_token_ids.gpu[:batch_size],
        )

        return next_token_ids, valid_sampled_tokens_count

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        num_actual_reqs = len(num_draft_tokens)
        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0 for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens, dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_actual_reqs + 1]
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_actual_reqs]
        new_seq_lens_cpu = seq_lens_cpu - num_rejected_tokens

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(new_query_start_loc_np[:-1], new_num_tokens_per_req_np)
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offsets = self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np)
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_np = token_offsets + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(device, non_blocking=True)

        common_attn_metadata.slot_mapping[: token_indices.shape[0]].copy_(
            common_attn_metadata.slot_mapping[token_indices]
        )
        common_attn_metadata.slot_mapping[token_indices.shape[0] :].fill_(-1)

        # NOTE: Currently positions and seq_lens are not used in attn forward
        # so we do not need to fixed them. But if they are used in the future,
        # we should fixed them.
        spec_common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            seq_lens_cpu=new_seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            positions=common_attn_metadata.positions[token_indices],
            positions_cpu=common_attn_metadata.positions_cpu[token_indices],
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
            max_seq_len=0,
        )
        return spec_common_attn_metadata, token_indices

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.
        """
        if HAS_TRITON:
            num_reqs = common_attn_metadata.num_reqs
            device = valid_sampled_tokens_count.device

            token_indices_to_sample = torch.empty((num_reqs,), dtype=torch.int32, device=device)
            num_rejected_tokens_gpu = torch.empty((num_reqs,), dtype=torch.int32, device=device)
            num_blocks_needed = triton.cdiv(num_reqs, _PREPARE_INPUTS_BLOCK_SIZE)
            num_vector_core = get_vectorcore_num()
            grid_size = min(num_blocks_needed, num_vector_core)
            grid = (grid_size,)

            prepare_inputs_padded_kernel[grid](
                spec_decode_metadata.cu_num_draft_tokens,
                valid_sampled_tokens_count,
                common_attn_metadata.query_start_loc,
                token_indices_to_sample,
                num_rejected_tokens_gpu,
                num_reqs,
                BLOCK_SIZE=_PREPARE_INPUTS_BLOCK_SIZE,
            )
        else:
            num_draft_tokens_gpu = torch.cat(
                [
                    spec_decode_metadata.cu_num_draft_tokens[0:1],
                    spec_decode_metadata.cu_num_draft_tokens[1:] - spec_decode_metadata.cu_num_draft_tokens[:-1],
                ]
            )

            num_rejected_tokens_gpu = torch.where(
                num_draft_tokens_gpu > 0,
                num_draft_tokens_gpu + 1 - valid_sampled_tokens_count,
                torch.zeros_like(num_draft_tokens_gpu),
            )

            token_indices_to_sample = common_attn_metadata.query_start_loc[1:] - 1 - num_rejected_tokens_gpu

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()
        token_indices = self.arange[:total_num_tokens]

        # NOTE: Currently positions and seq_lens are not used in attn forward
        # so we do not need to fixed them. But if they are used in the future,
        # we should fixed them.
        spec_common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens if self.pcp_size > 1 else total_num_tokens,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            positions=common_attn_metadata.positions,
            positions_cpu=common_attn_metadata.positions_cpu,
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
            num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu,
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=0,
        )

        return spec_common_attn_metadata, token_indices, token_indices_to_sample, num_rejected_tokens_gpu

    def _split_pcp_input(self, req_scheduled_tokens, input_ids, target_hidden_states):
        """
        Split prefill input_ids and target_hidden_states in pcp group.
        1. input_ids padding: [t0, t1, t2, t3, t4, t5] -> [t0, t1, t2, t3, t4, t5, pad, pad]
        2. split input_ids: pcp0 [t0, t1, pad, pad], pcp1 [t2, t3, t4, t5]
        3. split target_hidden_states (already include pcp padding):
        [h0, h1, h2, h3, h4, h5, pad, pad] -> pcp0 [h0, h1, pad, pad], pcp1 [h2, h3, h4, h5]
        4. also update max_query_len, seq_lens, cu_num_tokens according to pcp split.
        """
        if len(req_scheduled_tokens) == 0:
            # no prefill inputs to split, return empty result
            return (
                0,
                torch.zeros([0], device="npu"),
                torch.zeros([0, target_hidden_states.size(1)], device="npu"),
                0,
                torch.zeros([0]),
                torch.tensor([0], dtype=torch.int32),
            )

        def _pcp_pad_and_split(num_tokens):
            num_pcp_padded_scheduled_tokens = cdiv(num_tokens, 2 * self.pcp_size) * 2 * self.pcp_size
            pcp_pad = num_pcp_padded_scheduled_tokens - num_tokens
            chunk_size = num_pcp_padded_scheduled_tokens // (2 * self.pcp_size)

            # split position_ids (and use split position_ids to split input_ids afterwards)
            req_position_cp: list[int] = []
            req_position_cp.extend(self.full_indices[self.pcp_rank * chunk_size : (self.pcp_rank + 1) * chunk_size])
            req_position_cp.extend(
                self.full_indices[
                    num_pcp_padded_scheduled_tokens - (self.pcp_rank + 1) * chunk_size : num_pcp_padded_scheduled_tokens
                    - self.pcp_rank * chunk_size
                ]
            )

            return req_position_cp, num_pcp_padded_scheduled_tokens, pcp_pad

        num_pcp_scheduled_tokens = []
        ori_start_index = 0
        pad_start_index = 0
        pcp_split_input_ids_list = []
        pcp_split_hidden_states_list = []
        for ori_num_tokens in req_scheduled_tokens.values():
            req_position_pcp, num_pcp_padded_scheduled_tokens, num_pcp_pad = _pcp_pad_and_split(ori_num_tokens)
            actual_num_tokens = len(req_position_pcp)
            num_pcp_scheduled_tokens.append(actual_num_tokens)
            pad_input_ids = F.pad(input_ids[ori_start_index : ori_start_index + ori_num_tokens], (0, num_pcp_pad))
            ori_start_index += ori_num_tokens
            pcp_chunk_indices = [pad_start_index + pos for pos in req_position_pcp]
            pcp_split_input_ids = pad_input_ids[req_position_pcp]
            pcp_split_hidden_states = target_hidden_states[pcp_chunk_indices]
            pcp_split_input_ids_list.append(pcp_split_input_ids)
            pcp_split_hidden_states_list.append(pcp_split_hidden_states)
            pad_start_index += num_pcp_padded_scheduled_tokens
        num_tokens = sum(num_pcp_scheduled_tokens)
        input_ids = torch.cat(pcp_split_input_ids_list)
        target_hidden_states = torch.cat(pcp_split_hidden_states_list, dim=0)
        max_query_len = max(num_pcp_scheduled_tokens)
        seq_lens = torch.tensor(num_pcp_scheduled_tokens, dtype=torch.int32)
        cu_num_tokens = torch.tensor(np.insert(np.cumsum(np.array(num_pcp_scheduled_tokens)), 0, 0))
        return num_tokens, input_ids, target_hidden_states, max_query_len, seq_lens, cu_num_tokens

    # update full-graph params for one spec token
    def _update_full_graph_params(self, forward_context, num_tokens, draft_attn_metadatas=None):
        assert len(self.draft_attn_groups) > 0
        attn_backend = self.draft_attn_groups[0].backend
        update_full_graph_params(
            attn_backend,
            self.update_stream,
            forward_context,
            num_tokens,
            self.vllm_config,
            self.vllm_config.speculative_config,
            draft_attn_metadatas=draft_attn_metadatas,
        )

    # padding tensor into desired size
    def _pad_tensor(self, tensor, desired_size):
        pad_size = desired_size - tensor.shape[0]
        if pad_size > 0:
            pad = [0] * (2 * tensor.dim() - 1) + [pad_size]
            tensor = F.pad(tensor, pad, mode="constant", value=0)
        else:
            tensor = tensor[:desired_size]
        return tensor

    def maybe_pad_and_reduce(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_multimodal_model and _EXTRA_CTX.flash_comm_v1_enabled:
            return hidden_states, positions
        if self.method == "mtp":
            if _EXTRA_CTX.flash_comm_v1_enabled:
                hidden_states = torch.ops.vllm.maybe_pad_and_reduce(hidden_states)
                positions = positions.unsqueeze(-1)
                positions = torch.ops.vllm.maybe_pad_and_reduce(positions)
                positions = positions.squeeze(-1)
        else:
            if _EXTRA_CTX.flash_comm_v1_enabled:
                hidden_states = split_inputs_tp_to_sp(hidden_states, hidden_states)
        return hidden_states, positions

    def maybe_all_gather_and_unpad(
        self,
        last_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.method == "mtp":
            if self.enable_shared_expert_dp:
                last_hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                    last_hidden_states.contiguous(), True
                )
                positions = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(positions.contiguous(), True)
                if hidden_states is not None:
                    hidden_states = last_hidden_states
        else:
            if _EXTRA_CTX.flash_comm_v1_enabled:
                last_hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                    last_hidden_states.contiguous(), True
                )
                if hidden_states is not None:
                    hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states.contiguous(), True)
        return last_hidden_states, positions, hidden_states


class AscendEagleProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
