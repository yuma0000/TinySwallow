
def is_torch_greater_or_equal(library_version: str, accept_dev: bool = False):
    if not _is_package_available("torch"):
        return False

    if accept_dev:
        return version.parse(version.parse(importlib.metadata.version("torch")).base_version) >= version.parse(
            library_version
        )
    else:
        return version.parse(importlib.metadata.version("torch")) >= version.parse(library_version)

_is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)

def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return kv_idx <= q_idx

def prepare_padding_mask(
    attention_mask: Optional[torch.Tensor], kv_length: int, kv_offset: int, _slice: bool = True
) -> Optional[torch.Tensor]:
    local_padding_mask = attention_mask
    if attention_mask is not None:
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
        if _slice:
            mask_indices = torch.arange(kv_length, device=local_padding_mask.device)
            mask_indices += kv_offset
            local_padding_mask = local_padding_mask[:, mask_indices]
    return local_padding_mask

_torch_available = False

def is_torch_available():
    return _torch_available

def is_torchdynamo_compiling():
    if not is_torch_available():
        return False

    import torch

    return torch.compiler.is_compiling()

def _ignore_causal_mask_sdpa(
    padding_mask: Optional[torch.Tensor],
    query_length: int,
    kv_length: int,
    kv_offset: int,
    local_attention_size: Optional[int] = None,
) -> bool:
    is_tracing = torch.jit.is_tracing() or isinstance(padding_mask, torch.fx.Proxy) or is_torchdynamo_compiling()
    if padding_mask is not None and padding_mask.shape[-1] > kv_length:
        mask_indices = torch.arange(kv_length, device=padding_mask.device)
        mask_indices += kv_offset
        padding_mask = padding_mask[:, mask_indices]

    if (
        not is_tracing
        and (query_length == 1 or kv_length == query_length)
        and (local_attention_size is None or kv_length < local_attention_size)
        and (padding_mask is None or padding_mask.all())
    ):
        return True

    return False

def sdpa_mask_recent_torch(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]:
    q_length = cache_position.shape[0]
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)

    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    kv_arange = torch.arange(kv_length, device=cache_position.device)
    kv_arange += kv_offset

    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    batch_arange = torch.arange(batch_size, device=cache_position.device)
    head_arange = torch.arange(1, device=cache_position.device)
    with TransformGetItemToIndex():
        causal_mask = _vmap_for_bhqkv(mask_function)(batch_arange, head_arange, cache_position, kv_arange)

    return causal_mask

sdpa_mask = sdpa_mask_recent_torch if _is_torch_greater_or_equal_than_2_6 else sdpa_mask_older_torch

def _preprocess_mask_arguments(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, torch.Tensor]],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor],
    layer_idx: Optional[int],
) -> tuple[bool, Optional[Union[torch.Tensor, torch.Tensor]], int, int]:

    if isinstance(attention_mask, (torch.Tensor, torch.Tensor)) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None

    if config._attn_implementation not in AttentionMaskInterface._global_mapping:
        return True, None, None, None, None

    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=cache_position.device, dtype=torch.bool)

    if past_key_values is not None:
        kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
    else:
        kv_length, kv_offset = input_embeds.shape[1], 0

    packed_sequence_mask = None
    if position_ids is not None and attention_mask is None and past_key_values is None:
        batch_size = input_embeds.shape[0]
        if batch_size != position_ids.shape[0]:
            position_ids = position_ids.expand(batch_size, -1)
        packed_sequence_mask = find_packed_sequence_indices(position_ids)

    return False, attention_mask, packed_sequence_mask, kv_length, kv_offset

def create_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, torch.Tensor]]:
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = causal_mask_function
    mask_interface = sdpa_mask

    allow_is_causal_skip = not past_key_values.is_compileable if past_key_values is not None else True

    if packed_sequence_mask is not None and _is_torch_greater_or_equal_than_2_6:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    if or_mask_function is not None:
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,
        dtype=dtype,
        config=config,
    )
    return causal_mask

def _vec_softmax(input: torch.Tensor, dim: int, dtype: Optional[torch.dtype] = None):

    dim = dim if dim >= 0 else input.dim() + dim

    shape = input.shape
    assert 0 <= dim < input.dim()

    outer_size = 1
    for i in range(dim):
        outer_size *= shape[i]

    inner_size = 1
    for i in range(dim + 1, input.dim()):
        inner_size *= shape[i]

    dim_size = shape[dim]

    input_transposed = input.permute(
        *[i for i in range(input.dim()) if i != dim], dim
    ).contiguous()
    reshaped = input_transposed.view(outer_size, dim_size, inner_size)

    max_vals, _ = reshaped.max(dim=1, keepdim=True)

    exps = (reshaped - max_vals).exp()
    sum_exps = exps.sum(dim=1, keepdim=True)

    softmaxed = exps / sum_exps

    output = softmaxed.view(input_transposed.shape).permute(
        *[i for i in range(input.dim()) if i != dim], input.dim() - 1
    ).contiguous()

    return output

def softmax(input: torch.Tensor, dim: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:

    orig_dtype = input.dtype
    if dtype is not None:
        input = input.to(dtype)
    elif input.dtype in (torch.float16, torch.bfloat16):
        input = input.to(torch.float32)

    max_val, _ = input.max(dim=dim, keepdim=True)
    input = input - max_val

    exp_input = input.exp()
    sum_exp = exp_input.sum(dim=dim, keepdim=True)
    softmax_output = exp_input / sum_exp

    return softmax_output.to(orig_dtype)

def safe_softmax(input: torch.Tensor, dim: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    dim = dim if dim >= 0 else input.dim() + dim

    out = softmax(input, dim=dim, dtype=dtype)

    masked = input == float("-inf")

    masked_rows = masked.all(dim=dim, keepdim=True)

    out = torch.where(masked_rows, torch.zeros_like(out), out)
    return out

def dropout_cpu(input: torch.Tensor, p: float, train: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if input.numel() == 0:
        return input, torch.empty_like(input)

    if train is None:
        train = True

    if train:
        p1m = 1.0 - p
        scale = 0.0 if p1m == 0 else 1.0 / p1m
        mask = torch.empty_like(input, dtype=torch.bool).bernoulli_(p1m)
        output = input * mask.to(input.dtype) * scale
    else:
        mask = torch.ones_like(input, dtype=torch.bool)
        output = input.clone()

    return output, mask

def dropout(input: torch.Tensor, p: float = 0.5, train: bool = True) -> torch.Tensor:
    if input.is_nested():
        raise NotImplementedError("nested tensor の dropout は未対応です")

    return dropout_cpu(input, p=p, train=train)

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor]:

    origin_dtype = query.dtype
    def maybe_upcast(t):
        return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t

    query = maybe_upcast(query)
    key = maybe_upcast(key)
    value = maybe_upcast(value)

    if query.size(1) != key.size(1):
        qh, kh = query.size(1), key.size(1)
        assert qh % kh == 0, "GQA: query_heads must be divisible by key_heads"
        repeat_factor = qh // kh
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

    if scale is None:
        scale = 1.0 / (key.size(-1) ** 0.5)

    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    if is_causal:
        q_len, k_len = attn_weights.size(-2), attn_weights.size(-1)
        causal_mask = torch.tril(torch.ones(q_len, k_len, device=attn_weights.device, dtype=torch.bool))
        attn_weights = attn_weights.masked_fill(~causal_mask.view(1, 1, q_len, k_len), float('-inf'))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))
        else:
            attn_weights = attn_weights + attn_mask

    attn_probs = safe_softmax(attn_weights, dim=-1)

    if dropout_p > 0.0 and torch.is_grad_enabled():
        attn_probs = dropout(attn_probs, p=dropout_p, training=True)

    out = torch.matmul(attn_probs, value)

    return out.to(origin_dtype)

class PreTrainedModel(nn.Module):
    _auto_class = None
    _keep_in_fp32_modules = None
    _keep_in_fp32_modules_strict = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None
    main_input_name = "input_ids"

    @property
    def device(self) -> torch.device:
        return get_parameter_device(self)

    def _check_attn_implementation(cls, attn_implementation: Union[dict, str]) -> Union[dict, str]:
        if isinstance(attn_implementation, str) and re.match(r"^[^/:]+/[^/:]+:[^/:]+$", attn_implementation):
            repo_id, kernel_name = attn_implementation.split(":")
            kernel_name = kernel_name.strip()
            repo_id = repo_id.strip()

            kernel = get_kernel(repo_id)
            AttentionInterface.register(f"kernel_{repo_id.replace('/', '_')}", getattr(kernel, kernel_name))
            attn_implementation = f"kernel_{repo_id.replace('/', '_')}"
        return attn_implementation

    def set_attention_implementation(self, attn_implementation: Union[dict, str]):
        requested_attn_implementation = self._check_attn_implementation(attn_implementation)

        for key in self.config.sub_configs.keys():
            sub_config = getattr(self.config, key)
            curr_attn_implementation = (
                requested_attn_implementation
                if not isinstance(requested_attn_implementation, dict)
                else requested_attn_implementation.get(key, None)
            )
            if (
                sub_config is not None
                and sub_config._attn_implementation_internal is None
                and curr_attn_implementation is not None
            ):
                sub_config._attn_implementation_internal = curr_attn_implementation

        if requested_attn_implementation == "flash_attention_3" and self._flash_attn_3_can_dispatch():
            self.config._attn_implementation = "flash_attention_3"
        if requested_attn_implementation == "flash_attention_2" and self._flash_attn_2_can_dispatch():
            self.config._attn_implementation = "flash_attention_2"
        elif requested_attn_implementation == "flex_attention" and self._flex_attn_can_dispatch():
            self.config._attn_implementation = "flex_attention"
        elif (
            requested_attn_implementation in [None, "sdpa"]
            and not is_torch_xla_available()
            and self._sdpa_can_dispatch(hard_check_only=requested_attn_implementation is not None)
        ):
            self.config._attn_implementation = "sdpa"
        elif requested_attn_implementation in AttentionInterface.valid_keys():
            self.config._attn_implementation = requested_attn_implementation
        elif isinstance(requested_attn_implementation, dict):
            self.config._attn_implementation = requested_attn_implementation.get("", None)
        else:
            self.config._attn_implementation = "eager"

        self.config._attn_implementation_autoset = True

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__()
        self.config = config

        if hasattr(config, "_attn_implementation_internal") and not getattr(
            config, "_attn_implementation_autoset", False
        ):
            self.set_attention_implementation(self.config._attn_implementation_internal)

        loss_type = self.__class__.__name__
        if loss_type not in LOSS_MAPPING:
            loss_groups = f"({'|'.join(LOSS_MAPPING)})"
            loss_type = re.findall(loss_groups, self.__class__.__name__)
            if len(loss_type) > 0:
                loss_type = loss_type[0]
            else:
                loss_type = None
        self.loss_type = loss_type

        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        self._keep_in_fp32_modules = copy.copy(self.__class__._keep_in_fp32_modules)
        self._keep_in_fp32_modules_strict = copy.copy(self.__class__._keep_in_fp32_modules_strict)

        self._no_split_modules = self._no_split_modules or []
        _CAN_RECORD_REGISTRY[str(self.__class__)] = self._can_record_outputs

    @classmethod
    @restore_default_torch_dtype
    def from_pretrained(
        cls: type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ) -> SpecificPreTrainedModelType:
        state_dict = kwargs.pop("state_dict", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        proxies = kwargs.pop("proxies", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        torch_dtype = kwargs.pop("torch_dtype", None)
        device_map = kwargs.pop("device_map", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        variant = kwargs.pop("variant", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", {})
        gguf_file = kwargs.pop("gguf_file", None)
        tp_size = kwargs.pop("tp_size", None)
        device_mesh = kwargs.pop("device_mesh", None)
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None and any(
            allowed_name in class_name.__name__.lower() for class_name in cls.__mro__[:-1] for allowed_name in VLMS
        ):
            key_mapping = cls._checkpoint_conversion_mapping

        _ = kwargs.pop("resume_download", None)
        _ = kwargs.pop("mirror", None)
        _ = kwargs.pop("_fast_init", True)
        _ = kwargs.pop("low_cpu_mem_usage", None)

        if commit_hash is None:
            commit_hash = getattr(config, "_commit_hash", None)

        if is_peft_available():
            _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

            if _adapter_model_path is None:
                _adapter_model_path = find_adapter_config_file(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    _commit_hash=commit_hash,
                    **adapter_kwargs,
                )
            if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
                with open(_adapter_model_path, "r", encoding="utf-8") as f:
                    _adapter_model_path = pretrained_model_name_or_path
                    pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]

        if device_map is None and not is_deepspeed_zero3_enabled():
            device_in_context = get_torch_context_manager_or_global_device()

            device_map = device_in_context

        from_pt = not (from_tf | from_flax)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}

        config = copy.deepcopy(config)

        kwarg_attn_imp = kwargs.pop("attn_implementation", None)
        if kwarg_attn_imp is not None:
            config._attn_implementation = kwarg_attn_imp

        model_kwargs = kwargs


        transformers_explicit_filename = getattr(config, "transformers_weights", None)

        hf_quantizer = None

        checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            variant=variant,
            gguf_file=gguf_file,
            from_tf=from_tf,
            from_flax=from_flax,
            use_safetensors=use_safetensors,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            user_agent=user_agent,
            revision=revision,
            commit_hash=commit_hash,
            is_remote_code=cls._auto_class is not None,
            transformers_explicit_filename=transformers_explicit_filename,
        )

        is_sharded = sharded_metadata is not None
        is_quantized = hf_quantizer is not None
        is_from_file = pretrained_model_name_or_path is not None or gguf_file is not None

        if (
            is_safetensors_available()
            and is_from_file
            and not is_sharded
            and checkpoint_files[0].endswith(".safetensors")
        ):
            with safe_open(checkpoint_files[0], framework="pt") as f:
                metadata = f.metadata()

        from_pt = not (from_tf | from_flax)
        if from_pt:
            config, torch_dtype, dtype_orig = _get_torch_dtype(
                cls, torch_dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only
        )

        config.name_or_path = pretrained_model_name_or_path

        model_init_context = cls.get_init_context(is_quantized, _is_ds_init_called)

        config = copy.deepcopy(config)
        with ContextManagers(model_init_context):
            model = cls(config, *model_args, **model_kwargs)

        model.tie_weights()

        config = model.config

        keep_in_fp32_regex = None

        if from_pt:
            if dtype_orig is not None:
                torch.set_default_dtype(dtype_orig)

            (
                model,
                missing_keys,
                unexpected_keys,
                mismatched_keys,
                offload_index,
                error_msgs,
            ) = cls._load_pretrained_model(
                model,
                state_dict,
                checkpoint_files,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                sharded_metadata=sharded_metadata,
                device_map=device_map,
                disk_offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                dtype=torch_dtype,
                hf_quantizer=hf_quantizer,
                keep_in_fp32_regex=keep_in_fp32_regex,
                device_mesh=device_mesh,
                key_mapping=key_mapping,
                weights_only=weights_only,
            )

        model._tp_size = tp_size
        model._device_mesh = device_mesh

        model.tie_weights()

        model.eval()

        if model.can_generate() and pretrained_model_name_or_path is not None:
            repo_loading_kwargs = {
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "local_files_only": local_files_only,
                "token": token,
                "revision": revision,
                "subfolder": subfolder,
                **kwargs,
            }

            model.generation_config = GenerationConfig.from_pretrained(
                pretrained_model_name_or_path,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **repo_loading_kwargs,
                )

        return model

    @classmethod
    def _set_default_torch_dtype(cls, dtype: torch.dtype) -> torch.dtype:

        dtype_orig = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        return dtype_orig

    @classmethod
    def get_init_context(cls, is_quantized: bool, _is_ds_init_called: bool):
        if is_deepspeed_zero3_enabled():
            import deepspeed

            init_contexts = [no_init_weights()]
            if not is_quantized and not _is_ds_init_called:
                init_contexts.extend([deepspeed.zero.Init(config_dict_or_path=deepspeed_config()), set_zero3_state()])
            elif is_quantized:
                init_contexts.extend([init_empty_weights(), set_quantized_state()])
        else:
            init_contexts = [no_init_weights(), init_empty_weights()]

        return init_contexts

    def _sdpa_can_dispatch(self, hard_check_only: bool = False) -> bool:
        if (
            torch.version.hip is not None
            and torch.cuda.device_count() > 1
            and version.parse(torch.__version__) < version.parse("2.4.1")
        ):
            torch.backends.cuda.enable_flash_sdp(False)

        _is_bettertransformer = getattr(self, "use_bettertransformer", False)
        if not is_torch_sdpa_available() or not self._supports_sdpa or _is_bettertransformer:
            return False

        return True

    @classmethod
    def can_generate(cls) -> bool:
        if "GenerationMixin" in str(cls.__bases__):
            return True
        for base in cls.__bases__:
            if not hasattr(base, "can_generate"):
                continue
            if "PreTrainedModel" not in str(base) and base.can_generate():
                return True
        return False

    def post_init(self):
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

        if self._keep_in_fp32_modules is not None or self._keep_in_fp32_modules_strict is not None:
            all_parameters = {name for name, _ in self.named_parameters() if len(name) > 0}
            unique_module_names = set()
            for param in all_parameters:
                unique_module_names.update(
                    [name for name in param.split(".") if not name.isnumeric() and name not in ["weight", "bias"]]
                )

        self._pp_plan = self.config.base_model_pp_plan.copy() if self.config.base_model_pp_plan is not None else None
        self._tp_plan = self.config.base_model_tp_plan.copy() if self.config.base_model_tp_plan is not None else {}
        for name, module in self.named_children():
            if plan := getattr(module, "_tp_plan", None):
                self._tp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})

    def _init_weights(self, module):
        pass

    def init_weights(self):
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        if _init_weights:
            self.initialize_weights()

            self.tie_weights()

    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            delattr(self.config, "gradient_checkpointing")

    def tie_weights(self):
        if getattr(self.config.get_text_config(decoder=True), "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            tied_weights = self._tie_encoder_decoder_weights(
                self.encoder, self.decoder, self.base_model_prefix, "encoder"
            )
            self._dynamic_tied_weights_keys = tied_weights

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    @classmethod
    def _load_pretrained_model(
        cls,
        model: "PreTrainedModel",
        state_dict: Optional[dict],
        checkpoint_files: Optional[list[str]],
        pretrained_model_name_or_path: Optional[str],
        ignore_mismatched_sizes: bool = False,
        sharded_metadata: Optional[dict] = None,
        device_map: Optional[dict] = None,
        disk_offload_folder: Optional[str] = None,
        offload_state_dict: Optional[bool] = None,
        dtype: Optional[torch.dtype] = None,
        hf_quantizer: Optional[HfQuantizer] = None,
        keep_in_fp32_regex: Optional[re.Pattern] = None,
        device_mesh: Optional["torch.distributed.device_mesh.DeviceMesh"] = None,
        key_mapping: Optional[dict[str, str]] = None,
        weights_only: bool = True,
    ):
        is_quantized = hf_quantizer is not None
        is_hqq_or_quark = is_quantized and hf_quantizer.quantization_config.quant_method in {
            QuantizationMethod.HQQ,
            QuantizationMethod.QUARK,
        }
        is_hqq_or_bnb = is_quantized and hf_quantizer.quantization_config.quant_method in {
            QuantizationMethod.HQQ,
            QuantizationMethod.BITS_AND_BYTES,
        }

        if sharded_metadata is not None:
            original_checkpoint_keys = sharded_metadata["all_checkpoint_keys"]
        elif state_dict is not None:
            original_checkpoint_keys = list(state_dict.keys())
        else:
            original_checkpoint_keys = list(
                load_state_dict(checkpoint_files[0], map_location="meta", weights_only=weights_only).keys()
            )

        prefix = model.base_model_prefix
        _prefix = f"{prefix}."
        has_prefix_module = any(s.startswith(prefix) for s in original_checkpoint_keys) if len(prefix) > 0 else False
        expects_prefix_module = hasattr(model, prefix) if len(prefix) > 0 else False
        loading_task_model_from_base_state_dict = not has_prefix_module and expects_prefix_module
        loading_base_model_from_task_state_dict = has_prefix_module and not expects_prefix_module

        key_renaming_mapping = model._get_key_renaming_mapping(
            original_checkpoint_keys,
            key_mapping,
            loading_base_model_from_task_state_dict,
            loading_task_model_from_base_state_dict,
        )
        checkpoint_keys = list(key_renaming_mapping.values())

        missing_keys, unexpected_keys = _find_missing_and_unexpected_keys(
            cls,
            model,
            original_checkpoint_keys,
            checkpoint_keys,
            loading_base_model_from_task_state_dict,
            hf_quantizer,
            device_map,
        )
        mismatched_keys, mismatched_shapes = _find_mismatched_keys(
            model,
            state_dict,
            checkpoint_files,
            ignore_mismatched_sizes,
            key_renaming_mapping,
            is_quantized,
            weights_only,
        )

        key_renaming_mapping = {k: v for k, v in key_renaming_mapping.items() if v not in mismatched_keys}
        checkpoint_keys = list(key_renaming_mapping.values())

        model._move_missing_keys_from_meta_to_cpu(missing_keys + mismatched_keys, unexpected_keys, dtype, hf_quantizer)

        model._initialize_missing_keys(checkpoint_keys, ignore_mismatched_sizes, is_quantized)

        if keep_in_fp32_regex is not None:
            for name, param in model.named_parameters():
                if keep_in_fp32_regex.search(name):
                    param.data = param.data.to(torch.float32)

        model_to_load = model
        if loading_task_model_from_base_state_dict:
            model_to_load = getattr(model, prefix)
            key_renaming_mapping = {k: v[len(_prefix) :] for k, v in key_renaming_mapping.items()}
            checkpoint_keys = list(key_renaming_mapping.values())
            if device_map is not None:
                device_map = {k[len(_prefix) :] if k.startswith(_prefix) else k: v for k, v in device_map.items()}
            task_specific_expected_keys = [s for s in model.state_dict().keys() if not s.startswith(_prefix)]
            base_model_expected_keys = list(model_to_load.state_dict().keys())
        reverse_key_renaming_mapping = {v: k for k, v in key_renaming_mapping.items()}

        is_offloaded_safetensors = False
        disk_offload_index = None
        disk_only_shard_files = []
        if device_map is not None and "disk" in device_map.values():
            if offload_state_dict is None:
                offload_state_dict = True
            if disk_offload_folder is not None:
                os.makedirs(disk_offload_folder, exist_ok=True)
            is_offloaded_safetensors = checkpoint_files is not None and checkpoint_files[0].endswith(".safetensors")
            if is_offloaded_safetensors:
                param_device_map = expand_device_map(device_map, checkpoint_keys)
                str_dtype = str(dtype).replace("torch.", "") if dtype is not None else "float32"
                if sharded_metadata is None:
                    weight_map = dict.fromkeys(checkpoint_keys, checkpoint_files[0])
                else:
                    folder = os.path.sep.join(checkpoint_files[0].split(os.path.sep)[:-1])
                    weight_map = {
                        key_renaming_mapping[k]: v
                        for k, v in sharded_metadata["weight_map"].items()
                        if k in key_renaming_mapping
                    }
                    weight_map = {k: os.path.join(folder, v) for k, v in weight_map.items()}
                    disk_only_shard_files = get_disk_only_shard_files(device_map, weight_map)
                disk_offload_index = {
                    name: {
                        "safetensors_file": file,
                        "weight_name": reverse_key_renaming_mapping[name],
                        "dtype": str_dtype,
                    }
                    for name, file in weight_map.items()
                    if param_device_map[name] == "disk"
                }
            else:
                disk_offload_index = {}

        cpu_offload_folder = None
        cpu_offload_index = None
        if offload_state_dict:
            cpu_offload_folder = tempfile.mkdtemp()
            cpu_offload_index = {}

        elif state_dict is not None:
            checkpoint_files = [""]

        expected_keys = list(model_to_load.state_dict().keys())
        if hf_quantizer is not None:
            expected_keys = hf_quantizer.update_expected_keys(model_to_load, expected_keys, checkpoint_keys)

        if device_map is not None and not is_hqq_or_quark:
            expanded_device_map = expand_device_map(device_map, expected_keys)
            caching_allocator_warmup(model_to_load, expanded_device_map, hf_quantizer)

        args_list = [
            (
                shard_file,
                state_dict,
                disk_only_shard_files,
                is_hqq_or_bnb,
                is_quantized,
                device_map,
                hf_quantizer,
                key_renaming_mapping,
                weights_only,
                model_to_load,
                expected_keys,
                reverse_key_renaming_mapping,
                disk_offload_folder,
                disk_offload_index,
                cpu_offload_folder,
                cpu_offload_index,
                is_offloaded_safetensors,
                keep_in_fp32_regex,
                unexpected_keys,
                device_mesh,
            )
            for shard_file in checkpoint_files
        ]

        error_msgs = []

        if (
            os.environ.get("HF_ENABLE_PARALLEL_LOADING", "").upper() in ENV_VARS_TRUE_VALUES
            and not is_deepspeed_zero3_enabled()
        ):
            _error_msgs, disk_offload_index, cpu_offload_index = load_shard_files_with_threadpool(args_list)
            error_msgs += _error_msgs
        else:
            for args in args_list:
                _error_msgs, disk_offload_index, cpu_offload_index = load_shard_file(args)
                error_msgs += _error_msgs

        if disk_offload_index is not None and len(disk_offload_index) > 0:
            if loading_task_model_from_base_state_dict:
                prefix = cls.base_model_prefix
                if not is_offloaded_safetensors:
                    for weight_name in disk_offload_index:
                        shutil.move(
                            os.path.join(disk_offload_folder, f"{weight_name}.dat"),
                            os.path.join(disk_offload_folder, f"{prefix}.{weight_name}.dat"),
                        )
                disk_offload_index = {f"{prefix}.{key}": value for key, value in disk_offload_index.items()}
            if not is_offloaded_safetensors:
                save_offload_index(disk_offload_index, disk_offload_folder)
                disk_offload_index = None
        if offload_state_dict:
            load_offloaded_weights(model_to_load, cpu_offload_index, cpu_offload_folder)
            shutil.rmtree(cpu_offload_folder)

        if hf_quantizer is not None:
            missing_keys = hf_quantizer.update_missing_keys_after_loading(model_to_load, missing_keys, prefix)

        if device_mesh is not None:
            tp_device = list(device_map.values())[0]
            for buffer in model.buffers():
                if buffer.device != tp_device:
                    buffer.data = buffer.to(tp_device)

            if loading_task_model_from_base_state_dict:
                parameters_to_initialize = {
                    name: param for name, param in model.named_parameters() if not name.startswith(prefix)
                }
                for name, param in parameters_to_initialize.items():
                    if param.device.type == "meta":
                        continue
                    to_contiguous, casting_dtype = _infer_parameter_dtype(model, name, param, keep_in_fp32_regex)
                    shard_and_distribute_module(
                        model,
                        param.to(tp_device),
                        param,
                        name,
                        casting_dtype,
                        to_contiguous,
                        device_mesh.get_local_rank(),
                        device_mesh,
                    )

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
        if len(unexpected_keys) > 0:
            archs = [] if model.config.architectures is None else model.config.architectures
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, (shape1, shape2) in zip(mismatched_keys, mismatched_shapes)
                ]
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, disk_offload_index, error_msgs

    def _get_key_renaming_mapping(
        self,
        checkpoint_keys: list[str],
        key_mapping: Optional[dict[str, str]] = None,
        loading_base_model_from_task_state_dict: bool = False,
        loading_task_model_from_base_state_dict: bool = False,
    ):
        prefix = self.base_model_prefix
        _prefix = f"{prefix}."

        renamed_keys = {}
        key_renaming_mapping = {}
        for key in checkpoint_keys:
            new_key, has_changed = self._fix_state_dict_key_on_load(key)

            if key_mapping is not None:
                for pattern, replacement in key_mapping.items():
                    new_key, n_replace = re.subn(pattern, replacement, new_key)
                    if n_replace > 0:
                        has_changed = True
                        break

            if loading_task_model_from_base_state_dict:
                new_key = ".".join([prefix, new_key])
            elif loading_base_model_from_task_state_dict:
                if not new_key.startswith(_prefix):
                    continue
                new_key = new_key[len(_prefix) :]

            key_renaming_mapping[key] = new_key

            if has_changed:
                if key.endswith("LayerNorm.gamma"):
                    renamed_keys["LayerNorm.gamma"] = (key, new_key)
                elif key.endswith("LayerNorm.beta"):
                    renamed_keys["LayerNorm.beta"] = (key, new_key)

        if renamed_keys:
            warning_msg = f"A pretrained model of type `{self.__class__.__name__}` "
            warning_msg += "contains parameters that have been renamed internally (a few are listed below but more are present in the model):\n"
            for old_key, new_key in renamed_keys.values():
                warning_msg += f"* `{old_key}` -> `{new_key}`\n"
            warning_msg += "If you are using a model from the Hub, consider submitting a PR to adjust these weights and help future users."

        return key_renaming_mapping

    @staticmethod
    def _fix_state_dict_key_on_load(key: str) -> tuple[str, bool]:
        if key.endswith("LayerNorm.beta"):
            return key.replace("LayerNorm.beta", "LayerNorm.bias"), True
        if key.endswith("LayerNorm.gamma"):
            return key.replace("LayerNorm.gamma", "LayerNorm.weight"), True

        if hasattr(nn.utils.parametrizations, "weight_norm"):
            if key.endswith("weight_g"):
                return key.replace("weight_g", "parametrizations.weight.original0"), True
            if key.endswith("weight_v"):
                return key.replace("weight_v", "parametrizations.weight.original1"), True
        else:
            if key.endswith("parametrizations.weight.original0"):
                return key.replace("parametrizations.weight.original0", "weight_g"), True
            if key.endswith("parametrizations.weight.original1"):
                return key.replace("parametrizations.weight.original1", "weight_v"), True

        return key, False

    def _move_missing_keys_from_meta_to_cpu(
        self,
        missing_keys: list[str],
        unexpected_keys: list[str],
        dtype: Optional[torch.dtype],
        hf_quantizer: Optional[HfQuantizer],
    ) -> "PreTrainedModel":
        is_quantized = hf_quantizer is not None

        if is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized:
            for key, param in self.named_parameters():
                value = torch.empty_like(param, dtype=dtype, device="cpu")
                _load_parameter_into_model(self, key, value)
            return

        model_state_dict = self.state_dict()
        for key in missing_keys:
            param = model_state_dict[key]
            if param.device == torch.device("meta"):
                value = torch.empty_like(param, dtype=dtype, device="cpu")
                if (
                    not is_quantized
                    or (getattr(hf_quantizer, "requires_parameters_quantization", False))
                    or not hf_quantizer.check_quantized_param(self, param_value=value, param_name=key, state_dict={})
                ):
                    _load_parameter_into_model(self, key, value)
                else:
                    hf_quantizer.create_quantized_param(self, value, key, "cpu", model_state_dict, unexpected_keys)

    def _initialize_missing_keys(
        self,
        loaded_keys: list[str],
        ignore_mismatched_sizes: bool,
        is_quantized: bool,
    ) -> "PreTrainedModel":
        if not ignore_mismatched_sizes:
            not_initialized_submodules = set_initialized_submodules(self, loaded_keys)
            if (
                hasattr(self.config.get_text_config(decoder=True), "tie_word_embeddings")
                and self.config.get_text_config(decoder=True).tie_word_embeddings
            ):
                output_embeddings = self.get_output_embeddings()
                if output_embeddings is not None:
                    if not hasattr(output_embeddings, "bias") or output_embeddings.bias is None:
                        output_embeddings._is_hf_initialized = True
        else:
            not_initialized_submodules = dict(self.named_modules())
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            not_initialized_parameters = list(
                set(
                    itertools.chain.from_iterable(
                        submodule.parameters(recurse=False) for submodule in not_initialized_submodules.values()
                    )
                )
            )
            with deepspeed.zero.GatheredParameters(not_initialized_parameters, modifier_rank=0):
                self.initialize_weights()
        else:
            self.initialize_weights()

    @torch.no_grad()
    def initialize_weights(self):
        if not hasattr(torch.nn.Module, "smart_apply"):
            def smart_apply(self, fn):
                for module in self.children():
                    if isinstance(module, PreTrainedModel):
                        module.smart_apply(module._initialize_weights)
                    else:
                        module.smart_apply(fn)
                fn(self)
                return self

            torch.nn.Module.smart_apply = smart_apply

        self.smart_apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if getattr(module, "_is_hf_initialized", False):
            return
        self._init_weights(module)
        module._is_hf_initialized = True

    def get_parameter_or_buffer(self, target: str):
        return self.get_parameter(target)
        return self.get_buffer(target)
        module, param_name = get_module_from_name(self, target)
        if (
            param_name == "_extra_state"
            and getattr(module.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
            is not torch.nn.Module.get_extra_state
        ):
            return module.get_extra_state()

        raise AttributeError(f"`{target}` is neither a parameter, buffer, nor extra state.")

class _LazyAutoMapping(OrderedDict[type[PretrainedConfig], _LazyAutoMappingValue]):
    def __init__(self, config_mapping, model_mapping) -> None:
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._model_mapping._model_mapping = self
        self._extra_content = {}
        self._modules = {}

    def __len__(self) -> int:
        common_keys = set(self._config_mapping.keys()).intersection(self._model_mapping.keys())
        return len(common_keys) + len(self._extra_content)

    def __getitem__(self, key: type[PretrainedConfig]) -> _LazyAutoMappingValue:
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)

        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)

    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self) -> list[type[PretrainedConfig]]:
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key: type[PretrainedConfig], default: _T) -> Union[_LazyAutoMappingValue, _T]:
        return self.__getitem__(key)

    def values(self) -> list[_LazyAutoMappingValue]:
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return mapping_values + list(self._extra_content.values())

    def items(self) -> list[tuple[type[PretrainedConfig], _LazyAutoMappingValue]]:
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self) -> Iterator[type[PretrainedConfig]]:
        return iter(self.keys())

    def register(self, key: type[PretrainedConfig], value: _LazyAutoMappingValue, exist_ok=False) -> None:
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            model_type = self._reverse_config_mapping[key.__name__]

        self._extra_content[key] = value

TOKENIZER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)

class GELUActivation(nn.Module):
    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: torch.Tensor) -> torch.Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.act(input)

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)

class ClippedGELUActivation(nn.Module):
    def __init__(self, min: float, max: float):

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clip(gelu(x), self.min, self.max)

class FastGELUActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))

class NewGELUActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class PytorchGELUTanh(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(input, approximate="tanh")

class AccurateGELUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1 + torch.tanh(self.precomputed_constant * (input + 0.044715 * torch.pow(input, 3))))

class LaplaceActivation(nn.Module):
    def forward(self, input, mu=0.707107, sigma=0.282095):
        input = (input - mu).div(sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + torch.erf(input))

class LinearActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input

class MishActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.functional.mish

    def _mish_python(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.tanh(nn.functional.softplus(input))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.act(input)

class QuickGELUActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(1.702 * input)

class ReLUSquaredActivation(nn.Module):
    def forward(self, input):
        relu_applied = nn.functional.relu(input)
        squared = torch.square(relu_applied)
        return squared

class AutoTokenizer:

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            kwargs["token"] = use_auth_token

        config = kwargs.pop("config", None)
        kwargs["_from_auto"] = True

        use_fast = kwargs.pop("use_fast", True)
        tokenizer_type = kwargs.pop("tokenizer_type", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        gguf_file = kwargs.get("gguf_file", None)

        if tokenizer_type is not None:
            tokenizer_class = None
            tokenizer_class_tuple = TOKENIZER_MAPPING_NAMES.get(tokenizer_type, None)

            tokenizer_class_name, tokenizer_fast_class_name = tokenizer_class_tuple

            if use_fast:
                if tokenizer_fast_class_name is not None:
                    tokenizer_class = tokenizer_class_from_name(tokenizer_fast_class_name)
            if tokenizer_class is None:
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_name)

            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        if "_commit_hash" in tokenizer_config:
            kwargs["_commit_hash"] = tokenizer_config["_commit_hash"]
        config_tokenizer_class = tokenizer_config.get("tokenizer_class")
        tokenizer_auto_map = None
        if "auto_map" in tokenizer_config:
            if isinstance(tokenizer_config["auto_map"], (tuple, list)):
                tokenizer_auto_map = tokenizer_config["auto_map"]
            else:
                tokenizer_auto_map = tokenizer_config["auto_map"].get("AutoTokenizer", None)

        if config_tokenizer_class is None:
            if not isinstance(config, PretrainedConfig):
                if gguf_file:
                    gguf_path = cached_file(pretrained_model_name_or_path, gguf_file, **kwargs)
                    config_dict = load_gguf_checkpoint(gguf_path, return_tensors=False)["config"]
                    config = AutoConfig.for_model(**config_dict)
                else:
                    config = AutoConfig.from_pretrained(
                        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                    )
            config_tokenizer_class = config.tokenizer_class
            if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
                tokenizer_auto_map = config.auto_map["AutoTokenizer"]

        has_remote_code = tokenizer_auto_map is not None
        has_local_code = type(config) in TOKENIZER_MAPPING or (
            config_tokenizer_class is not None
            and (
                tokenizer_class_from_name(config_tokenizer_class) is not None
                or tokenizer_class_from_name(config_tokenizer_class + "Fast") is not None
            )
        )
        if has_remote_code:
            if use_fast and tokenizer_auto_map[1] is not None:
                class_ref = tokenizer_auto_map[1]
            else:
                class_ref = tokenizer_auto_map[0]
            if "--" in class_ref:
                upstream_repo = class_ref.split("--")[0]
            else:
                upstream_repo = None
            trust_remote_code = resolve_trust_remote_code(
                trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code, upstream_repo
            )

        if has_remote_code and trust_remote_code:
            tokenizer_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
            _ = kwargs.pop("code_revision", None)
            tokenizer_class.register_for_auto_class()
            return tokenizer_class.from_pretrained(
                pretrained_model_name_or_path, *inputs, trust_remote_code=trust_remote_code, **kwargs
            )
        elif config_tokenizer_class is not None:
            tokenizer_class = None
            if use_fast and not config_tokenizer_class.endswith("Fast"):
                tokenizer_class_candidate = f"{config_tokenizer_class}Fast"
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            if tokenizer_class is None:
                tokenizer_class_candidate = config_tokenizer_class
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        if isinstance(config, EncoderDecoderConfig):
            config = config.encoder

        model_type = config_class_to_model_type(type(config).__name__)
        if model_type is not None:
            tokenizer_class_py, tokenizer_class_fast = TOKENIZER_MAPPING[type(config)]

            if tokenizer_class_fast and (use_fast or tokenizer_class_py is None):
                return tokenizer_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
            else:
                if tokenizer_class_py is not None:
                    return tokenizer_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

    @staticmethod
    def register(config_class, slow_tokenizer_class=None, fast_tokenizer_class=None, exist_ok=False):
        if config_class in TOKENIZER_MAPPING._extra_content:
            existing_slow, existing_fast = TOKENIZER_MAPPING[config_class]
            if slow_tokenizer_class is None:
                slow_tokenizer_class = existing_slow
            if fast_tokenizer_class is None:
                fast_tokenizer_class = existing_fast

        TOKENIZER_MAPPING.register(config_class, (slow_tokenizer_class, fast_tokenizer_class), exist_ok=exist_ok)


TOKENIZER_MAPPING_NAMES = OrderedDict[str, tuple[Optional[str], Optional[str]]](
[
    ("colqwen2", ("Qwen2Tokenizer", "Qwen2TokenizerFast" if is_tokenizers_available() else None)),
    ("internvl", ("Qwen2Tokenizer", "Qwen2TokenizerFast" if is_tokenizers_available() else None)),
    (
        "qwen2",
        (
            "Qwen2Tokenizer",
            "Qwen2TokenizerFast" if is_tokenizers_available() else None,
        ),
    ),
          ("qwen2_5_omni", ("Qwen2Tokenizer", "Qwen2TokenizerFast" if is_tokenizers_available() else None)),
        ("qwen2_5_vl", ("Qwen2Tokenizer", "Qwen2TokenizerFast" if is_tokenizers_available() else None)),
        ("qwen2_audio", ("Qwen2Tokenizer", "Qwen2TokenizerFast" if is_tokenizers_available() else None)),
        (
            "qwen2_moe",
            (
                "Qwen2Tokenizer",
                "Qwen2TokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        ("qwen2_vl", ("Qwen2Tokenizer", "Qwen2TokenizerFast" if is_tokenizers_available() else None)),
        (
            "qwen3",
            (
                "Qwen2Tokenizer",
                "Qwen2TokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
        (
            "qwen3_moe",
            (
                "Qwen2Tokenizer",
                "Qwen2TokenizerFast" if is_tokenizers_available() else None,
            ),
        ),
])

_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])

def _compute_linear_scaling_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
) -> tuple["torch.Tensor", float]:
    factor = config.rope_scaling["factor"]

    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len)

    inv_freq /= factor
    return inv_freq, attention_factor


def _compute_dynamic_ntk_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
) -> tuple["torch.Tensor", float]:
    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    max_position_embeddings = config.max_position_embeddings
    factor = config.rope_scaling["factor"]

    attention_factor = 1.0

    if seq_len is None:
        seq_len = max_position_embeddings
    elif isinstance(seq_len, torch.Tensor):
        seq_len = torch.maximum(
            seq_len,
            torch.tensor(max_position_embeddings, dtype=seq_len.dtype, device=seq_len.device),
        )
    else:
        seq_len = max(seq_len, max_position_embeddings)

    base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


def _compute_yarn_parameters(
    config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None
) -> tuple["torch.Tensor", float]:
    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    factor = config.rope_scaling["factor"]
    attention_factor = config.rope_scaling.get("attention_factor")
    mscale = config.rope_scaling.get("mscale")
    mscale_all_dim = config.rope_scaling.get("mscale_all_dim")

    if "original_max_position_embeddings" in config.rope_scaling:
        original_max_position_embeddings = config.rope_scaling["original_max_position_embeddings"]
        factor = config.max_position_embeddings / original_max_position_embeddings
    else:
        original_max_position_embeddings = config.max_position_embeddings

    def get_mscale(scale, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
        else:
            attention_factor = get_mscale(factor)

    beta_fast = config.rope_scaling.get("beta_fast") or 32
    beta_slow = config.rope_scaling.get("beta_slow") or 1

    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)

    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device, dtype=torch.float)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_factor

def _compute_longrope_parameters(
    config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None
) -> tuple["torch.Tensor", float]:
    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    long_factor = config.rope_scaling["long_factor"]
    short_factor = config.rope_scaling["short_factor"]
    factor = config.rope_scaling.get("factor")
    attention_factor = config.rope_scaling.get("attention_factor")

    if hasattr(config, "original_max_position_embeddings"):
        original_max_position_embeddings = config.original_max_position_embeddings
        factor = config.max_position_embeddings / config.original_max_position_embeddings
    else:
        original_max_position_embeddings = config.max_position_embeddings

    if attention_factor is None:
        if factor <= 1.0:
            attention_factor = 1.0
        else:
            attention_factor = math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))

    if seq_len and seq_len > original_max_position_embeddings:
        ext_factors = torch.tensor(long_factor, dtype=torch.float32, device=device)
    else:
        ext_factors = torch.tensor(short_factor, dtype=torch.float32, device=device)
    inv_freq_shape = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

    return inv_freq, attention_factor

def _compute_llama3_parameters(
    config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None
) -> tuple["torch.Tensor", float]:
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len)

    factor = config.rope_scaling["factor"]
    low_freq_factor = config.rope_scaling["low_freq_factor"]
    high_freq_factor = config.rope_scaling["high_freq_factor"]
    old_context_len = config.rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq

    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor


def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
) -> tuple["torch.Tensor", float]:
    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor

    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
    validation_fn = ROPE_VALIDATION_FUNCTIONS.get(rope_type)
    if validation_fn is not None:
        validation_fn(config, ignore_keys=ignore_keys)

ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
}

@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None


@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None


@dataclass
class GenerateBeamDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None


@dataclass
class GenerateBeamEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    encoder_attentions: Optional[tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[tuple[tuple[tuple[torch.FloatTensor]]]] = None

GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]

class GenerationMixin():
    def load_custom_generate(
        self,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        trust_remote_code: Optional[bool] = None,
        **kwargs,
    ) -> Callable:
        is_local_code = os.path.exists(pretrained_model_name_or_path)
        has_custom_generate_folder = True
        if is_local_code:
            if not os.path.exists(os.path.join(pretrained_model_name_or_path, "custom_generate/generate.py")):
                has_custom_generate_folder = False
        else:
            if not file_exists(pretrained_model_name_or_path, "custom_generate/generate.py"):
                has_custom_generate_folder = False

    def _validate_model_kwargs(self, model_kwargs: dict[str, Any]):
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)

        if self.config.is_encoder_decoder:
            base_model = getattr(self, self.base_model_prefix, None)

            encoder = getattr(self, "encoder", None)
            if encoder is None and base_model is not None:
                encoder = getattr(base_model, "encoder", None)

            if encoder is not None:
                encoder_model_args = set(inspect.signature(encoder.forward).parameters)
                model_args |= encoder_model_args

            decoder = getattr(self, "decoder", None)
            if decoder is None and base_model is not None:
                decoder = getattr(base_model, "decoder", None)

            if decoder is not None:
                decoder_model_args = set(inspect.signature(decoder.forward).parameters)
                model_args |= {f"decoder_{x}" for x in decoder_model_args}

        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = {}
        model_inputs["cache_position"] = cache_position

        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            inputs_embeds, input_ids = self._cache_dependant_input_preparation(
                input_ids, inputs_embeds, cache_position
            )

        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

        encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
        attention_mask = (
            kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
        )
        attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
        position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
        if (
            attention_mask is not None
            and kwargs.get(position_ids_key) is None
            and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs[position_ids_key] = position_ids

        for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        if (
            isinstance(past_key_values, Cache)
            and past_key_values.is_compileable
            and attention_mask is not None
            and attention_mask.ndim == 2
        ):
            if not self.config.is_encoder_decoder and model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape[:2]

            base_model = getattr(self, self.base_model_prefix, self)
            decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
            causal_mask_creation_function = getattr(
                base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
            if causal_mask_creation_function is None and decoder is not None:
                causal_mask_creation_function = getattr(
                    decoder, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )

            if causal_mask_creation_function is None:
                token_type_ids = getattr(model_input, "token_type_ids", None)
                position_ids = getattr(model_input, position_ids_key, None)
                causal_mask_creation_function = getattr(self, "create_masks_for_generate", create_masks_for_generate)
                attention_mask = causal_mask_creation_function(
                    config=self.config,
                    input_embeds=torch.empty((batch_size, sequence_length), dtype=self.dtype),
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    token_type_ids=token_type_ids,
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask is not None:
            model_inputs[attention_mask_key] = attention_mask

        if encoder_attention_mask is not None:
            model_inputs["attention_mask"] = encoder_attention_mask

        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        model_inputs.pop("labels", None)
        return model_inputs

    def _validate_assistant(self, assistant_model, tokenizer, assistant_tokenizer):
        if assistant_model is None:
            return

        if self.config.is_encoder_decoder and not assistant_model.config.is_encoder_decoder:
            attributes_to_check = ["encoder_attention_heads", "encoder_ffn_dim", "encoder_layers"]
            attributes_to_check = [attr for attr in dir(assistant_model.config) if attr in attributes_to_check]
            are_equal = all(
                getattr(self.config, attr) == getattr(assistant_model.config, attr) for attr in attributes_to_check
            )

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[str], dict[str, torch.Tensor]]:
        if (
            self.config.is_encoder_decoder
            and hasattr(self, "encoder")
            and self.encoder.main_input_name != self.main_input_name
        ):
            input_name = self.encoder.main_input_name
        else:
            input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None:
            inputs = inputs_kwarg

        if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
            if model_kwargs["inputs_embeds"] is None:
                model_kwargs.pop("inputs_embeds")
            elif not self.config.is_encoder_decoder:
                has_inputs_embeds_forwarding = "inputs_embeds" in set(
                    inspect.signature(self.prepare_inputs_for_generation).parameters.keys()
                )

                model_kwargs["input_ids"] = self._maybe_initialize_input_ids_for_generation(
                    inputs, bos_token_id, model_kwargs=model_kwargs
                )
                inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"
            else:
                inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:
        if inputs is not None:
            return inputs

        encoder_outputs = model_kwargs.get("encoder_outputs")
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            shape = encoder_outputs.last_hidden_state.size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100

        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break

        if "inputs_embeds" in model_kwargs:
            return torch.ones((batch_size, 0), dtype=torch.long, device=self.device)

        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    def _prepare_special_tokens(
        self,
        generation_config: GenerationConfig,
        kwargs_has_attention_mask: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        decoder_start_token_tensor = _tensor_or_none(generation_config.decoder_start_token_id, device=device)

        if self.config.is_encoder_decoder:
            decoder_start_token_tensor = (
                decoder_start_token_tensor if decoder_start_token_tensor is not None else bos_token_tensor
            )

        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]

        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._decoder_start_token_tensor = decoder_start_token_tensor

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        has_default_min_length,
        model_input_name,
        input_ids_length,
        inputs_tensor,
    ):

        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif (
            model_input_name == "inputs_embeds"
            and input_ids_length != inputs_tensor.shape[1]
            and not self.config.is_encoder_decoder
        ):
            generation_config.max_length -= inputs_tensor.shape[1]
        elif has_default_max_length:
            if generation_config.max_length == GenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        if generation_config.min_new_tokens is not None:
            generation_config.min_length = generation_config.min_new_tokens + input_ids_length

        elif (
            model_input_name == "inputs_embeds"
            and input_ids_length != inputs_tensor.shape[1]
            and not self.config.is_encoder_decoder
        ):
            generation_config.min_length = max(generation_config.min_length - inputs_tensor.shape[1], 0)

        return generation_config

    def _supports_logits_to_keep(self) -> bool:
        return "logits_to_keep" in set(inspect.signature(self.forward).parameters.keys())

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"

        if generation_config.min_new_tokens is not None:
            min_length = generation_config.min_new_tokens + input_ids_length

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
        device: torch.device,
    ) -> bool:
        is_hybrid_cache = any(class_name in self.__class__.__name__.lower() for class_name in ["mamba", "falconh1"])
        cache_name = "past_key_values" if not is_hybrid_cache else "cache_params"

        requires_cross_attention_cache = (
            self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
        )

        user_defined_cache = model_kwargs.get(cache_name)
        if user_defined_cache is not None:
            if isinstance(user_defined_cache, tuple) and self._supports_default_dynamic_cache():
                model_kwargs[cache_name] = (
                    DynamicCache.from_legacy_cache(user_defined_cache)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(user_defined_cache)
                )
            return

        if generation_config.use_cache is False:
            return

        if not self._supports_default_dynamic_cache():
            return

        if assistant_model is not None and generation_config.cache_implementation is not None:
            generation_config.cache_implementation = None

        generation_config.cache_implementation = generation_config.cache_implementation or getattr(
            self.config.get_text_config(), "cache_implementation", None
        )
        if generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                model_kwargs[cache_name] = self._get_cache(
                    cache_implementation=generation_config.cache_implementation,
                    batch_size=max(generation_config.num_beams, generation_config.num_return_sequences) * batch_size,
                    max_cache_len=max_cache_length,
                    device=device,
                    model_kwargs=model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                cache_config = (
                    generation_config.cache_config
                    if generation_config.cache_config is not None
                    else QuantizedCacheConfig()
                )
                cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]

                model_kwargs[cache_name] = cache_class(cache_config)
            elif generation_config.cache_implementation == "offloaded":
                model_kwargs[cache_name] = OffloadedCache()
            elif generation_config.cache_implementation == "dynamic":
                model_kwargs[cache_name] = DynamicCache()

        else:
            model_kwargs[cache_name] = (
                DynamicCache()
                if not requires_cross_attention_cache
                else EncoderDecoderCache(DynamicCache(), DynamicCache())
            )

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: Optional[int] = None,
        encoder_input_ids: torch.LongTensor = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        device: Optional[str] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorList:
        processors = LogitsProcessorList()
        if logits_processor is None:
            logits_processor = []

        if generation_config.guidance_scale is not None and generation_config.guidance_scale != 1:
            processors.append(
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    generation_config.guidance_scale,
                    self,
                    unconditional_ids=negative_prompt_ids,
                    unconditional_attention_mask=negative_prompt_attention_mask,
                    use_cache=generation_config.use_cache,
                )
            )
        if generation_config.sequence_bias is not None:
            processors.append(SequenceBiasLogitsProcessor(sequence_bias=generation_config.sequence_bias))

        if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=generation_config.diversity_penalty,
                    num_beams=generation_config.num_beams,
                    num_beam_groups=generation_config.num_beam_groups,
                )
            )
        if (
            generation_config.encoder_repetition_penalty is not None
            and generation_config.encoder_repetition_penalty != 1.0
        ):
            if len(encoder_input_ids.shape) == 2:
                processors.append(
                    EncoderRepetitionPenaltyLogitsProcessor(
                        penalty=generation_config.encoder_repetition_penalty,
                        encoder_input_ids=encoder_input_ids,
                    )
                )
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
        if (
            generation_config.encoder_no_repeat_ngram_size is not None
            and generation_config.encoder_no_repeat_ngram_size > 0
        ):
            if len(encoder_input_ids.shape) == 2:
                processors.append(
                    EncoderNoRepeatNGramLogitsProcessor(
                        generation_config.encoder_no_repeat_ngram_size,
                        encoder_input_ids,
                    )
                )
        if generation_config.bad_words_ids is not None:
            processors.append(
                NoBadWordsLogitsProcessor(
                    generation_config.bad_words_ids,
                    generation_config._eos_token_tensor,
                )
            )
        if (
            generation_config.min_length is not None
            and getattr(generation_config, "_eos_token_tensor", None) is not None
            and generation_config.min_length > 0
        ):
            processors.append(
                MinLengthLogitsProcessor(
                    generation_config.min_length,
                    generation_config._eos_token_tensor,
                    device=device,
                )
            )
        if (
            generation_config.min_new_tokens is not None
            and getattr(generation_config, "_eos_token_tensor", None) is not None
            and generation_config.min_new_tokens > 0
        ):
            processors.append(
                MinNewTokensLengthLogitsProcessor(
                    input_ids_seq_length,
                    generation_config.min_new_tokens,
                    generation_config._eos_token_tensor,
                    device=device,
                )
            )
        if prefix_allowed_tokens_fn is not None:
            processors.append(
                PrefixConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn,
                    generation_config.num_beams // generation_config.num_beam_groups,
                )
            )
        if generation_config.forced_bos_token_id is not None:
            processors.append(
                ForcedBOSTokenLogitsProcessor(
                    generation_config.forced_bos_token_id,
                )
            )
        if generation_config.forced_eos_token_id is not None:
            processors.append(
                ForcedEOSTokenLogitsProcessor(
                    generation_config.max_length,
                    generation_config.forced_eos_token_id,
                    device=device,
                )
            )
        if generation_config.remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())
        if generation_config.exponential_decay_length_penalty is not None:
            processors.append(
                ExponentialDecayLengthPenalty(
                    generation_config.exponential_decay_length_penalty,
                    generation_config._eos_token_tensor,
                    input_ids_seq_length,
                )
            )
        if generation_config.suppress_tokens is not None:
            processors.append(
                SuppressTokensLogitsProcessor(
                    generation_config.suppress_tokens,
                    device=device,
                )
            )
        if generation_config.begin_suppress_tokens is not None:
            begin_index = input_ids_seq_length
            begin_index = (
                begin_index
                if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
                else begin_index + 1
            )
            processors.append(
                SuppressTokensAtBeginLogitsProcessor(
                    generation_config.begin_suppress_tokens,
                    begin_index,
                    device=device,
                )
            )

        processors = self._merge_criteria_processor_list(processors, logits_processor)

        if generation_config.do_sample:
            if generation_config.num_beams > 1:
                if isinstance(generation_config._eos_token_tensor, list):
                    min_tokens_to_keep = len(generation_config._eos_token_tensor) + 1
                elif isinstance(generation_config._eos_token_tensor, torch.Tensor):
                    min_tokens_to_keep = generation_config._eos_token_tensor.shape[0] + 1
                else:
                    min_tokens_to_keep = 2
            else:
                min_tokens_to_keep = 1

            if generation_config.temperature is not None and generation_config.temperature != 1.0:
                processors.append(TemperatureLogitsWarper(generation_config.temperature))
            if generation_config.top_k is not None and generation_config.top_k != 0:
                processors.append(
                    TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep)
                )
            if generation_config.top_p is not None and generation_config.top_p < 1.0:
                processors.append(
                    TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep)
                )
            if generation_config.min_p is not None:
                processors.append(
                    MinPLogitsWarper(min_p=generation_config.min_p, min_tokens_to_keep=min_tokens_to_keep)
                )
            if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
                processors.append(
                    TypicalLogitsWarper(mass=generation_config.typical_p, min_tokens_to_keep=min_tokens_to_keep)
                )
            if generation_config.epsilon_cutoff is not None and 0.0 < generation_config.epsilon_cutoff < 1.0:
                processors.append(
                    EpsilonLogitsWarper(
                        epsilon=generation_config.epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep
                    )
                )
            if generation_config.eta_cutoff is not None and 0.0 < generation_config.eta_cutoff < 1.0:
                processors.append(
                    EtaLogitsWarper(
                        epsilon=generation_config.eta_cutoff, min_tokens_to_keep=min_tokens_to_keep, device=device
                    )
                )

        if generation_config.watermarking_config is not None:
            processors.append(
                generation_config.watermarking_config.construct_processor(
                    self.config.get_text_config().vocab_size, device
                )
            )

        if generation_config.renormalize_logits is True:
            processors.append(LogitNormalization())
        return processors

    def _merge_criteria_processor_list(
        self,
        default_list: Union[LogitsProcessorList, StoppingCriteriaList],
        custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
    ) -> Union[LogitsProcessorList, StoppingCriteriaList]:
        if len(custom_list) == 0:
            return default_list

        final_list = type(default_list)()
        for default in default_list:
            using_custom = False
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "stopping criteria" if isinstance(custom, ABC) else "logits processor"
                    final_list.append(custom)
                    using_custom = True
                    break
            if not using_custom:
                final_list.append(default)

        for custom in custom_list:
            if custom not in final_list:
                final_list.append(custom)
        return final_list

    def _get_stopping_criteria(
        self,
        generation_config: GenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList],
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **kwargs,
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
        if generation_config.stop_strings is not None:
            criteria.append(StopStringCriteria(stop_strings=generation_config.stop_strings, tokenizer=tokenizer))
        if generation_config._eos_token_tensor is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config._eos_token_tensor))
        if (
            generation_config.is_assistant
            and generation_config.assistant_confidence_threshold is not None
            and generation_config.assistant_confidence_threshold > 0
        ):
            criteria.append(
                ConfidenceCriteria(assistant_confidence_threshold=generation_config.assistant_confidence_threshold)
            )
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        if expand_size == 1:
            return input_ids, model_kwargs

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            if self.config._attn_implementation == "flash_attention_2" and getattr(
                model_kwargs.get("past_key_values"), "is_compileable", False
            ):
                if generation_config.compile_config is None:
                    generation_config.compile_config = CompileConfig(fullgraph=False)
                elif generation_config.compile_config.fullgraph:
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            next_token_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def _get_initial_cache_position(self, seq_length, device, model_kwargs):
        if "cache_position" in model_kwargs and model_kwargs["cache_position"]:
            return model_kwargs
        if "inputs_embeds" in model_kwargs and not self.config.is_encoder_decoder:
            cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
        elif "decoder_inputs_embeds" in model_kwargs and self.config.is_encoder_decoder:
            cache_position = (
                torch.ones_like(model_kwargs["decoder_inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
            )
        else:
            cache_position = torch.ones(seq_length, dtype=torch.int64, device=device).cumsum(0) - 1

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()

            cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    def _valid_auto_compile_criteria(self, model_kwargs: dict, generation_config: GenerationConfig) -> bool:
        if generation_config.disable_compile:
            return False

        valid_hardware = self.device.type == "cuda" or bool(
            generation_config.compile_config is not None and generation_config.compile_config._compile_all_devices
        )
        using_compilable_cache = (
            isinstance(model_kwargs.get("past_key_values"), Cache) and model_kwargs["past_key_values"].is_compileable
        )
        can_compile = valid_hardware and using_compilable_cache and self._supports_static_cache

        if getattr(self, "hf_quantizer", None) is not None:
            can_compile &= self.hf_quantizer.is_compileable

        if hasattr(self, "hf_device_map"):
            all_model_devices = set(self.hf_device_map.values())
            has_cpu_offload = "cpu" in all_model_devices and len(all_model_devices) > 1
            can_compile &= not has_cpu_offload

            has_disk_offload = "disk" in all_model_devices
            can_compile &= not has_disk_offload

        return can_compile

    def _has_unfinished_sequences(self, this_peer_finished: bool, synced_gpus: bool, device: torch.device) -> bool:
        if synced_gpus:
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0, device=device)
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            if this_peer_finished_flag.item() == 0.0:
                return False
        elif this_peer_finished:
            return False
        return True

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs

class AttentionMaskInterface(GeneralInterface):
    _global_mapping = {
        "sdpa": sdpa_mask,
    }
"""
def _get_frameworks_and_test_func(x):
    framework_to_test = {
        "pt": is_torch_tensor,
        "tf": is_tf_tensor,
        "jax": is_jax_tensor,
        "np": is_numpy_array,
        "mlx": is_mlx_array,
    }
    preferred_framework = infer_framework_from_repr(x)
    frameworks = [] if preferred_framework is None else [preferred_framework]
    if preferred_framework != "np":
        frameworks.append("np")
    frameworks.extend([f for f in framework_to_test if f not in [preferred_framework, "np"]])
    return {f: framework_to_test[f] for f in frameworks}



def is_tensor(x):
    framework_to_test_func = _get_frameworks_and_test_func(x)
    for test_func in framework_to_test_func.values():
        if test_func(x):
            return True

    if is_torch_fx_proxy(x):
        return True

    if is_flax_available():
        from jax.core import Tracer

        if isinstance(x, Tracer):
            return True

    return False
"""
@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None

class AttentionInterface(GeneralInterface):
    _global_mapping = {
        "sdpa": sdpa_attention_forward,
    }
ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()

@contextmanager
def no_init_weights():
    global _init_weights
    old_init_weights = _init_weights

    _init_weights = False

    def _skip_init(*args, **kwargs):
        pass

    for name, init_func in TORCH_INIT_FUNCTIONS.items():
        setattr(torch.nn.init, name, _skip_init)

    try:
        yield
    finally:
        _init_weights = old_init_weights
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, init_func)

def _get_torch_dtype(
    cls,
    torch_dtype: Optional[Union[str, torch.dtype, dict]],
    checkpoint_files: Optional[list[str]],
    config: PretrainedConfig,
    sharded_metadata: Optional[dict],
    state_dict: Optional[dict],
    weights_only: bool,
) -> tuple[PretrainedConfig, Optional[torch.dtype], Optional[torch.dtype]]:
    dtype_orig = None
    is_sharded = sharded_metadata is not None

    if torch_dtype is not None:
        if isinstance(torch_dtype, str):
            if torch_dtype == "auto":
                if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
                    torch_dtype = config.torch_dtype
                else:
                    if is_sharded and "dtype" in sharded_metadata:
                        torch_dtype = sharded_metadata["dtype"]
                    elif state_dict is not None:
                        torch_dtype = get_state_dict_dtype(state_dict)
                    else:
                        state_dict = load_state_dict(
                            checkpoint_files[0], map_location="meta", weights_only=weights_only
                        )
                        torch_dtype = get_state_dict_dtype(state_dict)
            elif hasattr(torch, torch_dtype):
                torch_dtype = getattr(torch, torch_dtype)
                config.torch_dtype = torch_dtype
                for sub_config_key in config.sub_configs.keys():
                    sub_config = getattr(config, sub_config_key)
                    sub_config.torch_dtype = torch_dtype
        elif isinstance(torch_dtype, torch.dtype):
            config.torch_dtype = torch_dtype
            for sub_config_key in config.sub_configs.keys():
                sub_config = getattr(config, sub_config_key)
                sub_config.torch_dtype = torch_dtype
        elif isinstance(torch_dtype, dict):
            for key, curr_dtype in torch_dtype.items():
                if hasattr(config, key):
                    value = getattr(config, key)
                    curr_dtype = curr_dtype if not isinstance(curr_dtype, str) else getattr(torch, curr_dtype)
                    value.torch_dtype = curr_dtype
            torch_dtype = torch_dtype.get("")
            torch_dtype = torch_dtype if not isinstance(torch_dtype, str) else getattr(torch, torch_dtype)
            config.torch_dtype = torch_dtype
            if torch_dtype is None:
                torch_dtype = torch.float32

        dtype_orig = cls._set_default_torch_dtype(torch_dtype)
    else:
        default_dtype = torch.get_default_dtype()
        config.torch_dtype = default_dtype
        for key in config.sub_configs.keys():
            value = getattr(config, key)
            value.torch_dtype = default_dtype

    return config, torch_dtype, dtype_orig


def _get_resolved_checkpoint_files(
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    subfolder: str,
    variant: Optional[str],
    gguf_file: Optional[str],
    from_tf: bool,
    from_flax: bool,
    use_safetensors: bool,
    cache_dir: str,
    force_download: bool,
    proxies: Optional[dict[str, str]],
    local_files_only: bool,
    token: Optional[Union[str, bool]],
    user_agent: dict,
    revision: str,
    commit_hash: Optional[str],
    is_remote_code: bool,
    transformers_explicit_filename: Optional[str] = None,
) -> tuple[Optional[list[str]], Optional[dict]]:
    is_sharded = False

    if pretrained_model_name_or_path is not None and gguf_file is None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if is_local:
            if transformers_explicit_filename is not None:
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, transformers_explicit_filename)
                is_sharded = transformers_explicit_filename.endswith(".safetensors.index.json")
            elif from_tf and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
            ):
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
            elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)):
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
            elif from_flax and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
            ):
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
            ):
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant)
                )
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
            ):
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            elif not use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
            ):
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
                )
            elif not use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
            ):
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
        elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            archive_file = pretrained_model_name_or_path
            is_local = True
        elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path + ".index")):
            archive_file = os.path.join(subfolder, pretrained_model_name_or_path + ".index")
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            filename = pretrained_model_name_or_path
            resolved_archive_file = download_url(pretrained_model_name_or_path)
        else:
            if transformers_explicit_filename is not None:
                filename = transformers_explicit_filename
                is_sharded = transformers_explicit_filename.endswith(".safetensors.index.json")
            elif from_tf:
                filename = TF2_WEIGHTS_NAME
            elif from_flax:
                filename = FLAX_WEIGHTS_NAME
            elif use_safetensors is not False:
                filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
            else:
                filename = _add_variant(WEIGHTS_NAME, variant)

            try:
                cached_file_kwargs = {
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                    "proxies": proxies,
                    "local_files_only": local_files_only,
                    "token": token,
                    "user_agent": user_agent,
                    "revision": revision,
                    "subfolder": subfolder,
                    "_raise_exceptions_for_gated_repo": False,
                    "_raise_exceptions_for_missing_entries": False,
                    "_commit_hash": commit_hash,
                }
                resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                    elif use_safetensors:
                        if revision == "main":
                            resolved_archive_file, revision, is_sharded = auto_conversion(
                                pretrained_model_name_or_path, **cached_file_kwargs
                            )
                        cached_file_kwargs["revision"] = revision
                    else:
                        filename = _add_variant(WEIGHTS_NAME, variant)
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, filename, **cached_file_kwargs
                        )
                if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                if not local_files_only and not is_offline_mode():
                    if resolved_archive_file is not None:
                        if filename in [WEIGHTS_NAME, WEIGHTS_INDEX_NAME]:
                            safe_weights_name = SAFE_WEIGHTS_INDEX_NAME if is_sharded else SAFE_WEIGHTS_NAME
                            has_file_kwargs = {
                                "revision": revision,
                                "proxies": proxies,
                                "token": token,
                                "cache_dir": cache_dir,
                                "local_files_only": local_files_only,
                            }
                            cached_file_kwargs = {
                                "cache_dir": cache_dir,
                                "force_download": force_download,
                                "local_files_only": local_files_only,
                                "user_agent": user_agent,
                                "subfolder": subfolder,
                                "_raise_exceptions_for_gated_repo": False,
                                "_raise_exceptions_for_missing_entries": False,
                                "_commit_hash": commit_hash,
                                **has_file_kwargs,
                            }
                            if (
                                not has_file(pretrained_model_name_or_path, safe_weights_name, **has_file_kwargs)
                                and not is_remote_code
                            ):
                                Thread(
                                    target=auto_conversion,
                                    args=(pretrained_model_name_or_path,),
                                    kwargs={"ignore_errors_during_conversion": True, **cached_file_kwargs},
                                    name="Thread-auto_conversion",
                                ).start()
                    else:
                        has_file_kwargs = {
                            "revision": revision,
                            "proxies": proxies,
                            "token": token,
                            "cache_dir": cache_dir,
                            "local_files_only": local_files_only,
                        }

            except Exception as e:
                raise

        if is_local:
            resolved_archive_file = archive_file
    elif gguf_file:
        if os.path.isfile(gguf_file):
            resolved_archive_file = gguf_file
        else:
            cached_file_kwargs = {
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "local_files_only": local_files_only,
                "token": token,
                "user_agent": user_agent,
                "revision": revision,
                "subfolder": subfolder,
                "_raise_exceptions_for_gated_repo": False,
                "_raise_exceptions_for_missing_entries": False,
                "_commit_hash": commit_hash,
            }

            resolved_archive_file = cached_file(pretrained_model_name_or_path, gguf_file, **cached_file_kwargs)

    sharded_metadata = None
    if is_sharded:
        checkpoint_files, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            user_agent=user_agent,
            revision=revision,
            subfolder=subfolder,
            _commit_hash=commit_hash,
        )
    else:
        checkpoint_files = [resolved_archive_file] if pretrained_model_name_or_path is not None else None

    return checkpoint_files, sharded_metadata

class DirectionalMaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, threshold=0.3):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.threshold = threshold

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input @ self.weight.T
        if self.bias is not None:
            output += self.bias
        return output

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Qwen2Config(PretrainedConfig):
    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )

    attn_weights = None

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class CustomQwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            is_causal=True,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class CustomQwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class CustomQwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CustomQwen2Attention(config=config, layer_idx=layer_idx)

        self.mlp = CustomQwen2MLP(config)
        self.input_layernorm = CustomQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CustomQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen2DecoderLayer,
        "attentions": CustomQwen2Attention,
    }

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, CustomQwen2RMSNorm):
            module.weight.data.fill_(1.0)

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class CustomQwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = CustomQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):

            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }

            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class CustomQwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs: dict
    ) -> tuple[GenerationConfig, dict]:
        using_model_generation_config = False
        if generation_config is None:
            if (
                self.generation_config._from_model_config
                and self.generation_config._original_object_hash == hash(self.generation_config)
                and len(self.config._get_non_default_generation_parameters()) > 0
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    self.generation_config = new_generation_config

            generation_config = self.generation_config
            using_model_generation_config = True

        generation_config = copy.deepcopy(generation_config)

        if not using_model_generation_config:
            model_base_version = version.parse(version.parse(self.generation_config.transformers_version).base_version)
            if use_model_defaults is True or (
                use_model_defaults is None and model_base_version >= version.parse("4.50.0")
            ):
                modified_values = {}
                global_default_generation_config = GenerationConfig()
                model_generation_config = self.generation_config
                for key, model_gen_config_value in model_generation_config.__dict__.items():
                    if key.startswith("_") or key == "transformers_version":
                        continue
                    global_default_value = getattr(global_default_generation_config, key, None)
                    custom_gen_config_value = getattr(generation_config, key, None)
                    if (
                        custom_gen_config_value == global_default_value
                        and model_gen_config_value != global_default_value
                    ):
                        modified_values[key] = model_gen_config_value
                        setattr(generation_config, key, model_gen_config_value)
                if generation_config.temperature == 0.0:
                    generation_config.do_sample = False
            else:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.decoder_start_token_id is None:
                    generation_config.decoder_start_token_id = self.generation_config.decoder_start_token_id

        model_kwargs = generation_config.update(**kwargs)

        return generation_config, model_kwargs


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        if custom_generate is not None:
            global_keys_to_exclude = {
                "self",
                "kwargs",
                "global_keys_to_exclude",
                "trust_remote_code",
                "custom_generate",
            }
            generate_arguments = {key: value for key, value in locals().items() if key not in global_keys_to_exclude}
            generate_arguments.update(kwargs)

            custom_generate_function = self.load_custom_generate(
                custom_generate, trust_remote_code=trust_remote_code, **kwargs
            )
            return custom_generate_function(model=self, **generate_arguments)

        tokenizer = kwargs.pop("tokenizer", None)
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
        )

        generation_mode = "sample"

        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        model_kwargs["use_cache"] = generation_config.use_cache

        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        result = self._sample(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

        if (
            generation_config.return_legacy_cache is True
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()
        return result