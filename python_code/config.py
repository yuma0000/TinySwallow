class GenerationConfig():
    extra_output_flags = ("output_attentions", "output_hidden_states", "output_scores", "output_logits")

    def __init__(self, **kwargs):
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.min_length = kwargs.pop("min_length", 0)
        self.min_new_tokens = kwargs.pop("min_new_tokens", None)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.max_time = kwargs.pop("max_time", None)
        self.stop_strings = kwargs.pop("stop_strings", None)

        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.penalty_alpha = kwargs.pop("penalty_alpha", None)
        self.dola_layers = kwargs.pop("dola_layers", None)

        self.use_cache = kwargs.pop("use_cache", True)
        self.cache_implementation = kwargs.pop("cache_implementation", None)
        self.cache_config = kwargs.pop("cache_config", None)
        if self.cache_implementation is not None and self.cache_implementation in CACHE_CONFIG_MAPPING:
            cache_config_class = CACHE_CONFIG_MAPPING[self.cache_implementation]
            if isinstance(self.cache_config, dict):
                self.cache_config = cache_config_class.from_dict(self.cache_config)
        self.return_legacy_cache = kwargs.pop("return_legacy_cache", None)
        self.prefill_chunk_size = kwargs.pop("prefill_chunk_size", None)

        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.min_p = kwargs.pop("min_p", None)
        self.typical_p = kwargs.pop("typical_p", 1.0)
        self.epsilon_cutoff = kwargs.pop("epsilon_cutoff", 0.0)
        self.eta_cutoff = kwargs.pop("eta_cutoff", 0.0)
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.force_words_ids = kwargs.pop("force_words_ids", None)
        self.renormalize_logits = kwargs.pop("renormalize_logits", False)
        self.constraints = kwargs.pop("constraints", None)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
        self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
        self.suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)
        self.sequence_bias = kwargs.pop("sequence_bias", None)
        self.token_healing = kwargs.pop("token_healing", False)
        self.guidance_scale = kwargs.pop("guidance_scale", None)
        self.low_memory = kwargs.pop("low_memory", None)
        watermarking_config = kwargs.pop("watermarking_config", None)
        if watermarking_config is None:
            self.watermarking_config = None
        elif isinstance(watermarking_config, BaseWatermarkingConfig):
            self.watermarking_config = watermarking_config
        else:
            self.watermarking_config = WatermarkingConfig.from_dict(watermarking_config)

        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_scores = kwargs.pop("output_scores", False)
        self.output_logits = kwargs.pop("output_logits", None)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)

        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        self.is_assistant = False
        self.num_assistant_tokens = kwargs.pop("num_assistant_tokens", 20)
        self.num_assistant_tokens_schedule = kwargs.pop("num_assistant_tokens_schedule", "constant")
        self.assistant_confidence_threshold = kwargs.pop("assistant_confidence_threshold", 0.4)
        self.prompt_lookup_num_tokens = kwargs.pop("prompt_lookup_num_tokens", None)
        self.max_matching_ngram_size = kwargs.pop("max_matching_ngram_size", None)
        self.assistant_early_exit = kwargs.pop("assistant_early_exit", None)
        self.assistant_lookbehind = kwargs.pop("assistant_lookbehind", 10)
        self.target_lookbehind = kwargs.pop("target_lookbehind", 10)

        self.compile_config = kwargs.pop("compile_config", None)
        self.disable_compile = kwargs.pop("disable_compile", False)

        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        if not self._from_model_config:
            for key, value in kwargs.items():
                setattr(self, key, value)

        self.validate()

    def validate(self, strict=False):
        minor_issues = {}

        if self.cache_config is not None:
            cache_class = CACHE_CONFIG_MAPPING.get(self.cache_implementation)
            if not isinstance(self.cache_config, cache_class):
                self.cache_config = cache_class.from_dict(self.cache_config)
            self.cache_config.validate()
        if self.watermarking_config is not None:
            if not (
                isinstance(self.watermarking_config, WatermarkingConfig)
                or isinstance(self.watermarking_config, SynthIDTextWatermarkingConfig)
            ):
                minor_issues["watermarking_config"] = (
                    "`watermarking_config` as a dict is deprecated and will be removed in v4.54.0. Please construct "
                    "`watermarking_config` object with `WatermarkingConfig` or `SynthIDTextWatermarkingConfig` class."
                )
                self.watermarking_config = WatermarkingConfig.from_dict(self.watermarking_config)
            self.watermarking_config.validate()

        if self.do_sample is False:
            if self.temperature is not None and self.temperature != 1.0:
                minor_issues["temperature"] = greedy_wrong_parameter_msg.format(
                    flag_name="temperature", flag_value=self.temperature
                )
            if self.top_p is not None and self.top_p != 1.0:
                minor_issues["top_p"] = greedy_wrong_parameter_msg.format(flag_name="top_p", flag_value=self.top_p)
            if self.min_p is not None:
                minor_issues["min_p"] = greedy_wrong_parameter_msg.format(flag_name="min_p", flag_value=self.min_p)
            if self.typical_p is not None and self.typical_p != 1.0:
                minor_issues["typical_p"] = greedy_wrong_parameter_msg.format(
                    flag_name="typical_p", flag_value=self.typical_p
                )
            if (
                self.top_k is not None and self.top_k != 50 and self.penalty_alpha is None
            ):
                minor_issues["top_k"] = greedy_wrong_parameter_msg.format(flag_name="top_k", flag_value=self.top_k)
            if self.epsilon_cutoff is not None and self.epsilon_cutoff != 0.0:
                minor_issues["epsilon_cutoff"] = greedy_wrong_parameter_msg.format(
                    flag_name="epsilon_cutoff", flag_value=self.epsilon_cutoff
                )
            if self.eta_cutoff is not None and self.eta_cutoff != 0.0:
                minor_issues["eta_cutoff"] = greedy_wrong_parameter_msg.format(
                    flag_name="eta_cutoff", flag_value=self.eta_cutoff
                )

        if self.num_beams == 1:
            if self.early_stopping is not False:
                minor_issues["early_stopping"] = single_beam_wrong_parameter_msg.format(
                    flag_name="early_stopping", flag_value=self.early_stopping
                )
            if self.num_beam_groups is not None and self.num_beam_groups != 1:
                minor_issues["num_beam_groups"] = single_beam_wrong_parameter_msg.format(
                    flag_name="num_beam_groups", flag_value=self.num_beam_groups
                )
            if self.diversity_penalty is not None and self.diversity_penalty != 0.0:
                minor_issues["diversity_penalty"] = single_beam_wrong_parameter_msg.format(
                    flag_name="diversity_penalty", flag_value=self.diversity_penalty
                )
            if self.length_penalty is not None and self.length_penalty != 1.0:
                minor_issues["length_penalty"] = single_beam_wrong_parameter_msg.format(
                    flag_name="length_penalty", flag_value=self.length_penalty
                )
            if self.constraints is not None:
                minor_issues["constraints"] = single_beam_wrong_parameter_msg.format(
                    flag_name="constraints", flag_value=self.constraints
                )

        if self.use_cache is False:
            for arg_name in ("cache_implementation", "cache_config", "return_legacy_cache"):
                if getattr(self, arg_name) is not None:
                    minor_issues[arg_name] = no_cache_warning.format(
                        cache_arg=arg_name, cache_arg_value=getattr(self, arg_name)
                    )

        generate_arguments = (
            "logits_processor",
            "stopping_criteria",
            "prefix_allowed_tokens_fn",
            "synced_gpus",
            "assistant_model",
            "streamer",
            "negative_prompt_ids",
            "negative_prompt_attention_mask",
            "use_model_defaults",
        )

        if len(minor_issues) > 0:
            attributes_with_issues = list(minor_issues.keys())
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        self.validate(strict=True)

        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            kwargs["token"] = use_auth_token

        config_file_name = config_file_name if config_file_name is not None else GENERATION_CONFIG_NAME

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        output_config_file = os.path.join(save_directory, config_file_name)

        self.to_json_file(output_config_file, use_diff=True)
        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -> "GenerationConfig":
        config_file_name = config_file_name if config_file_name is not None else GENERATION_CONFIG_NAME

        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        if use_auth_token is not None:
            token = use_auth_token

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        config_path = os.path.join(pretrained_model_name, config_file_name)
        config_path = str(config_path)

        is_local = os.path.exists(config_path)
        if os.path.isfile(os.path.join(subfolder, config_path)):
            resolved_config_file = config_path
            is_local = True
        elif is_remote_url(config_path):
            configuration_file = config_path
            resolved_config_file = download_url(config_path)
        else:
            configuration_file = config_file_name
            resolved_config_file = cached_file(
                pretrained_model_name,
                configuration_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=commit_hash,
            )
            commit_hash = extract_commit_hash(resolved_config_file, commit_hash)

        config_dict = cls._dict_from_json_file(resolved_config_file)
        config_dict["_commit_hash"] = commit_hash

        if kwargs.get("return_unused_kwargs") is True:
            config, unused_kwargs = cls.from_dict(config_dict, **kwargs)
            config._original_object_hash = hash(config)
            return config, unused_kwargs
        else:
            config = cls.from_dict(config_dict, **kwargs)
            config._original_object_hash = hash(config)
            return config

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs) -> "GenerationConfig":
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        config = cls(**{**config_dict, **kwargs})
        unused_kwargs = config.update(**kwargs)

        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

    def dict_torch_dtype_to_str(self, d: dict[str, Any]) -> None:
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_torch_dtype_to_str(value)

    def to_diff_dict(self) -> dict[str, Any]:
        config_dict = self.to_dict()

        default_config_dict = GenerationConfig().to_dict()

        serializable_config_dict = {}

        for key, value in config_dict.items():
            if key not in default_config_dict or key == "transformers_version" or value != default_config_dict[key]:
                serializable_config_dict[key] = value

        self.dict_torch_dtype_to_str(serializable_config_dict)
        return serializable_config_dict

    def to_dict(self) -> dict[str, Any]:
        output = copy.deepcopy(self.__dict__)

        if "_commit_hash" in output:
            del output["_commit_hash"]
        if "_original_object_hash" in output:
            del output["_original_object_hash"]
        if "compile_config" in output:
            del output["compile_config"]

        output["transformers_version"] = __version__

        self.dict_torch_dtype_to_str(output)
        return output

    def to_json_string(self, use_diff: bool = True, ignore_metadata: bool = False) -> str:
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        if ignore_metadata:
            for metadata_field in METADATA_FIELDS:
                config_dict.pop(metadata_field, None)

        def convert_keys_to_string(obj):
            if isinstance(obj, dict):
                return {str(key): convert_keys_to_string(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys_to_string(item) for item in obj]
            else:
                return obj

        def convert_dataclass_to_dict(obj):
            if isinstance(obj, dict):
                return {key: convert_dataclass_to_dict(value) for key, value in obj.items()}
            elif is_dataclass(obj):
                return obj.to_dict()
            else:
                return obj

        config_dict = convert_keys_to_string(config_dict)
        config_dict = convert_dataclass_to_dict(config_dict)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    @classmethod
    def from_model_config(cls, model_config: PretrainedConfig) -> "GenerationConfig":
        config_dict = model_config.to_dict()
        config_dict.pop("_from_model_config", None)

        config_dict = {key: value for key, value in config_dict.items() if value is not None}

        generation_config = cls.from_dict(config_dict, return_unused_kwargs=False, _from_model_config=True)

        decoder_config = model_config.get_text_config(decoder=True)
        if decoder_config is not model_config:
            default_generation_config = GenerationConfig()
            decoder_config_dict = decoder_config.to_dict()
            for attr in generation_config.to_dict().keys():
                is_unset = getattr(generation_config, attr) == getattr(default_generation_config, attr)
                if attr in decoder_config_dict and is_unset:
                    setattr(generation_config, attr, decoder_config_dict[attr])
        if generation_config.return_dict_in_generate is False:
            if any(
                getattr(generation_config, extra_output_flag, False)
                for extra_output_flag in generation_config.extra_output_flags
            ):
                generation_config.return_dict_in_generate = True

        generation_config._original_object_hash = hash(generation_config)
        return generation_config

    def update(self, **kwargs):
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        self.validate()

        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs