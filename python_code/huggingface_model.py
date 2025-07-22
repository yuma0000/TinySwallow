_hf_deepspeed_config_weak_ref = None

def get_module_from_name(module, tensor_name: str) -> tuple[Any, str]:
    if "." in tensor_name:
        module_name, tensor_name = tensor_name.rsplit(".", 1)
        module = module.get_submodule(module_name)
    return module, tensor_name

class AutoHfQuantizer:
    @classmethod
    def from_config(cls, quantization_config: Union[dict], **kwargs):
        if isinstance(quantization_config, dict):
            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        quant_method = quantization_config.quant_method

        if quant_method == QuantizationMethod.BITS_AND_BYTES:
            if quantization_config.load_in_8bit:
                quant_method += "_8bit"
            else:
                quant_method += "_4bit"

        target_cls = AUTO_QUANTIZER_MAPPING[quant_method]
        return target_cls(quantization_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        quantization_config = AutoQuantizationConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_config(quantization_config)

    @classmethod
    def merge_quantization_configs(
        cls,
        quantization_config: Union[dict],
        quantization_config_from_args,
    ):
        if quantization_config_from_args is not None:
            warning_msg = (
                "You passed `quantization_config` or equivalent parameters to `from_pretrained` but the model you're loading"
                " already has a `quantization_config` attribute. The `quantization_config` from the model will be used."
            )
        else:
            warning_msg = ""

        if isinstance(quantization_config, dict):
            if isinstance(quantization_config_from_args, AutoRoundConfig):
                quantization_config = AutoRoundConfig.from_dict(quantization_config)
            else:
                quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        if (
            isinstance(
                quantization_config, (GPTQConfig, AwqConfig, AutoRoundConfig, FbgemmFp8Config, CompressedTensorsConfig)
            )
            and quantization_config_from_args is not None
        ):
            loading_attr_dict = quantization_config_from_args.get_loading_attributes()
            for attr, val in loading_attr_dict.items():
                setattr(quantization_config, attr, val)

            warning_msg += f"However, loading attributes (e.g. {list(loading_attr_dict.keys())}) will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored."

        if warning_msg != "":
            warnings.warn(warning_msg)

        return quantization_config

    @staticmethod
    def supports_quant_method(quantization_config_dict):
        quant_method = quantization_config_dict.get("quant_method", None)
        if quantization_config_dict.get("load_in_8bit", False) or quantization_config_dict.get("load_in_4bit", False):
            suffix = "_4bit" if quantization_config_dict.get("load_in_4bit", False) else "_8bit"
            quant_method = QuantizationMethod.BITS_AND_BYTES + suffix

        if quant_method not in AUTO_QUANTIZATION_CONFIG_MAPPING.keys():
            return False
        return True

class HfQuantizer():
    requires_calibration = False
    required_packages = None
    requires_parameters_quantization = False

    def __init__(self, quantization_config, **kwargs):
        self.quantization_config = quantization_config

        self.modules_to_not_convert = kwargs.pop("modules_to_not_convert", [])
        self.pre_quantized = kwargs.pop("pre_quantized", True)

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        return torch_dtype

    def update_device_map(self, device_map: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        return device_map

    def adjust_target_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        return torch_dtype

    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
        return missing_keys

    def update_unexpected_keys(self, model, unexpected_keys: list[str], prefix: str) -> list[str]:
        return unexpected_keys

    def update_missing_keys_after_loading(self, model, missing_keys: list[str], prefix: str) -> list[str]:
        return missing_keys

    def update_expected_keys(self, model, expected_keys: list[str], loaded_keys: list[str]) -> list[str]:
        return expected_keys

    def adjust_max_memory(self, max_memory: dict[str, Union[int, str]]) -> dict[str, Union[int, str]]:
        return max_memory

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> bool:
        return False

    def create_quantized_param(self, *args, **kwargs) -> "torch.nn.Parameter":
        if not self.requires_parameters_quantization:
            raise AttributeError(
                f"`.create_quantized_param()` method is not supported by quantizer class {self.__class__.__name__}."
            )

    def validate_environment(self, *args, **kwargs):
        return

    def update_tp_plan(self, config):
        "updates the tp plan for the scales"
        return config

    def preprocess_model(self, model: "PreTrainedModel", **kwargs):
        model.is_quantized = True
        model.quantization_method = self.quantization_config.quant_method
        if self.pre_quantized:
            self._convert_model_for_quantization(model)
        return self._process_model_before_weight_loading(model, **kwargs)

    def postprocess_model(self, model: "PreTrainedModel", **kwargs):
        return self._process_model_after_weight_loading(model, **kwargs)

    @staticmethod
    def get_modules_to_not_convert(
        model: "PreTrainedModel",
        skip_modules: Optional[list[str]] = None,
        keep_in_fp32_modules: Optional[list[str]] = None,
        add_default_skips: bool = False,
    ):
        from ..integrations import get_keys_to_not_convert

        if skip_modules is None or add_default_skips:
            modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            modules_to_not_convert = []

        if skip_modules is not None:
            modules_to_not_convert.extend(skip_modules)

        if keep_in_fp32_modules is not None:
            modules_to_not_convert.extend(keep_in_fp32_modules)

        return modules_to_not_convert

    @property
    def is_qat_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return False

    @abstractmethod
    def _process_model_before_weight_loading(self, model, **kwargs): ...

    @abstractmethod
    def _process_model_after_weight_loading(self, model, **kwargs): ...

    @abstractmethod
    def is_serializable(self, safe_serialization=None): ...

    @property
    @abstractmethod
    def is_trainable(self): ...

    def _convert_model_for_quantization(self, model):
        from accelerate import init_empty_weights
        """
        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name in MODULES_TO_PATCH_FOR_QUANTIZATION.keys() and (
                self.quantization_config.quant_method
                in MODULES_TO_PATCH_FOR_QUANTIZATION[module_class_name]["quantization_methods"]
            ):
                with init_empty_weights():
                    parent_module, name = get_module_from_name(model, name)
                    parent_module._modules[name] = MODULES_TO_PATCH_FOR_QUANTIZATION[module_class_name]["module_name"](
                        model.config.get_text_config()
                    )
        """