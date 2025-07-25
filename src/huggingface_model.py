import json
from safetensors.torch import load_file

class huggingface_model:

    def load_config(self, path: str) -> None:
        with open(path, "r", encoding = "utf-8") as f:
            self.config_data.update(json.load(f))

    def check(self) -> bool:
        for key, value in self.config_data.items():
            if value is None:
                print(f"not is ", key)
                return False
        return True

    def model_structure(self, path: str) -> None:
        self.state_dict = load_file(path)

    def __init__(self, path: str):
        self.config_data = {
            "_name_or_path": None,
            "architectures": None,
            "attention_dropout": None,
            "bos_token_id": None,
            "eos_token_id": None,
            "hidden_act": None,
            "hidden_size": None,
            "initializer_range": None,
            "intermediate_size": None,
            "max_position_embeddings": None,
            "max_window_layers": None,
            "model_type": None,
            "num_attention_heads": None,
            "num_hidden_layers": None,
            "num_key_value_heads": None,
            "rms_norm_eps": None,
            "rope_theta": None,
            "sliding_window": None,
            "tie_word_embeddings": None,
            "torch_dtype": None,
            "transformers_version": None,
            "use_cache": None,
            "use_sliding_window": None,
            "vocab_size": None
        }

        self.state_dict = None

        #configをロード
        self.load_config(path)

        #checkを行う
        che = self.check()

        if che is True:
            #構造と重みの読み込みをする
            self.model_structure(path)
        else:
            print("jsonが有効ではありませんでした。")

        print("処理が終了致しました。")