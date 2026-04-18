from typing import Any

from .config import LocalModelConfig, PipelineConfigError
from .helpers import extract_json_object

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


class LocalChatModel:
    def __init__(self, config: LocalModelConfig):
        if not config.model_path:
            raise PipelineConfigError("Model path is required.")
        self.config = config
        self._tokenizer = None
        self._model = None

    def _load(self):
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            raise PipelineConfigError("transformers and torch are required in the backend environment.")
        if self._tokenizer is not None and self._model is not None:
            return

        model_kwargs = {"trust_remote_code": True}
        if self.config.device == "auto" and torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(self.config.model_path, **model_kwargs)

        if "device_map" not in model_kwargs:
            target_device = "cuda" if self.config.device == "cuda" and torch.cuda.is_available() else "cpu"
            self._model.to(target_device)
        self._model.eval()

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    @property
    def model(self):
        self._load()
        return self._model

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> dict[str, Any]:
        prompt = self._render_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_device = next(self.model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature <= 0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        try:
            payload = extract_json_object(text)
            parse_error = ""
        except Exception as exc:
            payload = {}
            parse_error = str(exc)
        payload["_prompt"] = prompt
        payload["_raw_text"] = text
        if parse_error:
            payload["_parse_error"] = parse_error
        return payload
