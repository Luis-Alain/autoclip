"""Gemma on the Apple Silicon GPU, through MLX.

The weights stay on this machine. Highlight scoring sends text only — a
proposed moment, not the video — and the model is loaded once per run, because
a 12B 8-bit checkpoint is about 13 GB and must not be read back for every moment.
"""

from __future__ import annotations

import importlib.util
import logging
import platform
import re
import sys

from .base import DetectionConfig, LLMProvider, ProviderError, ProviderStatus

log = logging.getLogger(__name__)

#: The id the raw pipeline asks for. It is the MLX 8-bit build of Gemma 4 12B
#: published for Apple Silicon; override it with --model or providers.mlx.model.
DEFAULT_MODEL = "lmstudio-community/gemma-4-12B-it-MLX-8bit"

#: A verdict is a small JSON object. Extra room covers a short chain of thought
#: when the model ignores the request to skip it.
MAX_TOKENS = 800

_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class MlxProvider(LLMProvider):
    name = "mlx"
    requires_key = False

    def __init__(self, model: str = "", *, api_key: str | None = None, base_url: str | None = None):
        super().__init__(model or DEFAULT_MODEL, api_key=api_key, base_url=base_url)
        self._bundle: tuple[object, object, object] | None = None

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        ok, detail = runtime_status()
        if not ok:
            raise ProviderError(detail, provider=self.name)
        import asyncio

        return await asyncio.to_thread(self._generate, system, user, config.temperature)

    async def health_check(self) -> ProviderStatus:
        ok, detail = runtime_status()
        if not ok:
            return ProviderStatus(
                name=self.name, available=False, detail=detail, models=[DEFAULT_MODEL]
            )
        return ProviderStatus(
            name=self.name,
            available=True,
            detail=self.model,
            models=[DEFAULT_MODEL],
        )

    def _generate(self, system: str, user: str, temperature: float) -> str:
        model, processor, model_config = self._load()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = _format_prompt(apply_chat_template, processor, model_config, messages)
        try:
            result = generate(
                model,
                processor,
                prompt,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                verbose=False,
            )
        except Exception as exc:
            raise ProviderError(
                f"MLX generation failed: {exc}",
                provider=self.name,
                hint=(
                    f"The model id is '{self.model}'. Gemma 4 MLX builds need mlx-vlm 0.6 or "
                    "newer, and the 8-bit 12B weights need about 13 GB of unified memory."
                ),
            ) from exc

        text = getattr(result, "text", result)
        if not isinstance(text, str):
            text = str(text)
        return _THINKING.sub("", text).strip()

    def _load(self) -> tuple[object, object, object]:
        if self._bundle is not None:
            return self._bundle
        log.info("Loading %s onto the Apple GPU. The first run downloads the weights.", self.model)
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError as exc:  # pragma: no cover - runtime_status checks this first
            raise ProviderError(
                "mlx-vlm is not installed.",
                provider=self.name,
                hint="On an Apple Silicon Mac it installs with the project dependencies.",
            ) from exc
        try:
            model, processor = load(self.model)
            model_config = load_config(self.model)
        except Exception as exc:
            raise ProviderError(
                f"Could not load '{self.model}'.",
                provider=self.name,
                hint=(
                    "Check the Hugging Face id, the network, and that this Mac has about 13 GB "
                    "free for an 8-bit 12B model. Override the id with --model."
                ),
            ) from exc
        self._bundle = (model, processor, model_config)
        return self._bundle


def runtime_status() -> tuple[bool, str]:
    """Whether this machine can score with MLX. Does not load the weights."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False, (
            "The MLX provider runs on Apple Silicon only. "
            "Use --provider openai or --provider ollama on this machine."
        )
    if importlib.util.find_spec("mlx_vlm") is None:
        return False, (
            "mlx-vlm is not installed, so Gemma cannot run on the GPU. "
            "From the project directory run `uv sync` — on an Apple Silicon Mac that "
            "installs mlx-vlm."
        )
    return True, ""


def _format_prompt(apply_chat_template, processor, model_config, messages: list[dict]) -> str:
    """Chat-template the verdict request. Skip Gemma's thinking mode when the call allows it."""
    try:
        return apply_chat_template(
            processor, model_config, messages, num_images=0, enable_thinking=False
        )
    except TypeError:
        return apply_chat_template(processor, model_config, messages, num_images=0)
