from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..base import BaseProvider
from .apis.base_api import BaseGeminiAPIModule
from .apis.generate_content import GenerateContentAPI
from .apis.generate_videos import GenerateVideosAPI


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self) -> None:
        self._modules: List[BaseGeminiAPIModule] = []

    @classmethod
    def is_available(cls) -> bool:
        # Available if any sub-module is available
        try:
            return GenerateContentAPI.is_available() or GenerateVideosAPI.is_available()
        except Exception:
            return False

    def install(self, collector: Any) -> Optional[Callable[[], None]]:
        unpatchers: List[Callable[[], None]] = []

        # Build module list (extensible: append additional APIs here)
        modules: List[BaseGeminiAPIModule] = []
        if GenerateContentAPI.is_available():
            modules.append(GenerateContentAPI())
        if GenerateVideosAPI.is_available():
            modules.append(GenerateVideosAPI())

        for mod in modules:
            try:
                up = mod.install(collector)
                if up:
                    unpatchers.append(up)
                self._modules.append(mod)
            except Exception:
                continue

        def unpatch_all() -> None:
            for up in reversed(unpatchers):
                try:
                    up()
                except Exception:
                    pass

        return unpatch_all if unpatchers else None

