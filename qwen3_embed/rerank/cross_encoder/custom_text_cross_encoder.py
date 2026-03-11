from qwen3_embed.common.model_description import BaseModelDescription
from qwen3_embed.rerank.cross_encoder.onnx_text_cross_encoder import OnnxTextCrossEncoder


class CustomTextCrossEncoder(OnnxTextCrossEncoder):
    SUPPORTED_MODELS: list[BaseModelDescription] = []

    @classmethod
    def _list_supported_models(cls) -> list[BaseModelDescription]:
        return cls.SUPPORTED_MODELS

    @classmethod
    def add_model(
        cls,
        model_description: BaseModelDescription,
    ) -> None:
        cls.SUPPORTED_MODELS.append(model_description)
