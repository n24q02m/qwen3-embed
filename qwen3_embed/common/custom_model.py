"""One-call registration helper for bring-your-own (BYO) ONNX embedding models."""

from dataclasses import dataclass, field

from qwen3_embed.common.model_description import (
    CustomDenseModelDescription,
    ModelSource,
    PoolingType,
)


@dataclass
class CustomModelSpec:
    """One-call registration of a BYO ONNX embedding model.

    Only ONNX-able models with one of the four supported output shapes are
    accepted: bert-bi (``CLS``/``MEAN``), causal last-token (``LAST_TOKEN``), or
    raw 3-D output (``DISABLED``).

    Usage::

        from qwen3_embed import CustomModelSpec, TextEmbedding

        CustomModelSpec(
            model_id="Org/gte-multilingual-base-onnx",
            hf="Org/gte-multilingual-base-onnx",
            model_file="onnx/model.onnx",
            dim=768, pooling="CLS", normalization=True,
        ).register()

        model = TextEmbedding("Org/gte-multilingual-base-onnx")
    """

    model_id: str
    hf: str | None = None
    url: str | None = None
    model_file: str = "onnx/model.onnx"
    dim: int | None = None
    pooling: str | PoolingType = PoolingType.MEAN
    normalization: bool = True
    max_seq_len: int | None = None
    additional_files: list[str] = field(default_factory=list)

    def register(self) -> None:
        """Register this model with :class:`TextEmbedding` so it can be loaded by id."""
        from qwen3_embed import TextEmbedding

        if self.dim is None:
            raise ValueError("dim is required for an embedding model")

        description = CustomDenseModelDescription(
            model=self.model_id,
            dim=self.dim,
            sources=ModelSource(hf=self.hf, url=self.url),
            model_file=self.model_file,
            additional_files=self.additional_files,
        )
        TextEmbedding.add_custom_model(
            description,
            pooling=PoolingType(self.pooling),
            normalization=self.normalization,
        )
