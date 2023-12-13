"""Embedding adapter model."""

import logging
from typing import Any, List, Optional, Type, cast

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding
from llama_index.utils import infer_torch_device

logger = logging.getLogger(__name__)


class AdapterEmbeddingModel(BaseEmbedding):
    """Adapter for any embedding model.  # 任何嵌入模型的适配器。

    This is a wrapper around any embedding model that adds an adapter layer \
        on top of it.  # 这是任何嵌入模型的包装器，它在其顶部添加了一个适配器层。
    This is useful for finetuning an embedding model on a downstream task.  # 这对于在下游任务上微调嵌入模型很有用。
    The embedding model can be any model - it does not need to expose gradients.  # 嵌入模型可以是任何模型-它不需要暴露梯度。

    Args:
        base_embed_model (BaseEmbedding): Base embedding model.  # 基本嵌入模型。
        adapter_path (str): Path to adapter.  # 适配器路径。
        adapter_cls (Optional[Type[Any]]): Adapter class. Defaults to None, in which \
            case a linear adapter is used.  # 适配器类。默认为None，在这种情况下使用线性适配器。
        transform_query (bool): Whether to transform query embeddings. Defaults to True.  # 是否转换查询嵌入。默认为True。
        device (Optional[str]): Device to use. Defaults to None.  # 要使用的设备。默认为None。
        embed_batch_size (int): Batch size for embedding. Defaults to 10.  # 嵌入的批处理大小。默认为10。
        callback_manager (Optional[CallbackManager]): Callback manager. \
            Defaults to None.  # 回调管理器。默认为None。

    """

    _base_embed_model: BaseEmbedding = PrivateAttr()
    _adapter: Any = PrivateAttr()
    _transform_query: bool = PrivateAttr()
    _device: Optional[str] = PrivateAttr()
    _target_device: Any = PrivateAttr()

    def __init__(
        self,
        base_embed_model: BaseEmbedding,
        adapter_path: str,
        adapter_cls: Optional[Type[Any]] = None,
        transform_query: bool = True,
        device: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Init params."""  # 初始化参数。
        import torch

        from llama_index.embeddings.adapter_utils import BaseAdapter, LinearLayer

        if device is None:
            device = infer_torch_device()
            logger.info(f"Use pytorch device: {device}")
        self._target_device = torch.device(device)

        self._base_embed_model = base_embed_model

        if adapter_cls is None:
            adapter_cls = LinearLayer
        else:
            adapter_cls = cast(Type[BaseAdapter], adapter_cls)

        adapter = adapter_cls.load(adapter_path)
        self._adapter = cast(BaseAdapter, adapter)
        self._adapter.to(self._target_device)

        self._transform_query = transform_query
        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=f"Adapter for {base_embed_model.model_name}",
        )

    @classmethod
    def class_name(cls) -> str:
        return "AdapterEmbeddingModel"  # 适配器嵌入模型

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""  # 获取查询嵌入。
        import torch

        query_embedding = self._base_embed_model._get_query_embedding(query)
        if self._transform_query:
            query_embedding_t = torch.tensor(query_embedding).to(self._target_device)
            query_embedding_t = self._adapter.forward(query_embedding_t)
            query_embedding = query_embedding_t.tolist()

        return query_embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""  # 获取查询嵌入。
        import torch

        query_embedding = await self._base_embed_model._aget_query_embedding(query)
        if self._transform_query:
            query_embedding_t = torch.tensor(query_embedding).to(self._target_device)
            query_embedding_t = self._adapter.forward(query_embedding_t)
            query_embedding = query_embedding_t.tolist()

        return query_embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._base_embed_model._get_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._base_embed_model._aget_text_embedding(text)


# Maintain for backwards compatibility  # 保持向后兼容
LinearAdapterEmbeddingModel = AdapterEmbeddingModel
