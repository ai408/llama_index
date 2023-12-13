"""Base embeddings file."""

import asyncio
from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Coroutine, List, Optional, Tuple

import numpy as np

from llama_index.bridge.pydantic import Field, validator
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.constants import (
    DEFAULT_EMBED_BATCH_SIZE,
)
from llama_index.schema import BaseNode, MetadataMode, TransformComponent
from llama_index.utils import get_tqdm_iterable

# TODO: change to numpy array
Embedding = List[float]


class SimilarityMode(str, Enum):
    """Modes for similarity/distance."""  # 这里的str是指继承的是str类

    DEFAULT = "cosine"  # 这里的DEFAULT是指继承的是str类的DEFAULT属性
    DOT_PRODUCT = "dot_product"  # 这里的DOT_PRODUCT是指继承的是str类的DOT_PRODUCT属性
    EUCLIDEAN = "euclidean"  # 这里的EUCLIDEAN是指继承的是str类的EUCLIDEAN属性


def mean_agg(embeddings: List[Embedding]) -> Embedding:
    """Mean aggregation for embeddings."""  # 这里的Embedding是指上面的Embedding = List[float]
    return list(np.array(embeddings).mean(axis=0))


def similarity(
    embedding1: Embedding,
    embedding2: Embedding,
    mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float:
    """Get embedding similarity."""  # 这里的Embedding是指上面的Embedding = List[float]
    if mode == SimilarityMode.EUCLIDEAN:
        # Using -euclidean distance as similarity to achieve same ranking order
        return -float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
    elif mode == SimilarityMode.DOT_PRODUCT:
        return np.dot(embedding1, embedding2)
    else:
        product = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return product / norm


class BaseEmbedding(TransformComponent):
    """Base class for embeddings."""  # 这里的TransformComponent是指继承的是TransformComponent类

    model_name: str = Field(
        default="unknown", description="The name of the embedding model."
    )
    embed_batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        description="The batch size for embedding calls.",
        gt=0,
        lte=2048,
    )
    callback_manager: CallbackManager = Field(
        default_factory=lambda: CallbackManager([]), exclude=True
    )

    class Config:
        arbitrary_types_allowed = True

    @validator("callback_manager", pre=True)  # 这里的callback_manager是指上面的callback_manager
    def _validate_callback_manager(
        cls, v: Optional[CallbackManager]
    ) -> CallbackManager:
        if v is None:
            return CallbackManager([])
        return v

    @abstractmethod
    def _get_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query synchronously.  # 这里的query是指上面的query

        Subclasses should implement this method. Reference get_query_embedding's
        docstring for more information.  # 这里的get_query_embedding是指上面的get_query_embedding
        """

    @abstractmethod
    async def _aget_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query asynchronously.

        Subclasses should implement this method. Reference get_query_embedding's
        docstring for more information.
        """

    def get_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query.  # 这里的query是指上面的query

        When embedding a query, depending on the model, a special instruction
        can be prepended to the raw query string. For example, "Represent the
        question for retrieving supporting documents: ". If you're curious,
        other examples of predefined instructions can be found in
        embeddings/huggingface_utils.py.
        """
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            query_embedding = self._get_query_embedding(query)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [query],
                    EventPayload.EMBEDDINGS: [query_embedding],
                },
            )
        return query_embedding

    async def aget_query_embedding(self, query: str) -> Embedding:
        """Get query embedding."""  # 获取查询嵌入
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            query_embedding = await self._aget_query_embedding(query)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [query],
                    EventPayload.EMBEDDINGS: [query_embedding],
                },
            )
        return query_embedding

    def get_agg_embedding_from_queries(
        self,
        queries: List[str],
        agg_fn: Optional[Callable[..., Embedding]] = None,
    ) -> Embedding:
        """Get aggregated embedding from multiple queries."""  # 获取多个查询的聚合嵌入
        query_embeddings = [self.get_query_embedding(query) for query in queries]
        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    async def aget_agg_embedding_from_queries(
        self,
        queries: List[str],
        agg_fn: Optional[Callable[..., Embedding]] = None,
    ) -> Embedding:
        """Async get aggregated embedding from multiple queries."""  # 从多个查询中异步获取聚合嵌入
        query_embeddings = [await self.aget_query_embedding(query) for query in queries]
        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    @abstractmethod
    def _get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text synchronously.  # 这里的text是指上面的text

        Subclasses should implement this method. Reference get_text_embedding's
        docstring for more information.  # 这里的get_text_embedding是指上面的get_text_embedding
        """

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text asynchronously.  # 嵌入输入文本异步

        Subclasses can implement this method if there is a true async
        implementation. Reference get_text_embedding's docstring for more
        information.  # 子类可以实现此方法，如果有真正的异步实现。有关更多信息，请参见get_text_embedding的docstring。
        """
        # Default implementation just falls back on _get_text_embedding  # 默认实现只是回退到_get_text_embedding
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Embed the input sequence of text synchronously.  # 嵌入输入的文本序列同步

        Subclasses can implement this method if batch queries are supported.  # 如果支持批量查询，子类可以实现此方法。
        """
        # Default implementation just loops over _get_text_embedding
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Embed the input sequence of text asynchronously.  # 嵌入输入的文本序列异步

        Subclasses can implement this method if batch queries are supported.  # 如果支持批量查询，子类可以实现此方法。
        """
        return await asyncio.gather(
            *[self._aget_text_embedding(text) for text in texts]
        )

    def get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text.  # 嵌入输入文本

        When embedding text, depending on the model, a special instruction  # 嵌入文本时，根据模型，可以在原始文本字符串之前添加特殊指令
        can be prepended to the raw text string. For example, "Represent the  # 例如，“代表
        document for retrieval: ". If you're curious, other examples of  # 检索文档：”。如果你好奇，其他的例子
        predefined instructions can be found in embeddings/huggingface_utils.py. # 预定义的指令可以在embeddings/huggingface_utils.py中找到。
        """
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            text_embedding = self._get_text_embedding(text)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [text],
                    EventPayload.EMBEDDINGS: [text_embedding],
                }
            )

        return text_embedding

    async def aget_text_embedding(self, text: str) -> Embedding:
        """Async get text embedding."""  # 异步获取文本嵌入
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            text_embedding = await self._aget_text_embedding(text)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [text],
                    EventPayload.EMBEDDINGS: [text_embedding],
                }
            )

        return text_embedding

    def get_text_embedding_batch(
        self,
        texts: List[str],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[Embedding]:
        """Get a list of text embeddings, with batching."""  # 获取文本嵌入的列表，批处理
        cur_batch: List[str] = []
        result_embeddings: List[Embedding] = []

        queue_with_progress = enumerate(
            get_tqdm_iterable(texts, show_progress, "Generating embeddings")
        )

        for idx, text in queue_with_progress:
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
                # flush
                with self.callback_manager.event(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                ) as event:
                    embeddings = self._get_text_embeddings(cur_batch)
                    result_embeddings.extend(embeddings)
                    event.on_end(
                        payload={
                            EventPayload.CHUNKS: cur_batch,
                            EventPayload.EMBEDDINGS: embeddings,
                        },
                    )
                cur_batch = []

        return result_embeddings

    async def aget_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False
    ) -> List[Embedding]:
        """Asynchronously get a list of text embeddings, with batching."""  # 异步获取文本嵌入的列表，批处理
        cur_batch: List[str] = []
        callback_payloads: List[Tuple[str, List[str]]] = []
        result_embeddings: List[Embedding] = []
        embeddings_coroutines: List[Coroutine] = []
        for idx, text in enumerate(texts):
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
                # flush
                event_id = self.callback_manager.on_event_start(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                )
                callback_payloads.append((event_id, cur_batch))
                embeddings_coroutines.append(self._aget_text_embeddings(cur_batch))
                cur_batch = []

        # flatten the results of asyncio.gather, which is a list of embeddings lists
        # 平铺asyncio.gather的结果，这是一个嵌入列表的列表
        nested_embeddings = []
        if show_progress:
            try:
                from tqdm.auto import tqdm

                nested_embeddings = [
                    await f
                    for f in tqdm(
                        asyncio.as_completed(embeddings_coroutines),
                        total=len(embeddings_coroutines),
                        desc="Generating embeddings",
                    )
                ]
            except ImportError:
                nested_embeddings = await asyncio.gather(*embeddings_coroutines)
        else:
            nested_embeddings = await asyncio.gather(*embeddings_coroutines)

        result_embeddings = [
            embedding for embeddings in nested_embeddings for embedding in embeddings
        ]

        for (event_id, text_batch), embeddings in zip(
            callback_payloads, nested_embeddings
        ):
            self.callback_manager.on_event_end(
                CBEventType.EMBEDDING,
                payload={
                    EventPayload.CHUNKS: text_batch,
                    EventPayload.EMBEDDINGS: embeddings,
                },
                event_id=event_id,
            )

        return result_embeddings

    def similarity(
        self,
        embedding1: Embedding,
        embedding2: Embedding,
        mode: SimilarityMode = SimilarityMode.DEFAULT,
    ) -> float:
        """Get embedding similarity."""  # 获取嵌入相似性
        return similarity(embedding1=embedding1, embedding2=embedding2, mode=mode)

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        embeddings = self.get_text_embedding_batch(
            [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
            **kwargs,
        )

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        return nodes

    async def acall(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        embeddings = await self.aget_text_embedding_batch(
            [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
            **kwargs,
        )

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        return nodes
