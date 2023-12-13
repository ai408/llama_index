"""Base embeddings file."""

import asyncio
from abc import abstractmethod
from typing import Coroutine, List, Tuple

from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.embeddings.base import (
    BaseEmbedding,
    Embedding,
)
from llama_index.schema import ImageType
from llama_index.utils import get_tqdm_iterable


class MultiModalEmbedding(BaseEmbedding):
    """Base class for Multi Modal embeddings."""  # 基类多模嵌入。

    @abstractmethod
    def _get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """
        Embed the input image synchronously.  # 同步嵌入输入图像。

        Subclasses should implement this method. Reference get_image_embedding's
        docstring for more information.  # 子类应该实现这个方法。参考get_image_embedding的文档字符串以获取更多信息。
        """

    @abstractmethod
    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """
        Embed the input image asynchronously.  # 异步嵌入输入图像。

        Subclasses should implement this method. Reference get_image_embedding's
        docstring for more information.  # 子类应该实现这个方法。参考get_image_embedding的文档字符串以获取更多信息。
        """

    def get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """
        Embed the input image.  # 嵌入输入图像。
        """
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            image_embedding = self._get_image_embedding(img_file_path)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [img_file_path],
                    EventPayload.EMBEDDINGS: [image_embedding],
                },
            )
        return image_embedding

    async def aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """Get image embedding."""
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            image_embedding = await self._aget_image_embedding(img_file_path)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [img_file_path],
                    EventPayload.EMBEDDINGS: [image_embedding],
                },
            )
        return image_embedding

    def _get_image_embeddings(self, img_file_paths: List[ImageType]) -> List[Embedding]:
        """
        Embed the input sequence of image synchronously.  # 同步嵌入输入图像序列。

        Subclasses can implement this method if batch queries are supported.  # 如果支持批量查询，子类可以实现这个方法。
        """
        # Default implementation just loops over _get_image_embedding  # 默认实现只是循环遍历_get_image_embedding
        return [
            self._get_image_embedding(img_file_path) for img_file_path in img_file_paths
        ]

    async def _aget_image_embeddings(
        self, img_file_paths: List[ImageType]
    ) -> List[Embedding]:
        """
        Embed the input sequence of image asynchronously.  # 异步嵌入输入图像序列。

        Subclasses can implement this method if batch queries are supported.  # 如果支持批量查询，子类可以实现这个方法。
        """
        return await asyncio.gather(
            *[
                self._aget_image_embedding(img_file_path)
                for img_file_path in img_file_paths
            ]
        )

    def get_image_embedding_batch(
        self, img_file_paths: List[ImageType], show_progress: bool = False
    ) -> List[Embedding]:
        """Get a list of image embeddings, with batching."""  # 获取图像嵌入列表，批处理。
        cur_batch: List[ImageType] = []
        result_embeddings: List[Embedding] = []

        queue_with_progress = enumerate(
            get_tqdm_iterable(
                img_file_paths, show_progress, "Generating image embeddings"
            )
        )

        for idx, img_file_path in queue_with_progress:
            cur_batch.append(img_file_path)
            if (
                idx == len(img_file_paths) - 1
                or len(cur_batch) == self.embed_batch_size
            ):
                # flush
                with self.callback_manager.event(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                ) as event:
                    embeddings = self._get_image_embeddings(cur_batch)
                    result_embeddings.extend(embeddings)
                    event.on_end(
                        payload={
                            EventPayload.CHUNKS: cur_batch,
                            EventPayload.EMBEDDINGS: embeddings,
                        },
                    )
                cur_batch = []

        return result_embeddings

    async def aget_image_embedding_batch(
        self, img_file_paths: List[ImageType], show_progress: bool = False
    ) -> List[Embedding]:
        """Asynchronously get a list of image embeddings, with batching."""  # 异步获取图像嵌入列表，批处理。
        cur_batch: List[ImageType] = []
        callback_payloads: List[Tuple[str, List[ImageType]]] = []
        result_embeddings: List[Embedding] = []
        embeddings_coroutines: List[Coroutine] = []
        for idx, img_file_path in enumerate(img_file_paths):
            cur_batch.append(img_file_path)
            if (
                idx == len(img_file_paths) - 1
                or len(cur_batch) == self.embed_batch_size
            ):
                # flush
                event_id = self.callback_manager.on_event_start(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                )
                callback_payloads.append((event_id, cur_batch))
                embeddings_coroutines.append(self._aget_image_embeddings(cur_batch))
                cur_batch = []

        # flatten the results of asyncio.gather, which is a list of embeddings lists  # 扁平化asyncio.gather的结果，这是一个嵌入列表的列表
        nested_embeddings = []
        if show_progress:
            try:
                from tqdm.auto import tqdm

                nested_embeddings = [
                    await f
                    for f in tqdm(
                        asyncio.as_completed(embeddings_coroutines),
                        total=len(embeddings_coroutines),
                        desc="Generating image embeddings",
                    )
                ]
            except ImportError:
                nested_embeddings = await asyncio.gather(*embeddings_coroutines)
        else:
            nested_embeddings = await asyncio.gather(*embeddings_coroutines)

        result_embeddings = [
            embedding for embeddings in nested_embeddings for embedding in embeddings
        ]

        for (event_id, image_batch), embeddings in zip(
            callback_payloads, nested_embeddings
        ):
            self.callback_manager.on_event_end(
                CBEventType.EMBEDDING,
                payload={
                    EventPayload.CHUNKS: image_batch,
                    EventPayload.EMBEDDINGS: embeddings,
                },
                event_id=event_id,
            )

        return result_embeddings
