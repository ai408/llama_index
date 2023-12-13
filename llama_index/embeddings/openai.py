"""OpenAI embeddings file."""  # OpenAI嵌入文件。

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx  # HTTP客户端
from openai import AsyncOpenAI, OpenAI

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks.base import CallbackManager
from llama_index.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding
from llama_index.llms.openai_utils import (
    create_retry_decorator,
    resolve_openai_credentials,
)

embedding_retry_decorator = create_retry_decorator(  # 创建重试装饰器
    max_retries=6,
    random_exponential=True,
    stop_after_delay_seconds=60,
    min_seconds=1,
    max_seconds=20,
)


class OpenAIEmbeddingMode(str, Enum):
    """OpenAI embedding mode."""  # OpenAI嵌入模式。

    SIMILARITY_MODE = "similarity"
    TEXT_SEARCH_MODE = "text_search"


class OpenAIEmbeddingModelType(str, Enum):
    """OpenAI embedding model type."""  # OpenAI嵌入模型类型。

    DAVINCI = "davinci"
    CURIE = "curie"
    BABBAGE = "babbage"
    ADA = "ada"
    TEXT_EMBED_ADA_002 = "text-embedding-ada-002"


class OpenAIEmbeddingModeModel(str, Enum):
    """OpenAI embedding mode model."""  # OpenAI嵌入模式模型。

    # davinci
    TEXT_SIMILARITY_DAVINCI = "text-similarity-davinci-001"
    TEXT_SEARCH_DAVINCI_QUERY = "text-search-davinci-query-001"
    TEXT_SEARCH_DAVINCI_DOC = "text-search-davinci-doc-001"

    # curie
    TEXT_SIMILARITY_CURIE = "text-similarity-curie-001"
    TEXT_SEARCH_CURIE_QUERY = "text-search-curie-query-001"
    TEXT_SEARCH_CURIE_DOC = "text-search-curie-doc-001"

    # babbage
    TEXT_SIMILARITY_BABBAGE = "text-similarity-babbage-001"
    TEXT_SEARCH_BABBAGE_QUERY = "text-search-babbage-query-001"
    TEXT_SEARCH_BABBAGE_DOC = "text-search-babbage-doc-001"

    # ada
    TEXT_SIMILARITY_ADA = "text-similarity-ada-001"
    TEXT_SEARCH_ADA_QUERY = "text-search-ada-query-001"
    TEXT_SEARCH_ADA_DOC = "text-search-ada-doc-001"

    # text-embedding-ada-002
    TEXT_EMBED_ADA_002 = "text-embedding-ada-002"


# convenient shorthand  # 方便的速记
OAEM = OpenAIEmbeddingMode
OAEMT = OpenAIEmbeddingModelType
OAEMM = OpenAIEmbeddingModeModel

EMBED_MAX_TOKEN_LIMIT = 2048  # 嵌入最大令牌限制


_QUERY_MODE_MODEL_DICT = {  # 查询模式模型字典
    (OAEM.SIMILARITY_MODE, "davinci"): OAEMM.TEXT_SIMILARITY_DAVINCI,
    (OAEM.SIMILARITY_MODE, "curie"): OAEMM.TEXT_SIMILARITY_CURIE,
    (OAEM.SIMILARITY_MODE, "babbage"): OAEMM.TEXT_SIMILARITY_BABBAGE,
    (OAEM.SIMILARITY_MODE, "ada"): OAEMM.TEXT_SIMILARITY_ADA,
    (OAEM.SIMILARITY_MODE, "text-embedding-ada-002"): OAEMM.TEXT_EMBED_ADA_002,
    (OAEM.TEXT_SEARCH_MODE, "davinci"): OAEMM.TEXT_SEARCH_DAVINCI_QUERY,
    (OAEM.TEXT_SEARCH_MODE, "curie"): OAEMM.TEXT_SEARCH_CURIE_QUERY,
    (OAEM.TEXT_SEARCH_MODE, "babbage"): OAEMM.TEXT_SEARCH_BABBAGE_QUERY,
    (OAEM.TEXT_SEARCH_MODE, "ada"): OAEMM.TEXT_SEARCH_ADA_QUERY,
    (OAEM.TEXT_SEARCH_MODE, "text-embedding-ada-002"): OAEMM.TEXT_EMBED_ADA_002,
}

_TEXT_MODE_MODEL_DICT = {  # 文本模式模型字典
    (OAEM.SIMILARITY_MODE, "davinci"): OAEMM.TEXT_SIMILARITY_DAVINCI,
    (OAEM.SIMILARITY_MODE, "curie"): OAEMM.TEXT_SIMILARITY_CURIE,
    (OAEM.SIMILARITY_MODE, "babbage"): OAEMM.TEXT_SIMILARITY_BABBAGE,
    (OAEM.SIMILARITY_MODE, "ada"): OAEMM.TEXT_SIMILARITY_ADA,
    (OAEM.SIMILARITY_MODE, "text-embedding-ada-002"): OAEMM.TEXT_EMBED_ADA_002,
    (OAEM.TEXT_SEARCH_MODE, "davinci"): OAEMM.TEXT_SEARCH_DAVINCI_DOC,
    (OAEM.TEXT_SEARCH_MODE, "curie"): OAEMM.TEXT_SEARCH_CURIE_DOC,
    (OAEM.TEXT_SEARCH_MODE, "babbage"): OAEMM.TEXT_SEARCH_BABBAGE_DOC,
    (OAEM.TEXT_SEARCH_MODE, "ada"): OAEMM.TEXT_SEARCH_ADA_DOC,
    (OAEM.TEXT_SEARCH_MODE, "text-embedding-ada-002"): OAEMM.TEXT_EMBED_ADA_002,
}


@embedding_retry_decorator  # 嵌入重试装饰器
def get_embedding(client: OpenAI, text: str, engine: str, **kwargs: Any) -> List[float]:
    """Get embedding.  # 获取嵌入。

    NOTE: Copied from OpenAI's embedding utils:  # 注意：从OpenAI的嵌入工具复制：
    https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py

    Copied here to avoid importing unnecessary dependencies  # 复制到这里以避免导入不必要的依赖项
    like matplotlib, plotly, scipy, sklearn.  # 如matplotlib、plotly、scipy、sklearn。

    """
    text = text.replace("\n", " ")

    return (
        client.embeddings.create(input=[text], model=engine, **kwargs).data[0].embedding
    )


@embedding_retry_decorator  # 嵌入重试装饰器
async def aget_embedding(
    aclient: AsyncOpenAI, text: str, engine: str, **kwargs: Any
) -> List[float]:
    """Asynchronously get embedding.  # 异步获取嵌入。

    NOTE: Copied from OpenAI's embedding utils:  # 注意：从OpenAI的嵌入工具复制：
    https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py

    Copied here to avoid importing unnecessary dependencies  # 复制到这里以避免导入不必要的依赖项
    like matplotlib, plotly, scipy, sklearn.  # 如matplotlib、plotly、scipy、sklearn。

    """
    text = text.replace("\n", " ")

    return (
        (await aclient.embeddings.create(input=[text], model=engine, **kwargs))
        .data[0]
        .embedding
    )


@embedding_retry_decorator  # 嵌入重试装饰器
def get_embeddings(
    client: OpenAI, list_of_text: List[str], engine: str, **kwargs: Any
) -> List[List[float]]:
    """Get embeddings.  # 获取嵌入。

    NOTE: Copied from OpenAI's embedding utils:  # 注意：从OpenAI的嵌入工具复制：
    https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py

    Copied here to avoid importing unnecessary dependencies  # 复制到这里以避免导入不必要的依赖项
    like matplotlib, plotly, scipy, sklearn.  # 如matplotlib、plotly、scipy、sklearn。

    """
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."  # 批处理大小不应大于2048。

    list_of_text = [text.replace("\n", " ") for text in list_of_text]  # 将换行符替换为空格。

    data = client.embeddings.create(input=list_of_text, model=engine, **kwargs).data  # 创建嵌入。
    return [d.embedding for d in data]  # 返回嵌入


@embedding_retry_decorator  # 嵌入重试装饰器
async def aget_embeddings(
    aclient: AsyncOpenAI,
    list_of_text: List[str],
    engine: str,
    **kwargs: Any,
) -> List[List[float]]:
    """Asynchronously get embeddings.  # 异步获取嵌入。

    NOTE: Copied from OpenAI's embedding utils:  # 注意：从OpenAI的嵌入工具复制：
    https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py

    Copied here to avoid importing unnecessary dependencies  # 复制到这里以避免导入不必要的依赖项
    like matplotlib, plotly, scipy, sklearn.   # 如matplotlib、plotly、scipy、sklearn。

    """
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = (
        await aclient.embeddings.create(input=list_of_text, model=engine, **kwargs)
    ).data
    return [d.embedding for d in data]


def get_engine(
    mode: str,
    model: str,
    mode_model_dict: Dict[Tuple[OpenAIEmbeddingMode, str], OpenAIEmbeddingModeModel],
) -> OpenAIEmbeddingModeModel:
    """Get engine."""  # 获取引擎。
    key = (OpenAIEmbeddingMode(mode), OpenAIEmbeddingModelType(model))
    if key not in mode_model_dict:
        raise ValueError(f"Invalid mode, model combination: {key}")  # 无效的模型组合。
    return mode_model_dict[key]


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI class for embeddings.  # OpenAI类的嵌入。

    Args:
        mode (str): Mode for embedding.
            Defaults to OpenAIEmbeddingMode.TEXT_SEARCH_MODE.
            Options are:

            - OpenAIEmbeddingMode.SIMILARITY_MODE
            - OpenAIEmbeddingMode.TEXT_SEARCH_MODE

        model (str): Model for embedding.
            Defaults to OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002.
            Options are:

            - OpenAIEmbeddingModelType.DAVINCI
            - OpenAIEmbeddingModelType.CURIE
            - OpenAIEmbeddingModelType.BABBAGE
            - OpenAIEmbeddingModelType.ADA
            - OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002
    """

    additional_kwargs: Dict[str, Any] = Field(  # 附加的关键字参数
        default_factory=dict, description="Additional kwargs for the OpenAI API."
    )

    api_key: str = Field(description="The OpenAI API key.")  # OpenAI API密钥。
    api_base: str = Field(description="The base URL for OpenAI API.")  # OpenAI API的基本URL。
    api_version: str = Field(description="The version for OpenAI API.")  # OpenAI API的版本。

    max_retries: int = Field(  # 最大重试次数
        default=10, description="Maximum number of retries.", gte=0
    )
    timeout: float = Field(default=60.0, description="Timeout for each request.", gte=0)  # 每个请求的超时时间。
    default_headers: Optional[Dict[str, str]] = Field(  # 默认头
        default=None, description="The default headers for API requests."
    )
    reuse_client: bool = Field(  # 重用客户端
        default=True,
        description=(  # 描述
            "Reuse the OpenAI client between requests. When doing anything with large "
            "volumes of async API calls, setting this to false can improve stability."
        ),
    )

    _query_engine: OpenAIEmbeddingModeModel = PrivateAttr()  # 查询引擎
    _text_engine: OpenAIEmbeddingModeModel = PrivateAttr()  # 文本引擎
    _client: Optional[OpenAI] = PrivateAttr()  # 客户端
    _aclient: Optional[AsyncOpenAI] = PrivateAttr()  # 异步客户端
    _http_client: Optional[httpx.Client] = PrivateAttr()  # HTTP客户端

    def __init__(  # 初始化
        self,
        mode: str = OpenAIEmbeddingMode.TEXT_SEARCH_MODE,  # 模式
        model: str = OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,  # 模型
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,  # 嵌入批处理大小
        additional_kwargs: Optional[Dict[str, Any]] = None,  # 附加的关键字参数
        api_key: Optional[str] = None,  # API密钥
        api_base: Optional[str] = None,  # API的基本URL
        api_version: Optional[str] = None,  # API的版本
        max_retries: int = 10,  # 最大重试次数
        timeout: float = 60.0,  # 超时时间
        reuse_client: bool = True,  # 重用客户端
        callback_manager: Optional[CallbackManager] = None,  # 回调管理器
        default_headers: Optional[Dict[str, str]] = None,  # 默认头
        http_client: Optional[httpx.Client] = None,  # HTTP客户端
        **kwargs: Any,  # 关键字参数
    ) -> None:
        additional_kwargs = additional_kwargs or {}  # 附加的关键字参数

        api_key, api_base, api_version = resolve_openai_credentials(  # 解析OpenAI凭据
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )

        self._query_engine = get_engine(mode, model, _QUERY_MODE_MODEL_DICT)  # 获取引擎
        self._text_engine = get_engine(mode, model, _TEXT_MODE_MODEL_DICT)  # 获取引擎

        if "model_name" in kwargs:  # 模型名称
            model_name = kwargs.pop("model_name")
        else:
            model_name = model

        super().__init__(
            embed_batch_size=embed_batch_size,  # 嵌入批处理大小
            callback_manager=callback_manager,  # 回调管理器
            model_name=model_name,  # 模型名称
            additional_kwargs=additional_kwargs,  # 附加的关键字参数
            api_key=api_key,  # API密钥
            api_base=api_base,  # API的基本URL
            api_version=api_version,  # API的版本
            max_retries=max_retries,  # 最大重试次数
            reuse_client=reuse_client,  # 重用客户端
            timeout=timeout,  # 超时时间
            default_headers=default_headers,  # 默认头
            **kwargs,
        )

        self._client = None  # 客户端
        self._aclient = None  # 异步客户端
        self._http_client = http_client  # HTTP客户端

    def _get_client(self) -> OpenAI:  # 获取客户端
        if not self.reuse_client:  # 重用客户端
            return OpenAI(**self._get_credential_kwargs())  # 客户端

        if self._client is None:  # 客户端
            self._client = OpenAI(**self._get_credential_kwargs())  # 客户端
        return self._client

    def _get_aclient(self) -> AsyncOpenAI:  # 获取异步客户端
        if not self.reuse_client:
            return AsyncOpenAI(**self._get_credential_kwargs())

        if self._aclient is None:
            self._aclient = AsyncOpenAI(**self._get_credential_kwargs())
        return self._aclient

    @classmethod
    def class_name(cls) -> str:
        return "OpenAIEmbedding"

    def _get_credential_kwargs(self) -> Dict[str, Any]:  # 获取凭据关键字参数
        return {
            "api_key": self.api_key,
            "base_url": self.api_base,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "default_headers": self.default_headers,
            "http_client": self._http_client,
        }

    def _get_query_embedding(self, query: str) -> List[float]:  # 获取查询嵌入
        """Get query embedding."""  # 获取查询嵌入
        client = self._get_client()
        return get_embedding(
            client,
            query,
            engine=self._query_engine,
            **self.additional_kwargs,
        )

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""  # _get_query_embedding的异步版本
        aclient = self._get_aclient()
        return await aget_embedding(
            aclient,
            query,
            engine=self._query_engine,
            **self.additional_kwargs,
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""  # 获取文本嵌入
        client = self._get_client()
        return get_embedding(
            client,
            text,
            engine=self._text_engine,
            **self.additional_kwargs,
        )

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""  # 异步获取文本嵌入
        aclient = self._get_aclient()
        return await aget_embedding(
            aclient,
            text,
            engine=self._text_engine,
            **self.additional_kwargs,
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings.  # 获取文本嵌入

        By default, this is a wrapper around _get_text_embedding.  # 默认情况下，这是_get_text_embedding的包装器
        Can be overridden for batch queries.  # 可以覆盖批量查询

        """
        client = self._get_client()
        return get_embeddings(
            client,
            texts,
            engine=self._text_engine,
            **self.additional_kwargs,
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""  # 异步获取文本嵌入
        aclient = self._get_aclient()
        return await aget_embeddings(
            aclient,
            texts,
            engine=self._text_engine,
            **self.additional_kwargs,
        )
