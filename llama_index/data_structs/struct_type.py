"""IndexStructType class."""

from enum import Enum


class IndexStructType(str, Enum):
    """Index struct type. Identifier for a "type" of index.  # 索引结构类型。索引“类型”的标识符。

    Attributes:
        TREE ("tree"): Tree index. See :ref:`Ref-Indices-Tree` for tree indices.  # 树索引。参见:ref:`Ref-Indices-Tree`树索引。
        LIST ("list"): Summary index. See :ref:`Ref-Indices-List` for summary indices.  # 摘要索引。参见:ref:`Ref-Indices-List`摘要索引。
        KEYWORD_TABLE ("keyword_table"): Keyword table index. See
            :ref:`Ref-Indices-Table`
            for keyword table indices.  # 关键字表索引。参见:ref:`Ref-Indices-Table`关键字表索引。
        DICT ("dict"): Faiss Vector Store Index. See
            :ref:`Ref-Indices-VectorStore`
            for more information on the faiss vector store index.  # Faiss向量存储索引。参见:ref:`Ref-Indices-VectorStore`有关faiss向量存储索引的更多信息。
        SIMPLE_DICT ("simple_dict"): Simple Vector Store Index. See
            :ref:`Ref-Indices-VectorStore`  # 简单的向量存储索引。参见:ref:`Ref-Indices-VectorStore`
            for more information on the simple vector store index.
        WEAVIATE ("weaviate"): Weaviate Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Weaviate vector store index.  # Weaviate向量存储索引。参见:ref:`Ref-Indices-VectorStore`有关Weaviate向量存储索引的更多信息。
        PINECONE ("pinecone"): Pinecone Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Pinecone vector store index.  # Pinecone向量存储索引。参见:ref:`Ref-Indices-VectorStore`有关Pinecone向量存储索引的更多信息。
        DEEPLAKE ("deeplake"): DeepLake Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Pinecone vector store index.  # DeepLake向量存储索引。参见:ref:`Ref-Indices-VectorStore`有关Pinecone向量存储索引的更多信息。
        QDRANT ("qdrant"): Qdrant Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Qdrant vector store index.  # Qdrant向量存储索引。参见:ref:`Ref-Indices-VectorStore`有关Qdrant向量存储索引的更多信息。
        LANCEDB ("lancedb"): LanceDB Vector Store Index
            See :ref:`Ref-Indices-VectorStore`
            for more information on the LanceDB vector store index.  # LanceDB向量存储索引。参见:ref:`Ref-Indices-VectorStore`有关LanceDB向量存储索引的更多信息。
        MILVUS ("milvus"): Milvus Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Milvus vector store index.  # Milvus向量存储索引。参见:ref:`Ref-Indices-VectorStore`有关Milvus向量存储索引的更多信息。
        CHROMA ("chroma"): Chroma Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Chroma vector store index.  # Chroma向量存储索引。参见:ref:`Ref-Indices-VectorStore`有关Chroma向量存储索引的更多信息。
        OPENSEARCH ("opensearch"): Opensearch Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Opensearch vector store index.  # Opensearch向量存储索引。参见:ref:`Ref-Indices-VectorStore`有关Opensearch向量存储索引的更多信息。
        MYSCALE ("myscale"): MyScale Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the MyScale vector store index.  # MyScale向量存储索引。参见:ref:`Ref-Indices-VectorStore`有关MyScale向量存储索引的更多信息。
        EPSILLA ("epsilla"): Epsilla Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Epsilla vector store index.  # Epsilla向量存储索引。参见:ref:`Ref-Indices-VectorStore`有关Epsilla向量存储索引的更多信息。
        CHATGPT_RETRIEVAL_PLUGIN ("chatgpt_retrieval_plugin"): ChatGPT
            retrieval plugin index.  # ChatGPT检索插件索引。
        SQL ("SQL"): SQL Structured Store Index.
            See :ref:`Ref-Indices-StructStore`
            for more information on the SQL vector store index.  # SQL结构化存储索引。参见:ref:`Ref-Indices-StructStore`有关SQL向量存储索引的更多信息。
        DASHVECTOR ("dashvector"): DashVector Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Dashvecotor vector store index.  # DashVector向量存储索引。参见:ref:`Ref-Indices-VectorStore`有关Dashvecotor向量存储索引的更多信息。
        KG ("kg"): Knowledge Graph index.
            See :ref:`Ref-Indices-Knowledge-Graph` for KG indices.  # 知识图谱索引。参见:ref:`Ref-Indices-Knowledge-Graph`有关KG索引的更多信息。
        DOCUMENT_SUMMARY ("document_summary"): Document Summary Index.
            See :ref:`Ref-Indices-Document-Summary` for Summary Indices.  # 文档摘要索引。参见:ref:`Ref-Indices-Document-Summary`有关摘要索引的更多信息。

    """

    # TODO: refactor so these are properties on the base class  # 重构，使这些属性在基类上

    NODE = "node"
    TREE = "tree"
    LIST = "list"
    KEYWORD_TABLE = "keyword_table"

    # faiss
    DICT = "dict"
    # simple
    SIMPLE_DICT = "simple_dict"
    WEAVIATE = "weaviate"
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    LANCEDB = "lancedb"
    MILVUS = "milvus"
    CHROMA = "chroma"
    MYSCALE = "myscale"
    VECTOR_STORE = "vector_store"
    OPENSEARCH = "opensearch"
    DASHVECTOR = "dashvector"
    CHATGPT_RETRIEVAL_PLUGIN = "chatgpt_retrieval_plugin"
    DEEPLAKE = "deeplake"
    EPSILLA = "epsilla"
    # multimodal
    MULTIMODAL_VECTOR_STORE = "multimodal"
    # for SQL index
    SQL = "sql"
    # for KG index
    KG = "kg"
    SIMPLE_KG = "simple_kg"
    NEBULAGRAPH = "nebulagraph"
    FALKORDB = "falkordb"

    # EMPTY
    EMPTY = "empty"
    COMPOSITE = "composite"

    PANDAS = "pandas"

    DOCUMENT_SUMMARY = "document_summary"

    # Managed
    VECTARA = "vectara"
