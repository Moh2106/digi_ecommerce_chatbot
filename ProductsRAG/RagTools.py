from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter


class RagTools:
    def __init__(self):
        pass

    def get_router_query_engine(self, file_path: str, embed_model=None):
        """Get router query engine."""
        embed_model = embed_model

        # load documents
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        # splitter documents
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)

        # Vectorize documents
        vector_index = VectorStoreIndex(nodes, embed_model=embed_model)

        return vector_index