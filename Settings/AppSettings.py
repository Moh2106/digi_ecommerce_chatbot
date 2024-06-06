from dotenv import load_dotenv
import os

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

load_dotenv()

class AppSettings:

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    def __init__(self):
        pass

    def get_llm(self):
        llm = Groq(model="llama3-8b-8192", api_key=self.GROQ_API_KEY,
                   system_prompt="You are a sales expert. Your goal is to provide useful and satisfying"
                                 " answers to help the customer make an informed decision. "
                                 "You have a document that contains information about the different products. "
                                 "When the customer asks a question you answer in clear and precise ways."
                                 " If the customer requests information about a product you give him the necessary "
                                 "information that is in the document. When the customer requests the product"
                                 " information you provide the product description."
                   )
        return llm

    def get_embeddings(self):
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        return embed_model
