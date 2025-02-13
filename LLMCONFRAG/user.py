from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding 

class User:
    def __init__(self, llm_name, embedding_name, openAI_api, temperature=0):
        
        self.llm_name = llm_name
        self.embedding_name = embedding_name
        self.openAI_api = openAI_api
        self.temperature = temperature

        self.model_llm = None
        self.model_embd = None

        if len(self.openAI_api) > 0:
            self.set_llm(llm_name)
            self.set_embd(embedding_name)

    def set_llm(self, llm_name):
        self.llm_name = llm_name
        if self.llm_name == 'OpenAI GPT-4o mini':
            self.model_llm = OpenAI(model="gpt-4o-mini", temperature=self.temperature, api_key=self.openAI_api)
        elif self.llm_name == 'OpenAI GPT-4o':
            self.model_llm = OpenAI(model="gpt-4o", temperature=self.temperature, api_key=self.openAI_api)
        else:
            raise ValueError('Selected LLM name is not supported.')

    def set_embd(self, embedding_name):
        self.embedding_name = embedding_name
        if self.embedding_name == 'OpenAI text-embedding-3-small':
            self.model_embd = OpenAIEmbedding(model="text-embedding-3-small", api_key=self.openAI_api)
        else:
            raise ValueError('Selected Embedding name is not supported.')
        
    def set_api(self, openAI_api):
        self.openAI_api = openAI_api
        self.set_llm(llm_name=self.llm_name)
        self.set_embd(embedding_name=self.embedding_name)

    
    