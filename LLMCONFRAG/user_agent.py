from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding 
from llama_index.core.agent.react import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.memory import ChatMemoryBuffer

from knowledgeBase.query_engines import load_query_engine


class UserAgent:
    def __init__(self, llm_name, embedding_name, openAI_api, query_engines_details=[], temperature=0):
        
        self.llm_name = llm_name
        self.embedding_name = embedding_name
        self.openAI_api = openAI_api
        self.temperature = temperature

        self.model_llm = None
        self.model_embd = None
        self.agent = None

        self.query_engines_details = query_engines_details
        
        if self.openAI_api != "":
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
    
    def set_agent(self, query_engines_details):
        
        # Load query engines from file    
        qs_list = []
        for qs_detail_i in query_engines_details:
            qs_i = load_query_engine(qs_detail_i['name'], embed_model=self.model_embd)
            if qs_i is None:
                print('>    Could not load the query engine.')
            else:
                print('>    Query engine {} was loaded.'.format(qs_detail_i['name']))
                qs_i_tool = QueryEngineTool.from_defaults(
                    query_engine=qs_i,
                    description=qs_detail_i['description'],
                )
                qs_list.append(qs_i_tool)

        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

        self.agent = ReActAgent.from_tools(
            tools=qs_list,
            llm=self.model_llm,
            memory=memory,
            verbose=True
        )

    def set_api(self, openAI_api):
        self.openAI_api = openAI_api
        self.set_llm(llm_name=self.llm_name)
        self.set_embd(embedding_name=self.embedding_name)
        self.set_agent(query_engines_details=self.query_engines_details)
    
    