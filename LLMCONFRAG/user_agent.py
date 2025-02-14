from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding 
from llama_index.core.agent.react import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.memory import ChatMemoryBuffer

from knowledgeBase.query_engines import load_query_engine


class UserAgent:
    """
    A class to manage and interact with language models and embedding models from OpenAI, 
    and to set up query engines and agents for querying.
    Attributes:
        llm_name (str): The name of the language model to use.
        embedding_name (str): The name of the embedding model to use.
        openAI_api (str): The API key for accessing OpenAI services.
        query_engines_details (list): A list of details for query engines to be used.
        temperature (float): The temperature setting for the language model.
        model_llm (object): The language model instance.
        model_embd (object): The embedding model instance.
        agent (object): The agent instance for querying.
    Methods:
        __init__(llm_name, embedding_name, openAI_api, query_engines_details=[], temperature=0):
            Initializes the UserAgent with the specified parameters.
        set_llm(llm_name):
            Sets the language model based on the provided name.
        set_embd(embedding_name):
            Sets the embedding model based on the provided name.
        set_agent(query_engines_details):
            Sets up the agent with the provided query engines details.
        set_api(openAI_api):
            Sets the OpenAI API key and reinitializes the models and agent.
    """
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
        """
        Set the language model (LLM) based on the provided LLM name.

        Parameters:
        llm_name (str): The name of the language model to set. Supported values are 'OpenAI GPT-4o mini' and 'OpenAI GPT-4o'.

        Raises:
        ValueError: If the provided LLM name is not supported.
        """
        self.llm_name = llm_name
        if self.llm_name == 'OpenAI GPT-4o mini':
            self.model_llm = OpenAI(model="gpt-4o-mini", temperature=self.temperature, api_key=self.openAI_api)
        elif self.llm_name == 'OpenAI GPT-4o':
            self.model_llm = OpenAI(model="gpt-4o", temperature=self.temperature, api_key=self.openAI_api)
        else:
            raise ValueError('Selected LLM name is not supported.')

    def set_embd(self, embedding_name):
        """
        Sets the embedding model based on the provided embedding name.

        Parameters:
        embedding_name (str): The name of the embedding model to be set. Currently, only 'OpenAI text-embedding-3-small' is supported.

        Raises:
        ValueError: If the provided embedding name is not supported.
        """
        self.embedding_name = embedding_name
        if self.embedding_name == 'OpenAI text-embedding-3-small':
            self.model_embd = OpenAIEmbedding(model="text-embedding-3-small", api_key=self.openAI_api)
        else:
            raise ValueError('Selected Embedding name is not supported.')
    
    def set_agent(self, query_engines_details):
        """
        Initializes and sets up the agent with the provided query engines.
        Args:
            query_engines_details (list of dict): A list of dictionaries where each dictionary contains
                the details of a query engine. Each dictionary should have the following keys:
                - 'name' (str): The name of the query engine.
                - 'description' (str): A description of the query engine.
        The method performs the following steps:
        1. Loads and initializes query engines based on the provided details.
        2. Creates a list of QueryEngineTool instances from the loaded query engines.
        3. Initializes a ChatMemoryBuffer with a token limit of 1500.
        4. Creates a ReActAgent using the list of tools, the language model, and the memory buffer.
        5. Sets the created ReActAgent to the instance variable `self.agent`.
        If a query engine cannot be loaded, a message is printed indicating the failure.
        """

        # Load and initialize query engines based on provided set of query engines
        qs_list = []
        for qs_detail_i in query_engines_details:
            qs_i = load_query_engine(qs_detail_i['name'], llm_model=self.model_llm, embed_model=self.model_embd)
            if qs_i is None:
                print('>    Could not load the query engine.')
            else:
                print('>    Query engine {} was loaded.'.format(qs_detail_i['name']))
                # Create a QueryEngine tool instance from the loaded query engine
                qs_i_tool = QueryEngineTool.from_defaults(
                    query_engine=qs_i,
                    description=qs_detail_i['description'],
                )
                qs_list.append(qs_i_tool)

        # Initialize a ChatMemoryBuffer with a token limit
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

        # Create a ReActAgent using the list of tools, the language model, and the memory buffer
        self.agent = ReActAgent.from_tools(
            tools=qs_list,
            llm=self.model_llm,
            memory=memory,
            verbose=True
        )

    def set_api(self, openAI_api):
        """
        Sets the OpenAI API key and initializes the language model, embedding, and agent with the provided details.
        Args:
            openAI_api (str): The API key for accessing OpenAI services.
        """

        self.openAI_api = openAI_api
        self.set_llm(llm_name=self.llm_name)
        self.set_embd(embedding_name=self.embedding_name)
        self.set_agent(query_engines_details=self.query_engines_details)
    
    