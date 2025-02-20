import logging
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding 
from llama_index.core.agent.react import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticMultiSelector
from llama_index.core.query_engine import SubQuestionQueryEngine
from openai import AuthenticationError

from knowledgeBase.hybrid_query_engine import load_hybrid_query_engine
from utils import sort_dict_by_values


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
    def __init__(self, llm_name, embedding_name, openAI_api, mode, query_engines_details=[], temperature=0, system_message=None):
        
        self.llm_name = llm_name
        self.embedding_name = embedding_name
        self.openAI_api = openAI_api
        self.mode = mode
        self.temperature = temperature

        self.model_llm = None
        self.model_embd = None
        self.agent = None
        
        self.memory = None

        self.query_engines_details = query_engines_details
        
        if self.openAI_api != "":
            self.set_llm(llm_name)
            self.set_embd(embedding_name)

        if system_message is None:
            self.system_message = """You are an AI teacher, answering questions from students of an applied AI course on Large Language Models (LLMs or llm) and Retrieval Augmented Generation (RAG) for LLMs.
    Topics covered include training models, fine-tuning models, giving memory to LLMs, prompting tips, hallucinations and bias, vector databases, transformer architectures, embeddings, RAG frameworks such as Langchain and LlamaIndex, making LLMs interact with tools, AI agents, reinforcement learning with human feedback (RLHF). Questions should be understood in this context.
    Your answers are aimed to teach students, so they should be complete, clear, and easy to understand.
    Use the available tools to gather insights pertinent to the field of AI.
    To answer student questions, always use the all_sources_info tool.
    Only some information returned by the tools might be relevant to the question, so ignore the irrelevant part and answer the question with what you have.
    Your responses are exclusively based on the output provided by the tools. Refrain from incorporating information not directly obtained from the tool's responses.
    When the conversation deepens or shifts focus within a topic, adapt your input to the tools to reflect these nuances. This means if a user requests further elaboration on a specific aspect of a previously discussed topic, you should reformulate your input to the tool to capture this new angle or more profound layer of inquiry.
    Provide comprehensive answers, ideally structured in multiple paragraphs, drawing from the tool's variety of relevant details. The depth and breadth of your responses should align with the scope and specificity of the information retrieved.
    Should the tools repository lack information on the queried topic, politely inform the user that the question transcends the bounds of your current knowledge base, citing the absence of relevant content in the tool's documentation.
    At the end of your answers, always invite the students to ask deeper questions about the topic if they have any. Make sure reformulate the question to the tool to capture this new angle or more profound layer of inquiry.
    Do not refer to the documentation directly, but use the information provided within it to answer questions.
    If code is provided in the information, share it with the students. It's important to provide complete code blocks so they can execute the code when they copy and paste them.
    Make sure to format your answers in Markdown format, including code blocks and snippets.
    """
        else:
            self.system_message = system_message

    def interact_with_agent(self, message, chat_history):
        """
        Interacts with the AI agent based on the selected mode and updates the chat history.
        Parameters:
        message (str): The user's message to be sent to the AI agent.
        chat_history (list): The current chat history, which will be updated with the new interaction.
        Returns:
        tuple: An empty string and the updated chat history.
        Raises:
        ValueError: If the selected mode is not supported.
        The function operates in two modes:
        1. "ReAct-Powered Query Engines": Sends the user's message to the AI agent and collects article names and links from the sources.
        2. "Router-Based Query Engines": Sends the user's message to the Router Query Engine and collects article names and links from the source nodes.
        The collected references are formatted and appended to the bot's message, which is then added to the chat history.
        """
        references = {}
        if self.mode == "ReAct-Powered Query Engines":
            # Send the user's message to the AI agent 
            try:
                ai_answer = self.agent.chat(message)
            except AuthenticationError:
                bot_message = "An error occurred: Authentication Error. Please check your OpenAI API key."
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": bot_message})
                logging.error("Authentication error: Incorrect API key provided.")
                return "", chat_history
            except Exception as e:
                bot_message = f"An error occurred: {e}"
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": bot_message})
                logging.error(f"An unexpected error occurred: {e}")
                return "", chat_history
            
            bot_message = ai_answer.response

            # Collect article names and links
            for tool_output in ai_answer.sources:
                raw_output = tool_output.raw_output
                # Check if raw_output has the attribute 'source_nodes', to avoid situations when 
                # the agent has not decided to retrieve any information from the query engines
                if hasattr(raw_output, 'source_nodes'):
                    for node in raw_output.source_nodes:
                        name = node.node.metadata.get('Name')
                        link = node.node.metadata.get('Link')
                        if name and link:
                            if name and link:
                                current_score = node.score if node.score is not None else 0
                                if name and len(name) > 80:
                                    name = name[:80] + "..."
                                if (name, link) in references:
                                    # Update the score if the reference already exists
                                    references[(name, link)] = max(references[(name, link)], current_score)
                                else:
                                    references[(name, link)] = current_score
                else:
                    # Handle the case where source_nodes isn't available
                    logging.info("Warning: 'source_nodes' attribute not found in raw_output.")
        
        elif self.mode in ["Router-Based Query Engines", "SubQuestion-Based Query Engines"]:    
            # Send the user's message to the Router Query Engine
            try:
                response = self.agent.query(message)
            except AuthenticationError:
                bot_message = "An error occurred: Authentication Error. Please check your OpenAI API key."
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": bot_message})
                logging.error("Authentication error: Incorrect API key provided.")
                return "", chat_history
            except Exception as e:
                bot_message = f"An error occurred: {e}"
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": bot_message})
                logging.error(f"An unexpected error occurred: {e}")
                return "", chat_history

            bot_message = response.response
            for source in response.source_nodes:
                # Access the underlying node from the NodeWithScore object
                node = source.node  
                # Assuming metadata is stored as a dict in the node's metadata attribute:
                metadata = node.metadata  
                name = metadata.get('Name')
                link = metadata.get('Link')
                if name and link:
                    current_score = source.score if source.score is not None else 0
                    if name and len(name) > 80:
                            name = name[:80] + "..."
                    if (name, link) in references:
                        # Update the score if the reference already exists
                        references[(name, link)] = max(references[(name, link)], current_score)
                    else:
                        references[(name, link)] = current_score
        else:
            raise ValueError('Selected mode is not supported.')
        
        # Format the references
        if references:
            # Sort the references by LLM Judge score 
            references = sort_dict_by_values(references)
            formatted_references = []
            # Loop through references
            for item in references:
                # Unpack the first part of the tuple and the score
                (name, link), score = item
                # Format the reference as needed
                formatted_references.append(f"🔗 [{name}]({link}) ⭐ {score:.2f}/1  | " if score != 0 else f"🔗 [{name}]({link}) ⭐ -/1  | ")

            references = formatted_references

            references_text = "Some helpful articles, sorted by relevance according to LLM Judge, along with semantic scores:\n" + " ".join(references)
            bot_message += "\n\n" + references_text
        
        # Update the chat history
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history


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
            self.model_llm = OpenAI(model="gpt-4o-mini", temperature=self.temperature, api_key=self.openAI_api, system_prompt=self.system_message)
        elif self.llm_name == 'OpenAI GPT-4o':
            self.model_llm = OpenAI(model="gpt-4o", temperature=self.temperature, api_key=self.openAI_api, system_prompt=self.system_message)
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
        Set up the agent with the provided query engines details.
        This method initializes and configures the agent based on the provided query engines details.
        It supports two modes: "ReAct-Powered Query Engines" and "Router-Based Query Engines".
        Args:
            query_engines_details (list): A list of dictionaries, each containing details of a query engine.
                Each dictionary should have the following keys:
                - 'name': The name of the query engine.
                - 'description': A description of the query engine.
        Raises:
            ValueError: If the selected mode is not supported.
        """
        self.query_engines_details = query_engines_details

        # Load and initialize query engines based on provided set of query engines
        qs_list = []
        for qs_detail_i in query_engines_details:
            print(qs_detail_i)
            # Load hybrid query engine: Semantic + Keyword-based
            qs_i = load_hybrid_query_engine(
                            model_llm=self.model_llm, 
                            model_embd=self.model_embd, 
                            query_engine_name=qs_detail_i['name'], 
                            query_engine_description=qs_detail_i['description']
                        )

            if qs_i is None:
                logging.info('>    Query engine {} could not be loaded.'.format(qs_detail_i['name']))
            else:
                logging.info('>    Query engine {} was loaded.'.format(qs_detail_i['name']))
                # Create a QueryEngine tool instance from the loaded query engine
                qs_i_tool = QueryEngineTool.from_defaults(
                   query_engine=qs_i,
                   description=qs_detail_i['description'],
                )
                qs_list.append(qs_i_tool)

        if self.mode == "ReAct-Powered Query Engines":
            # Initialize a ChatMemoryBuffer with a token limit
            self.memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

            # Create a ReActAgent using the list of tools, the language model, and the memory buffer
            self.agent = ReActAgent.from_tools(
                tools=qs_list,
                llm=self.model_llm,
                memory=self.memory,
                verbose=True
            )
        elif self.mode == "Router-Based Query Engines":
            # Create a RouterQueryEngine using the list of tools
            self.agent = RouterQueryEngine(
                            selector=PydanticMultiSelector.from_defaults(llm=self.model_llm),
                            query_engine_tools=qs_list,
                            llm=self.model_llm,
                            verbose=True
                        )
        elif self.mode == "SubQuestion-Based Query Engines":
            self.agent = SubQuestionQueryEngine.from_defaults(
                query_engine_tools=qs_list,
                llm=self.model_llm,
                verbose=True
            )
        else:
           raise ValueError('Selected mode is not supported.')


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
    

    def set_mode(self, mode):
        """
        Sets the mode of the agent.
        Args:
            mode (str): The mode of the agent. Supported values are 'ReAct-Powered Query Engines' and 'Router-Based Query Engines'.
        """
        self.mode = mode
        self.set_agent(query_engines_details=self.query_engines_details)

    def reset_memory(self):
        """
        Resets the memory buffer of the agent.
        """
        if self.memory is not None:
            self.memory.reset()