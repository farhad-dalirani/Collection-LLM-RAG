import json
import logging
import gradio as gr

from knowledgeBase.collection import CollectionManager
from user_agent import UserAgent

collection_manager = CollectionManager()

def ai_response(user_message, chat_interface, user_models, selected_query_engines):
    """
    Generates a response from an AI agent based on the user's message and chat history.
    Args:
        user_message (str): The message input from the user.
        chat_interface (list): The chat history between the user and the AI agent.
        user_models (object): An instance of a user model that can interact with the AI agent.
    Returns:
        str: The response generated by the AI agent.
    """
    if user_models.openAI_api == "":
        chat_interface.append({"role": "user", "content": user_message})
        chat_interface.append({"role": "assistant", "content": "API key is not valid or missing. Please provide a valid API key."})
        return "", chat_interface

    # Check if the user has selected a query engine in case of Router-Based Query Engines mode
    if user_models.mode == "Router-Based Query Engines" and len(selected_query_engines) == 0:
        chat_interface.append({"role": "user", "content": user_message})
        chat_interface.append({"role": "assistant", "content": "Please select one or more query engines to answer your queries."})
        return "", chat_interface

    return user_models.interact_with_agent(message=user_message, chat_history=chat_interface)

def clear_chat(chat_interface, user_models):
    """
    Clears the chat interface.

    Args:
        chat_interface (gr.Chatbot): The chat interface component to be cleared.

    Returns:
        list: An empty list to reset the chat interface.
    """
    user_models.reset_memory()
    logging.info('>    Chat cleared.')
    return []

def toggle_button(text):
    """
    Function to check if the textbox has input and update the button's interactiveness.
    Args:
        text (str): The input text from the textbox.
    Returns:
        gradio.components.Component: An updated button component with its interactiveness set based on the input text.
    """
    return gr.update(interactive=bool(text))

def change_mode(mode_radio, user_models):
    """
    Update the user's selected mode based on the provided radio button selection.

    Args:
        mode_radio (str): The identifier of the selected mode.
        user_models (UserModels): An instance of the UserModels class that manages
                                  the user's models and settings.

    Returns:
        None
    """
    user_models.set_mode(mode_radio)
    logging.info(">    Mode changed to: {}".format(mode_radio))

def change_llm(llm_radio, user_models):
    """
    Update the user's selected LLM (Language Model).

    This function updates the user's selected LLM model based on the provided
    radio button selection.

    Args:
        llm_radio (str): The identifier of the selected LLM model.
        user_models (UserModels): An instance of the UserModels class that manages
                                  the user's models and settings.

    Returns:
        None
    """
    user_models.set_llm(llm_radio)
    logging.info(">    LLM changed to: {}".format(llm_radio))

def change_embd(emb_radio, user_models):
    """
    Update the user's embedding model based on the selected embedding option.

    Parameters:
    emb_radio (str): The selected embedding model identifier.
    user_models (UserModels): An instance of the UserModels class that manages user-specific models.

    Returns:
    None
    """
    user_models.set_embd(emb_radio)
    logging.info(">    Embedding model changed to: {}".format(emb_radio))

def change_models_api(open_ai_api_textbox, user_models):
    """
    Updates the user's models with a new API key if provided.

    Args:
        open_ai_api_textbox (str): The new API key entered by the user.
        user_models (object): The user's models object that has a method `set_api` to update the API key.

    Returns:
        None
    """
    if open_ai_api_textbox != "":
        user_models.set_api(open_ai_api_textbox)
    logging.info(">    API key updated.")

def new_query_engine(user_models, path_json_file, type_json, chat_interface):
    """
    Creates a new query engine based on a input json file that contain name of article/papers and their links.

    Args:
        user_models (list): A list of user models to be used by the query engine.
        path_json_file (str): The file path to the JSON configuration file.
        type_json (str): The type of JSON configuration (e.g., 'schema', 'data').

    Returns:
        None
    """
    if user_models.openAI_api == "":
        chat_interface.append({"role": "assistant", "content": "API key is not valid or missing. Please provide a valid API key."})
        return chat_interface
    
    try:
        collection_manager.create_new_collection(user_models, path_json_file, type_json)
    except Exception as e:
        chat_interface.append({"role": "assistant", "content": f"An error occurred: {e}"})
        return chat_interface

    logging.info('>    New Query Engine, Vector Index, and Keyword Index were created and saved.')

    return chat_interface

def on_select_query_engine(user_models, selected_query_engines):
    """
    Update the set of query engine tools for the agents based on the provided list of query engine names.

    Args:
        user_models (UserModels): An instance of UserModels containing the current state and details of the user's models.
        selected_query_engines (list of str): A list of query engine names to be set for the agents.

    Returns:
        None
    """
    user_models.set_agent(query_engines_details=collection_manager.get_query_engines_detail_by_name(selected_query_engines))
    logging.info('>    Query Engine(s) selected: {}'.format(selected_query_engines))

def delete_query_engine(selected_query_engine):
    """
    """
    collection_manager.delete_query_engine_by_name(selected_query_engine)
    logging.info('>   Query Engine {} was deleted.'.format(selected_query_engine))
    return None


def lock_component(*components):
    """
    Locks the given components by setting them to be non-interactive.
    Args:
        *components: A variable number of components to be locked.
    Returns:
        list: A list of updated components with their interactive property set to False.
    """
    
    return [gr.update(interactive=False) for _ in components]

def unlock_component(*components):
    """
    Unlocks the given components by setting them to be interactive.
    Args:
        *components: A variable number of components to be unlocked.
    Returns:
        list: A list of updated components with their 'interactive' attribute set to True.
    """
    
    return [gr.update(interactive=True) for _ in components]

def launch_app():
    """
    Launches the web-based GUI application for LLMConfRAG.
    This function initializes the application by loading configuration settings from a JSON file,
    setting up user models, and creating the graphical user interface using Gradio. The GUI allows
    users to select and configure large language models (LLMs) and embedding models, manage query
    engines, and interact with the AI through a chat interface.
    The main components of the GUI include:
    - Settings Accordion: Allows users to choose modes, LLMs, embedding models, and enter OpenAI API keys.
    - Query Engines Accordion: Provides options to create new query engines and select existing ones.
    - Chat Area: Displays user questions and AI responses, and includes a textbox for user input.
    The function also handles various interactive elements such as radio buttons, textboxes, file uploads,
    and buttons to create new query engines and submit user messages.
    Note:
    - The function assumes the presence of a configuration file at './LLMCONFRAG/program_init_config.json'.
    - The function uses Gradio for creating the web-based GUI.
    Raises:
    - FileNotFoundError: If the configuration file is not found.
    - json.JSONDecodeError: If there is an error in parsing the configuration file.
    """
    # Loading setting configurations
    with open('./Collection_LLM_RAG/program_init_config.json', 'r') as file:
        config_data = json.load(file)         
    llm_names = [name + ' (Local)' for name in config_data['LLMs']['local']]
    llm_names.extend([name for name in config_data['LLMs']['API']])
    emb_names = [name + ' (Local)' for name in config_data['Embedding']['local']]
    emb_names.extend([name for name in config_data['Embedding']['API']])

    # Web based GUI
    with gr.Blocks(theme=gr.themes.Ocean()) as app:
        
        # Each user has its own models and settings
        user_models = gr.State(
            UserAgent(
                llm_name=llm_names[0], 
                embedding_name=emb_names[0], 
                mode=config_data['Modes'][0],
                query_engines_details=collection_manager.get_query_engines_detail(), 
                openAI_api="")
            )
        
        with gr.Row():

            # First column
            with gr.Column(scale=1):
            
                # Settings related to choosing hyper parameters related
                # to llms, embeding models, etc
                with gr.Accordion("⚙️ Settings"):
                    
                    # Chosing mode: ReAct Agent or pure query engine
                    mode_radio = gr.Radio(config_data['Modes'], label='Mode:', value=config_data['Modes'][0], interactive=True)

                    # Chosing the llm for AI model
                    llm_radio = gr.Radio(llm_names, label='Large Language Model:', value=llm_names[0], interactive=True)
                    
                    # Chosing the embedding model for AI model
                    emb_radio = gr.Radio(emb_names, label='Embedding Model:', value=emb_names[0], interactive=True)

                    # Textbox for entering OpenAI API
                    open_ai_api_textbox = gr.Textbox(
                                    label="OpenAI API:",
                                    placeholder="Enter your OpenAI API here",
                                    lines=1,
                                    max_lines=1,
                                    type="password"
                                )
            
            # Second column, Chat area
            with gr.Column(scale=4):
                # Area to show user questions and AI responses
                chat_interface = gr.Chatbot(type='messages', min_height=600)

                # User input text box
                user_message = gr.Textbox(placeholder='Message LLMConfRag', 
                                          label='', submit_btn=True)

                # Button for clearing chat
                clear_button = gr.Button(value="Clear Chat")

                # Selecting one or more query engines to answer queries
                selected_query_engines = gr.CheckboxGroup(
                                            collection_manager.get_query_engines_name(), 
                                            value=collection_manager.get_query_engines_name(), 
                                            label="Select Existing Query Engines to Use", interactive=True)

            # Third column
            with gr.Column(scale=1):
                
                # Query engines
                with gr.Accordion("🗄️ Create New Query Engine"):                    
                    
                    # Upload a JSON file containing article/paper names and their links to create a new query engine.
                    path_documents_json_file = gr.File(label="Upload a JSON File", file_count='single', file_types=[".json"])
                    type_documents_folder = gr.Radio(config_data['QueryEngine-creation-input-type'],
                                                     value=config_data['QueryEngine-creation-input-type'][0], 
                                                     label='Type of Files in Directory', 
                                                     interactive=True)
                    button_create_new_Query_engine = gr.Button(value="Create", interactive=False)
                    
                with gr.Accordion("🗑️ Delete Query Engine"):
                    # Select a query engine to delete
                    delete_query_engine_dropdown = gr.Dropdown(collection_manager.get_query_engines_name(), label="Select Query Engine to Delete")
                    button_delete_query_engine = gr.Button(value="Delete", interactive=False)
                       

        # Event handling
        
        # Lock the components during changes
        lock_list = [mode_radio,
        llm_radio,
        user_message,
        emb_radio,
        open_ai_api_textbox,
        path_documents_json_file,
        type_documents_folder,
        button_create_new_Query_engine,
        selected_query_engines,
        clear_button]

        # Update the mode based on the selected radio button
        mode_radio.change(
            lock_component, inputs=lock_list, outputs=lock_list
        ).then(
            change_mode, inputs=[mode_radio, user_models]
        ).then(
            unlock_component, inputs=lock_list, outputs=lock_list
        )

        # Update the llm model based on the selected radio button
        llm_radio.change(
            lock_component, inputs=lock_list, outputs=lock_list
        ).then(
            change_llm, inputs=[llm_radio, user_models]
        ).then(
            unlock_component, inputs=lock_list, outputs=lock_list
        )

        # Update the embedding model based on the selected radio button
        emb_radio.change(
            lock_component, inputs=lock_list, outputs=lock_list
        ).then(
            change_embd, inputs=[emb_radio, user_models]
        ).then(
            unlock_component, inputs=lock_list, outputs=lock_list
        )
                
        # Update API key if provided
        open_ai_api_textbox.blur(
            lock_component, inputs=lock_list, outputs=lock_list
        ).then(
            change_models_api, inputs=[open_ai_api_textbox, user_models]
        ).then(
            unlock_component, inputs=lock_list, outputs=lock_list
        )

        # Connect the toggle function to the textbox input
        path_documents_json_file.change(
            lock_component, inputs=lock_list, outputs=lock_list
        ).then(
            fn=toggle_button, inputs=path_documents_json_file, outputs=button_create_new_Query_engine
        ).then(
            unlock_component, inputs=lock_list, outputs=lock_list
        )
        
        # Enable the delete button only if a query engine is selected
        delete_query_engine_dropdown.focus(
                fn=toggle_button, inputs=delete_query_engine_dropdown, outputs=button_delete_query_engine
        )
                     
        # Clear chat        
        clear_button.click(clear_chat, inputs=[chat_interface, user_models], outputs=[chat_interface])
        
        # Update the selected query engines based on the checkbox group
        selected_query_engines.select(
            lock_component, inputs=lock_list, outputs=lock_list
        ).then(
            fn=on_select_query_engine, inputs=[user_models, selected_query_engines]
        ).then(
            unlock_component, inputs=lock_list, outputs=lock_list
        )
            
        # Send current user message and previous user messages and AI asnwers the ai to get a new asnwer
        user_message.submit(
            lock_component, inputs=lock_list, outputs=lock_list
        ).then(ai_response, 
            inputs=[user_message, chat_interface, user_models, selected_query_engines], 
            outputs=[user_message, chat_interface]
        ).then(
            unlock_component, inputs=lock_list, outputs=lock_list
        )
        
        # Call function for deleting query engine if the button pressed
        button_delete_query_engine.click(
            lock_component, inputs=lock_list, outputs=lock_list
        ).then(
            delete_query_engine,
            inputs=[delete_query_engine_dropdown],
            outputs=None
        ).then(
            fn=lambda: gr.CheckboxGroup(choices=collection_manager.get_query_engines_name(), value=collection_manager.get_query_engines_name()), 
            outputs=selected_query_engines
        ).then(
            lambda: gr.Dropdown(
                choices=collection_manager.get_query_engines_name(), 
                value=collection_manager.get_query_engines_name()[0] if collection_manager.get_query_engines_name() else None), 
            outputs=delete_query_engine_dropdown
        ).then(
            fn=lambda: gr.Button(value="Delete", interactive=False), 
            outputs=button_delete_query_engine
        ).then(
            fn=on_select_query_engine, inputs=[user_models, selected_query_engines]
        ).then(
            unlock_component, inputs=lock_list, outputs=lock_list
        )
        
        # Call function for creating new query engine if the button pressed
        button_create_new_Query_engine.click(
            lock_component, inputs=lock_list, outputs=lock_list
        ).then(
            new_query_engine,
            inputs=[user_models, path_documents_json_file, type_documents_folder, chat_interface], 
            outputs=[chat_interface]
        ).then(
            lambda: gr.Button(value="Create", interactive=False), outputs=button_create_new_Query_engine
        ).then(
            lambda: gr.CheckboxGroup(
                choices=collection_manager.get_query_engines_name(), 
                value=collection_manager.get_query_engines_name()), 
            outputs=selected_query_engines
        ).then(
            lambda: gr.Dropdown(
                choices=collection_manager.get_query_engines_name(), 
                value=collection_manager.get_query_engines_name()[0] if collection_manager.get_query_engines_name() else None), 
            outputs=delete_query_engine_dropdown
        ).then(
            fn=on_select_query_engine, inputs=[user_models, selected_query_engines]
        ).then(
            unlock_component, inputs=lock_list, outputs=lock_list
        )


    # Launch the web based GUI
    app.launch()

if __name__ == '__main__':
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Logging is configured.')

    # Launch the web-based GUI
    launch_app()
    