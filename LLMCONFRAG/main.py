import gradio as gr
import json 
from knowledgeBase.query_engines import create_new_query_engine, get_query_engines_detail

from utils import get_query_engines_name, get_query_engines_detail_by_name, remove_duplicates_pairs
from user_agent import UserAgent

def interact_with_agent(message, chat_history, user_models):
    ai_answer = user_models.agent.chat(message)
    bot_message = ai_answer.response

    # Collect article names and links
    references = []
    for tool_output in ai_answer.sources:
        raw_output = tool_output.raw_output
        # Check if raw_output has the attribute 'source_nodes'
        if hasattr(raw_output, 'source_nodes'):
            for node in raw_output.source_nodes:
                name = node.node.metadata.get('Name')
                link = node.node.metadata.get('Link')
                if name and link:
                    references.append(f"[{name}]({link})")
        else:
            # Handle the case where source_nodes isn't available
            print("Warning: 'source_nodes' attribute not found in raw_output.")
    references = remove_duplicates_pairs(references)


    # Format the references
    if references:
        references_text = "Here are some articles you might find helpful:\n" + "\n".join(references)
        bot_message += "\n\n" + references_text
    
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})
    return "", chat_history

def toggle_button(text):
    """Function to check if textbox has input and update button interactiveness"""
    return gr.update(interactive=bool(text))

def change_llm(llm_radio, user_models):
    """Function to change User's llm if user changed the selected LLM model"""
    user_models.set_llm(llm_radio)

def change_embd(emb_radio, user_models):
    """Function to change User's embding model if user changed the selected embedding model"""
    user_models.set_embd(emb_radio)

def change_models_api(open_ai_api_textbox, user_models):
    """Function to change User's models if user changed the api"""
    if open_ai_api_textbox != "":
        user_models.set_api(open_ai_api_textbox)

def new_query_engine(user_models, path_json_file, type_json):
    """Creates a new query engine"""
    create_new_query_engine(user_models, path_json_file, type_json)

def on_select_query_engine(user_models, selected_query_engines):
    """Update agents query engine tools acoording to input name list"""
    user_models.query_engines_details = selected_query_engines
    user_models.set_agent(query_engines_details=get_query_engines_detail_by_name(selected_query_engines))

if __name__ == '__main__':
    
    # Loading setting configurations
    with open('./LLMCONFRAG/program_init_config.json', 'r') as file:
        config_data = json.load(file)         
    llm_names = [ name + ' (Local)' for name in config_data['LLMs']['local']]
    llm_names.extend([ name for name in config_data['LLMs']['API']])
    emb_names = [ name + ' (Local)' for name in config_data['Embedding']['local']]
    emb_names.extend([ name for name in config_data['Embedding']['API']])

    # Web based GUI
    with gr.Blocks() as app:
        
        # Each user has its own models and settings
        user_models = gr.State(
            UserAgent(
                llm_name=llm_names[0], 
                embedding_name=emb_names[0], 
                query_engines_details=get_query_engines_detail(), 
                openAI_api="")
            )
        
        with gr.Row():

            # First column
            with gr.Column(scale=1):
            
                # Settings related to choosing hyper parameters related
                # to llms, embeding models, etc
                with gr.Accordion("⚙️ Settings"):

                    # Chosing the llm for AI model
                    llm_radio = gr.Radio(llm_names, label='Large Language Model:', value=llm_names[0], interactive=True)
                    llm_radio.change(change_llm, inputs=[llm_radio, user_models])

                    # Chosing the embedding model for AI model
                    emb_radio = gr.Radio(emb_names, label='Embedding Model:', value=emb_names[0], interactive=True)
                    emb_radio.change(change_embd, inputs=[emb_radio, user_models])

                    # Textbox for entering OpenAI API
                    open_ai_api_textbox = gr.Textbox(
                                    label="OpenAI API:",
                                    placeholder="Enter your OpenAI API here",
                                    lines=1,
                                    max_lines=1,
                                    type="password"
                                )
                    open_ai_api_textbox.blur(
                        change_models_api, 
                        inputs=[open_ai_api_textbox, user_models])

                
                # Query engines
                with gr.Accordion("🗄️ Query Engines"):                    
                    
                    # Create new query engine
                    gr.Markdown("Create a New Query Engine")
                    path_documents_json_file = gr.File(label="Upload a File", file_count='single', file_types=[".json"])
                    type_documents_folder = gr.Radio(config_data['QueryEngine-creation-input-type'],
                                                     value=config_data['QueryEngine-creation-input-type'][0], 
                                                     label='Type of Files in Directory', 
                                                     interactive=True)
                    button_create_new_Query_engine = gr.Button(value="Create", interactive=False)
                    # Connect the toggle function to the textbox input
                    path_documents_json_file.change(
                            fn=toggle_button, inputs=path_documents_json_file, outputs=button_create_new_Query_engine
                        )

                    # Selecting one or more query engines to answer
                    # questions of users
                    gr.Markdown("Existing Query Engines")
                    selected_query_engines = gr.CheckboxGroup(get_query_engines_name(), value=get_query_engines_name(), label="Select Query Engine", interactive=True)
                    selected_query_engines.select(fn=on_select_query_engine, inputs=[user_models, selected_query_engines])

                    # Call function for creating new query engine if the button pressed
                    button_create_new_Query_engine.click(
                        new_query_engine,
                        inputs=[user_models, path_documents_json_file, type_documents_folder], 
                        outputs=None
                        ).then(lambda: gr.CheckboxGroup(choices=get_query_engines_name()), outputs=selected_query_engines)
                    

            # Second column, Chat area
            with gr.Column(scale=3):
                # Area to show user questions and AI responses
                chat_interface = gr.Chatbot(type='messages')

                # User input text box
                user_message = gr.Textbox(placeholder='Message LLMConfRag', 
                                          label='', submit_btn=True)


                # Send current user message and previous user messages and AI asnwers
                # the ai to get a new asnwer
                user_message.submit(interact_with_agent, 
                                    inputs=[user_message, chat_interface, user_models], 
                                    outputs=[user_message, chat_interface])
                
    
    app.launch()
    