import gradio as gr
import time 
import random
import json 
from createKnowledgeBase.create_query_engines import create_new_query_engine
from utils import get_folders


def respond(message, chat_history):
        bot_message = random.choice(["How are you?", "Today is a great day", "I'm very hungry"])
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        time.sleep(2)
        return "", chat_history

def toggle_button(text):
    """Function to check if textbox has input and update button interactiveness"""
    return gr.update(interactive=bool(text))

def new_query_engine(path_json_file, type_json):
    create_new_query_engine(path_json_file, type_json)

if __name__ == '__main__':
    
    with open('./LLMCONFRAG/program_init_config.json', 'r') as file:
        config_data = json.load(file)         

    # Web based GUI
    with gr.Blocks() as app:
        
        with gr.Row():

            # First column
            with gr.Column(scale=1):
            
                # Settings related to choosing hyper parameters related
                # to llms, embeding models, etc
                with gr.Accordion("⚙️ Settings"):

                    # Chosing the llm for AI model
                    llm_names = [ name + ' (Local)' for name in config_data['LLMs']['local']]
                    llm_names.extend([ name for name in config_data['LLMs']['API']])
                    llm_radio = gr.Radio(llm_names, label='Large Language Model:', value=llm_names[0], interactive=True)
                    
                    # Chosing the embedding model for AI model
                    emb_names = [ name + ' (Local)' for name in config_data['Embedding']['local']]
                    emb_names.extend([ name for name in config_data['Embedding']['API']])
                    emb_radio = gr.Radio(emb_names, label='Embedding Model:', value=emb_names[0], interactive=True)

                    temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, label="LLM Temperature") 

                # Query engines
                with gr.Accordion("🗄️ Query Engines"):                    
                    
                    # Create new query engine
                    gr.Markdown("Create a New Query Engine")
                    path_documents_folder = gr.Textbox(label='Path to documents directory:', placeholder='Enter Path')
                    type_documents_folder = gr.Radio(config_data['QueryEngine-creation-input-type'],
                                                     value=config_data['QueryEngine-creation-input-type'][0], 
                                                     label='Type of Files in Directory', 
                                                     interactive=True)
                    button_create_new_Query_engine = gr.Button(value="Create", interactive=False)
                    # Connect the toggle function to the textbox input
                    path_documents_folder.change(fn=toggle_button, inputs=path_documents_folder, outputs=button_create_new_Query_engine)
                    # Call function for creating new query engine if the button pressed
                    button_create_new_Query_engine.click(new_query_engine, inputs=[path_documents_folder, type_documents_folder], outputs=None)

                    # Selecting one or more query engines to answer
                    # questions of users
                    gr.Markdown("Existing Query Engines")
                    folders_name = get_folders()
                    selected_query_engines = gr.CheckboxGroup(folders_name, label="Select Query Engine", interactive=True)


            with gr.Column(scale=3):
                # Area to show user questions and AI responses
                chat_interface = gr.Chatbot(type='messages')

                # User input text box
                user_message = gr.Textbox(placeholder='Message LLMConfRag', 
                                          label='', submit_btn=True)


                # Send current user message and previous user messages and AI asnwers
                # the ai to get a new asnwer
                user_message.submit(respond, 
                                    [user_message, chat_interface], [user_message, chat_interface])
                
                


    app.launch()
    