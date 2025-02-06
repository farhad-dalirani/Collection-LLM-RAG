import gradio as gr
import time 
import random
import json 
from utils import get_folders

different_folder_kinds = ['.pdf (Research Papers)',  '.pdf', '.json', '.text/.txt/.html']

def respond(message, chat_history):
        bot_message = random.choice(["How are you?", "Today is a great day", "I'm very hungry"])
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        time.sleep(2)
        return "", chat_history


if __name__ == '__main__':
    
    with open('./LLMCONFRAG/config.json', 'r') as file:
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
                    type_documents_folder = gr.Radio(different_folder_kinds,
                                                     value=different_folder_kinds[0], 
                                                     label='Type of Files in Directory', 
                                                     interactive=True)
                    button_create_new_Query_engine = gr.Button(value="Create")

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
    