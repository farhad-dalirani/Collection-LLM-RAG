import gradio as gr
import time 
import random

def respond(message, chat_history):
        bot_message = random.choice(["How are you?", "Today is a great day", "I'm very hungry"])
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        time.sleep(2)
        return "", chat_history


if __name__ == '__main__':
    
    # Web based GUI
    with gr.Blocks() as app:
        
        with gr.Row():

            # First column
            with gr.Column(scale=1):
                
                # Selecting one or more query engines to answer
                # questions of users
                with gr.Accordion("📦 Select Query Engine"):
                    gr.Markdown("lorem ipsum")

                # Settings related to choosing hyper parameters related
                # to llms, embeding models, etc
                with gr.Accordion("⚙️ Settings"):
                    gr.Markdown("lorem ipsum")

            with gr.Column(scale=3):
                # Area to show user questions and AI responses
                chat_interface = gr.Chatbot(type='messages')

                # User input text box
                user_message = gr.Textbox(placeholder="Message LLMConfRag")


                # Send current user message and previous user messages and AI asnwers
                # the ai to get a new asnwer
                user_message.submit(respond, 
                                    [user_message, chat_interface], [user_message, chat_interface])



    app.launch()
    