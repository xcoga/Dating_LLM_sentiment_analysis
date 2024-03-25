import gradio as gr
import random
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# use xichen2 env

model_path = "TheBloke/Llama-2-13B-Chat-GPTQ"
device_map = "auto"
# response = None
tokenizer = None
model = None


def initialise_model(model_path, device_map):

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map=device_map,
                                                 #  max_memory={ 0: "2GB", 1: "5GB", 2: "10GB"},
                                                 trust_remote_code=False,
                                                 revision="main")

    return model, tokenizer


async def generate_response(prompt, model, tokenizer):

    prompt_template = f'''[INST] <<SYS>>" In the list, there are a set of tuples."\
        "The first element of the tuple is user1. The other is user 2."\
        "Give a grade on how interested user 1 is to user 2."\
    <</SYS>>
    {prompt}[/INST]
    \n\n ### Response:'''

    # General PROMPT template

    # print("\n\n*** Generate:")
    input_ids = tokenizer(prompt_template, return_tensors="pt").input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    output = model.generate(inputs=input_ids, temperature=0.7,
                            do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    response = tokenizer.decode(output[0])
    print(response)

    return response


def respond(message, chat_history):
    bot_message = random.choice(
        ["How are you?", "I love you", "I'm very hungry"])
    chat_history.append((message, bot_message))
    time.sleep(0.5)
    return "", chat_history


async def AI_interest_eval(chat_history):
    print("this is chat history", chat_history)
    stringed_chat = str(chat_history)
    print("string chat: ", stringed_chat)
    AI_response = await generate_response(chat_history, model, tokenizer)

    return AI_response


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    AI_assessment_btn = gr.Button(value="AI interest evaluation")
    AI_review = gr.Textbox(label="AI's opinion on this conversation")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    AI_assessment_btn.click(
        AI_interest_eval, inputs=chatbot, outputs=AI_review)


if __name__ == "__main__":
    model, tokenizer = initialise_model(model_path, device_map)
    demo.launch(debug=True)
