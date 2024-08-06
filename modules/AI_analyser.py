from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from modules.db_interactions import initialise_db, get_documents
import replicate
import os


model_path = "mistralai/mixtral-8x7b-instruct-v0.1"
response = None
model = None
os.environ["REPLICATE_API_TOKEN"] = "r8_HXkl38P4AwS8wMIQeiM32odzsKNwrir3Ff4BH"
collection = None



# use replicate API to generate response
def replicate_generation(chat_history):

    reference_chat = RAG(chat_history)
    prompt_template = """
    You are a dating advisor. You will be given a chat history and you are required to give a overall score on how interested user 1 is to user 2.

    Your task is to evalaute based on 2 categories: capital letters used and the frequency of the text.

    The score for each of them should be between 0 to 10.

    Please output the score in the following format: "category: /10".

    Please give a overall score based on the above two scores, which is between 0 to 20.

    Afterwards, give some suggestions for cur_user for his subsequent message. 

    1) Give an example message he should send
    2) Comment on how he can improve his messaging style
    3) Provide some interesting topics or pickup lines.
    4) Comment on the other user's interest level in the conversation
    
    There is a reference chat \
    that you can use when giving suggestions. The suggestion should create an engaging conversation that is \
    flirty and asks the person out.
    -------------------------------------------
    Here is one example:

    cur_user:

    Capital letters: 7/10
    Frequency: 8/10

    oth_user:

    Capital letters: 6/10
    Frequency: 8/10

    Overall score for cur_user: 20/20
    Overall score for oth_user: 19/20

    Suggestions:
    1) Cur user should...
    2) He is...
    -------------------------------------------
    The chat history is delimited by ###.
    Reference chat is delimited by !!!.

    ### {chat_history}
    !!! {reference_chat}

    Response:""".format(chat_history=chat_history, reference_chat = reference_chat)

    output = replicate.run(
        model_path,
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": prompt_template,
            "temperature": 0.3,
            "max_new_tokens": 1024,
            "prompt_template": "<s>[INST] {prompt} [/INST] ",
            "presence_penalty": 0,
            "frequency_penalty": 0
        }
    )
    output = "".join(output)
    return output

def RAG(chat_history):
    collection = initialise_db()
    results = get_documents(collection, chat_history)
    print("this is RAG results: ", results)

    return results

def AI_interest_eval(chat_history):


    # print("this is chat history", chat_history)
    # stringed_chat = str(chat_history)
    # print("string chat: ", stringed_chat)
    # AI_response = RAG(chat_history, collection)
    stringed_chat = str(chat_history)
    AI_response = replicate_generation(stringed_chat)
    return AI_response

#run LLM on local machine to get response
# def generate_response(chat_history, model):
#     prompt_template = "[INST]\n"+\
#         "You are a dating advisor. You are given a chat history which is in list. The chat history is delimited by ###. In the list, there are sub-lists which contain a conversation chunk."+\
#         "The first element of each sublist is user1. The other element is user 2.\n"+\
#         "Give a grade on how interested user 1 is to user 2.\n"+\
#         f"### {chat_history} \n"+\
#         "Response:\n[/INST]"
    
#     prompt = PromptTemplate.from_template(prompt_template)

#     chain = prompt | model
#     response = chain.invoke({"chat_history": chat_history})
#     print(response)
#     return response


#Shreyas Code
# def initialise_model(model_path,device_map):
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"

#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map=device_map,
#         trust_remote_code=False,
#         cache_dir='./models',
#         revision="main"
#     )

#     scoring_pipeline = pipeline(
#         model=model,
#         tokenizer=tokenizer,
#         task="text-generation",
#         return_full_text=False,
#         temperature=0.1,
#         do_sample=True,
#         top_p=0.95,
#         top_k=40,
#         max_new_tokens=300
#     )
#     mistral_pipeline = HuggingFacePipeline(pipeline=scoring_pipeline)
#     return mistral_pipeline