# from LLM course on Udemy
# Expert Knowledge Worker
# A question answering agent that is an expert knowledge worker
# To be used by employees of Insurellm, an Insurance Tech company
# The agent needs to be accurate and the solution should be low cost.
# This project will use RAG (Retrieval Augmented Generation) to ensure our question/answering assistant has high accuracy.
# This first implementation will use a simple, brute-force type of RAG..

# imports
import os
import glob
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI


# price is a factor for our company, so we're going to use a low cost model
MODEL = "gpt-4o-mini"

# Load environment variables in a file called .env
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
openai = OpenAI()


# With massive thanks to student Dr John S. for fixing a bug in the below for Windows users!
context = {}
employees = glob.glob("knowledge-base/employees/*")

for employee in employees:
    name = employee.split(' ')[-1][:-3]
    doc = ""
    with open(employee, "r", encoding="utf-8") as f:
        doc = f.read()
    context[name]=doc


products = glob.glob("knowledge-base/products/*")

for product in products:
    name = product.split(os.sep)[-1][:-3]
    doc = ""
    with open(product, "r", encoding="utf-8") as f:
        doc = f.read()
    context[name]=doc
    
 
system_message = "You are an expert in answering accurate questions about Insurellm, the Insurance Tech company. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context."

# get context if available
def get_relevant_context(message):
    relevant_context = []
    for context_title, context_details in context.items():
        if context_title.lower() in message.lower():
            relevant_context.append(context_details)
    return relevant_context          


# update message with the context
def add_context(message):
    relevant_context = get_relevant_context(message)
    if relevant_context:
        message += "\n\nThe following additional context might be relevant in answering this question:\n\n"
        for relevant in relevant_context:
            message += relevant + "\n\n"
    return message


# chat with added context
def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history
    message = add_context(message)
    messages.append({"role": "user", "content": message})

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response
        

# UI
view = gr.ChatInterface(chat, type="messages").launch()