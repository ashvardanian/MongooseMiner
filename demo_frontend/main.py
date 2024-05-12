import gradio as gr
import os
import json

from groq import Groq
from search import answer_query
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="./.env")
except:
    pass

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

tools = [
    {
        "type": "function",
        "function": {
                "name": "get_related_functions",
                "description": "Get docstrings for internal functions for any library on PyPi.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                            "description": "A query to retrieve docstrings and find useful information.",
                        }
                    },
                    "required": ["user_query"],
                },
        },
    }
]


def user(user_message, history):
    return "", history + [[user_message, None]]


def get_related_functions(user_query: str) -> dict:
    docstring_top10 = answer_query(user_query)
    print("added torch mul")
    return docstring_top10[0]


def generate_rag(history):
    messages = [
        {
            "role": "system",
            "content": "You are a function calling LLM that uses the data extracted from the get_related_functions function to answer questions around writing Python code. Use the extraced docstrings to write better code."
        },
        {
            "role": "user",
            "content": history[-1][0],
        }
    ]
    history[-1][1] = ""
    tool_call_count = 0
    max_tool_calls = 3
    while tool_call_count <= max_tool_calls:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            tools=tools if tool_call_count < 3 else None,
            tool_choice="auto",
            max_tokens=4096
        )
        tool_call_count += 1
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            available_functions = {
                "get_related_functions": get_related_functions,
            }
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    user_query=function_args.get("user_query")
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
        else:
            break

    history[-1][1] += response_message.content
    return history


def generate_llama3(history):
    history[-1][1] = ""
    stream = client.chat.completions.create(
        messages=[
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation.
            {
                "role": "system",
                "content": "you are a helpful assistant."
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": history[-1][0],
            }
        ],
        stream=True,
        model="llama3-8b-8192",
        max_tokens=1024,
        temperature=0
    )

    for chunk in stream:
        if chunk.choices[0].delta.content != None:
            history[-1][1] += chunk.choices[0].delta.content
            yield history
        else:
            return


with gr.Blocks() as demo:

    with gr.Row():
        with gr.Column():
            gr.Markdown("# Mongoose Miner Search Demo")
            gr.Markdown(
                "Augmenting LLM code generation with function-level search across all of PyPi.")

    with gr.Row():
        chatbot = gr.Chatbot(height="35rem", label="Llama3 unaugmented")
        chatbot2 = gr.Chatbot(
            height="35rem", label="Llama3 with MongooseMiner Search")
    msg = gr.Textbox()

    clear = gr.Button("Clear")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        generate_llama3, chatbot, chatbot
    )
    msg.submit(user, [msg, chatbot2], [msg, chatbot2], queue=False).then(
        generate_rag, chatbot2, chatbot2
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    clear.click(lambda: None, None, chatbot2, queue=False)


demo.queue()
demo.launch()
