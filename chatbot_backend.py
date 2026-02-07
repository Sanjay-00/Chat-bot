from langgraph.graph import StateGraph, START , END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated 
from langchain_core.messages import BaseMessage , HumanMessage
from dotenv import load_dotenv 
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import  SqliteSaver
import sqlite3

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
load_dotenv()


conn = sqlite3.connect('chatbot3.db', check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    title: str

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    temperature = 0.7
)
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    
# Make tool list
tools = [ search_tool, calculator]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response], 'title': state.get('title', '')}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START,'chat_node')
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")   
graph.add_edge('chat_node', END)

chatbot=graph.compile(checkpointer=checkpointer)
def retrive_all_thread():
    all_thread=set()
    for checkpoint in checkpointer.list(None):
        all_thread.add(checkpoint.config['configurable']['thread_id'])
    return list(all_thread)



def delete_thread(thread_id):
    """Physically removes all checkpoints for a specific thread from the DB."""
    # We open a fresh connection to ensure we don't interfere with the main app
    with sqlite3.connect('chatbot3.db') as conn:
        cursor = conn.cursor()
        # LangGraph stores its data in a table named 'checkpoints'
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()

if __name__ =='__main__':
    while True:

        user_message = input('type here: ')
        if user_message.strip().lower() == 'quit':
            break

        config={"configurable":{"thread_id":"5"}}

        # print(chatbot.invoke({'messages': user_message},config=config)['messages'][-1].content)
        for message_chunk,metadata in chatbot.stream({'messages':[HumanMessage(content= user_message)]}, config=config, stream_mode='messages'):
            if message_chunk:
                print(message_chunk.content)
        

    print(chatbot.get_state(config))
