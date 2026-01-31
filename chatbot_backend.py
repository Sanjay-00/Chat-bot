from langgraph.graph import StateGraph, START , END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated 
from langchain_core.messages import BaseMessage , HumanMessage
from dotenv import load_dotenv 
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import  SqliteSaver
import sqlite3

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

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response], 'title': state.get('title', '')}

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_edge(START,'chat_node')
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
