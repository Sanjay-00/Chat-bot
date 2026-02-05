import streamlit as st
from langsmith import traceable
from chatbot_backend import chatbot, retrive_all_thread
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from chatbot_backend import delete_thread
import os 

os.environ["LANGCHAIN_PROJECT"] = "Chatbot frontend"
# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================
# @traceable(name='Generate threads')
def generate_thread_id():
    """Generates a unique identifier for a new chat session."""
    return str(uuid.uuid4())
# @traceable(name='reset chat')
def reset_chat():
    """Clears current chat view and prepares a fresh thread ID."""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['msg_history'] = []
    # Note: thread_history is updated only after the first message is sent
# @traceable(name='retrive_msgs')
def retrive_msgs(thread_id):
    """Retrieves the full message history from LangGraph's checkpointer."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get('messages', [])
# @traceable
def retrive_title(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Retrieve the title we saved in Step 2
    return state.values.get('title', str(thread_id)[:8])


# =============================================================================
# 2. SESSION STATE INITIALIZATION
# =============================================================================

# Stores the display-ready list of dicts: [{'role': 'user', 'content': '...'}]
if 'msg_history' not in st.session_state:
    st.session_state['msg_history'] = []

# Stores the list of all unique thread IDs for the sidebar
if 'thread_history' not in st.session_state:
    st.session_state['thread_history'] = retrive_all_thread()

# Tracks the current active thread ID
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()


# =============================================================================
# 3. SIDEBAR UI & CONVERSATION SWITCHING
# =============================================================================

st.sidebar.title('Langgraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

# Loop through history to build navigation buttons --> Run only if thread_id is in thread_history
for thread_id in st.session_state['thread_history'][::-1]:


    # 1. Create two side-by-side containers
    col1, col2 = st.sidebar.columns([0.8, 0.2])

    # Get the AI-generated title, fallback to short UUID if not found
    button_label = retrive_title(thread_id)
    
    if col1.button(button_label, key=thread_id):
        st.session_state['thread_id'] = thread_id
        messages = retrive_msgs(thread_id)

        # Glue Code - Convert BaseMessage objects from LangGraph to Streamlit-friendly dicts
        temp_dict = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_dict.append({'role': role, 'content': msg.content})

        st.session_state['msg_history'] = temp_dict
        st.rerun()


    # 3. The Delete Button (Right side)
    if col2.button("🗑️", key=f"del_{thread_id}"):
        # A. Delete from the Hard Drive (Backend)
        
        delete_thread(thread_id)
        
        # B. Delete from the Sidebar List (Frontend RAM)
        st.session_state['thread_history'].remove(thread_id)
        
        # C. Logic Check: If the user deleted the active chat, clear the screen
        if st.session_state['thread_id'] == thread_id:
            reset_chat() 
            st.rerun()
        else:
            st.rerun()

# =============================================================================
# 4. MAIN CHAT DISPLAY
# =============================================================================

# Set current conversation config
CONFIG = {"configurable": {"thread_id": st.session_state['thread_id']},
          "metadata":{"thread_id":st.session_state['thread_id']},
          "run_name": "chat_turn"}

# Print whole chat history in the screen 

for msg in st.session_state['msg_history']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])

# =============================================================================
# 5. CHAT INPUT & PROCESSING
# =============================================================================

user_input = st.chat_input("Type here")

if user_input:
    first_msg = False
    
    # Check if this is a new thread (not yet in sidebar) --> 1st message
    if st.session_state['thread_id'] not in st.session_state['thread_history']:
        # Silent call to LLM to create a title for the sidebar
        summary_response = chatbot.invoke(
            {'messages': [HumanMessage(content=f'summarize this message as a topic in 1-5 words, msg : {user_input}')]},
            config={"configurable": {"thread_id": "temp_summarizer"}}
        )
        summary_text = summary_response['messages'][-1].content.strip().replace('"', '')
        
        st.session_state['thread_history'].append(st.session_state['thread_id'])
        first_msg = True
        
        chatbot.update_state(
            config=CONFIG,
            values={"title": summary_text}
        )

    # Update UI with user's message
    st.session_state['msg_history'].append({'role': 'user', 'content': user_input})
    
    with st.chat_message('user'):
        st.text(user_input)

    # Process AI Response via Streaming
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content 
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]}, 
                config=CONFIG, 
                stream_mode='messages'
            )
            # if isinstance(message_chunk, AIMessage) and message_chunk.content
        )

    # Save AI response to history
    st.session_state['msg_history'].append({'role': 'assistant', 'content': ai_message})

    # Force rerun if new thread was created to show it in the sidebar
    if first_msg:
        st.rerun()