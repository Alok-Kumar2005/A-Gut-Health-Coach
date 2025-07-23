import streamlit as st
import asyncio
import uuid
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.ai_component.graph.graph import GutHealthCoach

st.set_page_config(
    page_title="August - Your Gut Health Coach",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #4CAF50;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #2196F3;
    }
    
    .ai-message {
        background-color: #e8f5e8;
        border-left-color: #4CAF50;
    }
    
    .sidebar-info {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'coach' not in st.session_state:
    st.session_state.coach = GutHealthCoach()
if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False

def run_async(coro):
    """Run async function in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

st.markdown("""
<div class="main-header">
    <h1>🌱 August - Your Gut Health Coach</h1>
    <p>Your personalized guide to digestive wellness and gut health optimization</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div class="sidebar-info">
        <h3>About August</h3>
        <p>August is your AI-powered gut health coach, designed to provide personalized guidance on:</p>
        <ul>
            <li>🔍 Digestive health issues</li>
            <li>🥗 Nutrition and gut-friendly foods</li>
            <li>💊 Probiotics and supplements</li>
            <li>🧘 Lifestyle factors affecting gut health</li>
            <li>🩺 When to seek medical attention</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.write("**Session Information:**")
    st.write(f"Session ID: `{st.session_state.session_id[:8]}...`")
    st.write(f"Messages: {len(st.session_state.conversation_history)}")
    
    if st.button("🔄 Start New Conversation", type="secondary"):
        st.session_state.conversation_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.conversation_started = False
        st.rerun()
    
    st.markdown("---")
    st.markdown("**💡 Try asking:**")
    sample_questions = [
        "I've been bloated for three days — what should I do?",
        "How does gut health affect sleep?",
        "What are the best probiotics for lactose intolerance?",
        "Why do I feel brain fog after eating sugar?"
    ]
    
    for question in sample_questions:
        if st.button(f"📝 {question[:30]}...", key=f"sample_{question[:20]}", help=question):
            st.session_state.user_input = question

col1, col2 = st.columns([3, 1])

with col1:
    if not st.session_state.conversation_started:
        with st.spinner("Starting conversation with August..."):
            welcome_message = run_async(
                st.session_state.coach.start_conversation(st.session_state.session_id)
            )
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": welcome_message,
                "timestamp": datetime.now()
            })
            st.session_state.conversation_started = True
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {message["content"]}
                    <small style="color: #666; float: right;">{message["timestamp"].strftime("%H:%M")}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>🌱 August:</strong><br>
                    {message["content"]}
                    <small style="color: #666; float: right;">{message["timestamp"].strftime("%H:%M")}</small>
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")

user_input = st.text_area(
    "💬 Ask August about your gut health:",
    placeholder="Type your question here... (e.g., 'I've been feeling bloated after meals, what could be causing this?')",
    height=100,
    key="user_input" if "user_input" not in st.session_state else None
)

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    send_button = st.button("🚀 Send Message", type="primary", use_container_width=True)

with col2:
    if st.button("🧹 Clear Input", use_container_width=True):
        st.session_state.user_input = ""
        st.rerun()

if send_button and user_input.strip():
    st.session_state.conversation_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now()
    })
    with st.spinner("August is thinking..."):
        try:
            response = run_async(
                st.session_state.coach.process_message(
                    user_input, 
                    st.session_state.session_id
                )
            )
            
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })
            
            st.session_state.user_input = ""
            st.rerun()
            
        except Exception as e:
            st.error(f"Sorry, I encountered an error: {str(e)}")
            st.error("Please try again or start a new conversation.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>
        💡 <strong>Disclaimer:</strong> August provides educational information and should not replace professional medical advice. 
        Always consult with healthcare providers for medical concerns.
    </small>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<script>
    var element = document.querySelector('.main');
    element.scrollTop = element.scrollHeight;
</script>
""", unsafe_allow_html=True)