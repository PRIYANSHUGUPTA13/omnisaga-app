import os
import openai
import streamlit as st
from dotenv import load_dotenv
from serpapi import GoogleSearch
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
import tempfile
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_KEY")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Voice Recognition ---
def transcribe_audio(audio_file):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
            return r.recognize_google(audio)  # type: ignore[attr-defined]
    except Exception as e:
        st.error(f"Voice recognition failed: {str(e)}")
        return None

# --- Web Search ---
def web_search(query):
    if not serpapi_key:
        st.error("SerpAPI key not configured")
        return None
        
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": serpapi_key,
            "num": 3
        })
        results = search.get_dict()
        return " ".join(r.get("snippet", "") for r in results.get("organic_results", []))
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return None

# --- AI Response Generator ---
def generate_response(prompt, use_web=False):
    web_context = web_search(prompt) if use_web else None
    messages = [
        {"role": "system", "content": "You are OmniSage, a helpful AI assistant."},
        *st.session_state.messages[-6:],
        {"role": "user", "content": f"Web context: {web_context}\n\nQuestion: {prompt}" if web_context else prompt}
    ]
    
    try:
        response = openai.ChatCompletion.create(  # type: ignore[attr-defined]
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error: {str(e)}"

# --- Main App ---
def main():
    st.set_page_config(
        page_title="OmniSage Pro",
        page_icon="🧠",
        layout="centered"
    )
    
    st.title("🧠 OmniSage Pro")
    st.caption("Your AI assistant with voice, memory & web search")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        use_web = st.checkbox("Enable Web Search", True)
        voice_mode = st.checkbox("Voice Input", False)
        
        if voice_mode:
            st.caption("Press and hold to record:")
            audio_bytes = audio_recorder()
            
            if audio_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                    fp.write(audio_bytes)
                    if text := transcribe_audio(fp.name):
                        st.session_state.messages.append({"role": "user", "content": text})
                        st.rerun()
                    os.unlink(fp.name)

    # Chat interface
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            if response := generate_response(prompt, use_web):
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    # Clear button
    if st.session_state.messages and st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()