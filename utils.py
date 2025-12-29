import openai
import os
import math
import tempfile
from pydub import AudioSegment
import streamlit as st
import io

def get_api_key():
    """Retrieves API key from st.secrets or environment variables."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass # Secrets file not found or other streamlit error, fallback to env vars

    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    else:
        return None

def transcribe_audio(file_path, client):
    """Transcribes a single audio file using OpenAI Whisper."""
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcript

def split_and_transcribe(file_path, client):
    """Splits large audio files and transcribes chunks."""
    audio = AudioSegment.from_file(file_path)
    
    # 10 minutes in milliseconds
    chunk_length_ms = 10 * 60 * 1000 
    chunks = math.ceil(len(audio) / chunk_length_ms)
    
    full_transcript = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(chunks):
        start_time = i * chunk_length_ms
        end_time = min((i + 1) * chunk_length_ms, len(audio))
        
        chunk = audio[start_time:end_time]
        
        # Create a temp file for the chunk
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_chunk:
            chunk.export(temp_chunk.name, format="mp3")
            temp_chunk_path = temp_chunk.name
            
        status_text.text(f"Transcribing part {i+1} of {chunks}...")
        
        try:
            transcript_part = transcribe_audio(temp_chunk_path, client)
            full_transcript.append(transcript_part)
        finally:
            os.remove(temp_chunk_path)
            
        progress_bar.progress((i + 1) / chunks)
        
    return "\n".join(full_transcript)

def summarize_text(text, num_words, client):
    """Summarizes the text to approximately num_words."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Please summarize the following text in approximately {num_words} words:\n\n{text}"}
        ]
    )
    return response.choices[0].message.content

def extract_key_ideas(text, num_ideas, client):
    """Extracts key ideas from the text."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that identifies key ideas from text."},
            {"role": "user", "content": f"Please extract the top {num_ideas} key ideas from the following text as a bulleted list:\n\n{text}"}
        ]
    )
    return response.choices[0].message.content

def ask_question(context_text, question, client):
    """Asks a question about the context text."""
    response = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[
           {"role": "system", "content": "You are a helpful assistant. Answer the user's question based strictly on the provided context text. Be concise, clear, and direct."},
           {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
       ]
   )
    return response.choices[0].message.content
