from google import genai
from google.genai import types
import os
import streamlit as st
import tempfile
import yt_dlp

def get_api_key():
    """Retrieves Gemini API key from st.secrets or environment variables."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass # Secrets file not found or other streamlit error, fallback to env vars

    if os.getenv("GEMINI_API_KEY"):
        return os.getenv("GEMINI_API_KEY")
    else:
        return None

def get_gemini_client():
    """Initializes and returns the Gemini client."""
    api_key = get_api_key()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def download_youtube_audio(url):
    """Downloads audio from YouTube URL and returns the path to the temp file."""
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

def process_media_with_gemini(media_source, client, is_url=False):
    """
    Processes media (local file path or URL) using Gemini and returns structured JSON.
    """
    prompt = """
    Process the audio file and generate a detailed transcription.

    Requirements:
    1. Identify distinct speakers (e.g., Speaker 1, Speaker 2, or names if context allows).
    2. Provide accurate timestamps for each segment (Format: MM:SS).
    3. Detect the primary language of each segment.
    4. If the segment is in a language different than English, also provide the English translation.
    """

    if is_url:
        # For URL, we use the FileData with the URL directly if supported, 
        # but Gemini File API usually requires uploading or using a URI from GCS/etc.
        # However, the user's template showed YOUTUBE_URL being used in FileData.
        # Let's try that, but typically Gemini 2.x supports direct URI for some sources 
        # or we might need to upload. The template example uses a URL directly.
        file_part = types.Part(
            file_data=types.FileData(
                file_uri=media_source
            )
        )
    else:
        # For local file, we must upload it to Gemini's file service first
        file_upload = client.files.upload(path=media_source)
        file_part = types.Part(
            file_data=types.FileData(
                file_uri=file_upload.uri,
                mime_type=file_upload.mime_type
            )
        )

    response = client.models.generate_content(
        model="gemini-2.0-flash", # Using flash for speed/cost
        contents=[
            types.Content(
                parts=[
                    file_part,
                    types.Part(text=prompt)
                ]
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "summary": types.Schema(
                        type=types.Type.STRING,
                        description="A concise summary of the audio content.",
                    ),
                    "segments": types.Schema(
                        type=types.Type.ARRAY,
                        description="List of transcribed segments with speaker and timestamp.",
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "speaker": types.Schema(type=types.Type.STRING),
                                "timestamp": types.Schema(type=types.Type.STRING),
                                "content": types.Schema(type=types.Type.STRING),
                                "language": types.Schema(type=types.Type.STRING),
                                "language_code": types.Schema(type=types.Type.STRING),
                                "translation": types.Schema(type=types.Type.STRING),
                            },
                            required=["speaker", "timestamp", "content", "language", "language_code"],
                        ),
                    ),
                },
                required=["summary", "segments"],
            ),
        ),
    )
    
    # If we uploaded a file, we should probably clean it up in Gemini if possible, 
    # but the API doesn't always require immediate deletion. 
    # The local file cleanup is handled in calling function.
    
    return response.parsed

def summarize_text(text, num_words, client):
    """Summarizes text using Gemini."""
    prompt = f"Please summarize the following text in approximately {num_words} words:\n\n{text}"
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    return response.text

def extract_key_ideas(text, num_ideas, client):
    """Extracts key ideas using Gemini."""
    prompt = f"Please extract the top {num_ideas} key ideas from the following text as a bulleted list:\n\n{text}"
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    return response.text

def ask_question(context_text, question, client):
    """Asks a question about the context text using Gemini."""
    prompt = f"You are a helpful assistant. Answer the user's question based strictly on the provided context text. Be concise, clear, and direct.\n\nContext:\n{context_text}\n\nQuestion: {question}"
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    return response.text
