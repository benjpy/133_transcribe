import streamlit as st
import openai
import os
from dotenv import load_dotenv
from pydub import AudioSegment
import tempfile
import io
import utils

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = utils.get_api_key()

if not api_key:
    st.error("OPENAI_API_KEY not found. Please set it in .env or Streamlit Cloud Secrets.")
    st.stop()

client = openai.OpenAI(api_key=api_key)

st.title("Video/Audio Transcriber & Summarizer 🎙️ 📝")

tab1, tab2 = st.tabs(["Transcribe Audio/Video", "Summarize Text File"])

with tab1:
    st.write("Upload a video or audio file to transcribe it using OpenAI's Whisper model.")
    uploaded_file = st.file_uploader("Choose a media file", type=["mp3", "mp4", "wav", "m4a", "mpeg", "mpga", "webm"], key="media_uploader")

    if uploaded_file is not None:
        if st.button("Transcribe", key="transcribe_btn"):
            with st.spinner("Processing file..."):
                # Save uploaded file to temp file
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                converted_mp3_path = None
                
                try:
                    # Convert to MP3 first to reduce size
                    st.info("Converting file to audio (MP3)...")
                    audio = AudioSegment.from_file(tmp_file_path)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
                        audio.export(tmp_mp3.name, format="mp3")
                        converted_mp3_path = tmp_mp3.name
                    
                    # Check size of the converted MP3
                    file_size_mb = os.path.getsize(converted_mp3_path) / (1024 * 1024)
                    
                    if file_size_mb > 24: # Leave a little buffer for 25MB limit
                        st.info(f"Audio file size is {file_size_mb:.2f} MB. Splitting and transcribing...")
                        transcript_text = utils.split_and_transcribe(converted_mp3_path, client)
                    else:
                        st.info(f"Audio file size is {file_size_mb:.2f} MB. Transcribing directly...")
                        transcript_text = utils.transcribe_audio(converted_mp3_path, client)
                    
                    st.success("Transcription Complete!")
                    st.session_state['transcript_text'] = transcript_text
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
                    if converted_mp3_path and os.path.exists(converted_mp3_path):
                        os.remove(converted_mp3_path)

    if 'transcript_text' in st.session_state:
        st.subheader("Transcript")
        st.text_area("Full Transcript", st.session_state['transcript_text'], height=300)
        
        st.download_button(
            label="Download Transcript",
            data=st.session_state['transcript_text'],
            file_name="transcript.txt",
            mime="text/plain"
        )
        
        st.markdown("---")
        st.subheader("Summarize Transcript")
        
        col1, col2 = st.columns(2)
        with col1:
            num_words_tr = st.number_input("Approx. words for summary", min_value=50, max_value=2000, value=200, step=50, key="num_words_tr")
        with col2:
            num_ideas_tr = st.slider("Number of key ideas", min_value=1, max_value=20, value=10, key="num_ideas_tr")
            
        if st.button("Generate Summary & Key Ideas", key="sum_tr_btn"):
            with st.spinner("Generating summary and key ideas..."):
                summary = utils.summarize_text(st.session_state['transcript_text'], num_words_tr, client)
                key_ideas = utils.extract_key_ideas(st.session_state['transcript_text'], num_ideas_tr, client)
                
                st.subheader("Summary")
                st.write(summary)
                
                st.subheader("Key Ideas")
                st.write(key_ideas)

with tab2:
    st.write("Upload a text file to summarize it.")
    uploaded_text_file = st.file_uploader("Choose a text file", type=["txt"], key="text_uploader")
    
    if uploaded_text_file is not None:
        stringio = io.StringIO(uploaded_text_file.getvalue().decode("utf-8"))
        file_text = stringio.read()
        
        st.text_area("File Content", file_text, height=200)
        
        col1, col2 = st.columns(2)
        with col1:
            num_words_txt = st.number_input("Approx. words for summary", min_value=50, max_value=2000, value=200, step=50, key="num_words_txt")
        with col2:
            num_ideas_txt = st.slider("Number of key ideas", min_value=1, max_value=20, value=10, key="num_ideas_txt")
            
        if st.button("Generate Summary & Key Ideas", key="sum_txt_btn"):
            with st.spinner("Processing..."):
                summary_txt = utils.summarize_text(file_text, num_words_txt, client)
                key_ideas_txt = utils.extract_key_ideas(file_text, num_ideas_txt, client)
                
                st.subheader("Summary")
                st.write(summary_txt)
                
                st.subheader("Key Ideas")
                st.write(key_ideas_txt)

    if 'transcript_text' in st.session_state or ('file_text' in locals() and file_text):
        st.markdown("---")
        st.header("Ask a Question 🙋")
        
        # determine which context to use
        context_text = st.session_state.get('transcript_text', "")
        if 'file_text' in locals() and file_text:
             context_text = file_text
        
        # Audio Input
        audio_question = st.audio_input("Record your question")
        # Text Input
        text_question = st.text_input("Or type your question here")
        
        if st.button("Ask Question", key="ask_btn"):
             final_question = None
             
             if audio_question:
                 with st.spinner("Transcribing your question..."):
                     # Transcribe the audio question
                     # save to temp
                     suffix = os.path.splitext(audio_question.name)[1] if audio_question.name else ".wav"
                     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_audio_q:
                         tmp_audio_q.write(audio_question.getvalue())
                         tmp_audio_q_path = tmp_audio_q.name
                     
                     try:
                         final_question = utils.transcribe_audio(tmp_audio_q_path, client)
                         st.info(f"You asked: {final_question}")
                     finally:
                         os.remove(tmp_audio_q_path)
             elif text_question:
                 final_question = text_question
                 
             if final_question:
                 with st.spinner("Thinking..."):
                     answer = utils.ask_question(context_text, final_question, client)
                     st.subheader("Answer")
                     st.write(answer)
             else:
                 st.warning("Please enter a question or record voice.")
