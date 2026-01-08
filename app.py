import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import io
import utils

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = utils.get_gemini_client()

if not client:
    st.error("GEMINI_API_KEY not found. Please set it in .env or Streamlit Cloud Secrets.")
    st.stop()

st.set_page_config(page_title="Gemini Transcriber", page_icon="🎙️")

st.title("Gemini Video/Audio Transcriber 🎙️ 📝")

tab1, tab2 = st.tabs(["Transcribe Audio/Video", "Summarize Text File"])

with tab1:
    st.write("Process video or audio files using Google Gemini.")
    
    input_type = st.radio("Choose input source:", ["Upload File", "URL (YouTube, etc.)"])
    
    media_source = None
    is_url = False
    
    if input_type == "Upload File":
        uploaded_file = st.file_uploader("Choose a media file", type=["mp3", "mp4", "wav", "m4a", "mpeg", "mpga", "webm"], key="media_uploader")
        if uploaded_file is not None:
            # Save uploaded file to temp file
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                media_source = tmp_file.name
    else:
        url_input = st.text_input("Enter Media URL (e.g., YouTube URL):")
        if url_input:
            media_source = url_input
            is_url = True

    st.markdown("### Options")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        show_summary = st.checkbox("Summary on top", value=True)
        show_diarization = st.checkbox("Diarization (Speaker IDs)", value=True)
    with col_opt2:
        show_timestamps = st.checkbox("Timestamps", value=True)
        show_simple = st.checkbox("Simple Transcript", value=False)

    if media_source:
        if st.button("Process Media", key="process_btn"):
            with st.spinner("Processing with Gemini..."):
                try:
                    # If it's a URL, we might need to download it if Gemini doesn't support it directly
                    # The template suggested YOUTUBE_URL can be used directly, but usually it requires local file.
                    # Let's try to download it if it's a URL to be safe, or just pass it if Gemini 2.x handles it.
                    # Actually, let's use the local file approach to ensure it works.
                    
                    actual_path = media_source
                    if is_url:
                        st.info("Downloading audio from URL...")
                        actual_path = utils.download_youtube_audio(media_source)
                    
                    results = utils.process_media_with_gemini(actual_path, client, is_url=False)
                    
                    st.success("Processing Complete!")
                    st.session_state['gemini_results'] = results
                    
                    # Clean up temp file if we downloaded it
                    if is_url and os.path.exists(actual_path):
                        os.remove(actual_path)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    # Clean up uploaded temp file
                    if not is_url and media_source and os.path.exists(media_source):
                        os.remove(media_source)

    if 'gemini_results' in st.session_state:
        results = st.session_state['gemini_results']
        
        if show_summary:
            st.subheader("Summary")
            st.write(results.summary)
            st.markdown("---")

        st.subheader("Transcript")
        
        full_transcript_text = ""
        
        if show_simple:
            simple_text = " ".join([seg.content for seg in results.segments])
            st.text_area("Simple Transcript", simple_text, height=300)
            full_transcript_text = simple_text
        else:
            formatted_segments = []
            for seg in results.segments:
                line = ""
                if show_timestamps:
                    line += f"[{seg.timestamp}] "
                if show_diarization:
                    line += f"**{seg.speaker}**: "
                line += seg.content
                formatted_segments.append(line)
            
            transcript_display = "\n\n".join(formatted_segments)
            st.markdown(transcript_display)
            full_transcript_text = "\n".join(formatted_segments)

        st.download_button(
            label="Download Transcript",
            data=full_transcript_text,
            file_name="transcript.txt",
            mime="text/plain"
        )
        
        # Store for "Ask a Question"
        st.session_state['transcript_text'] = full_transcript_text

with tab2:
    st.write("Upload a text file to summarize it using Gemini.")
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
        
        context_text = st.session_state.get('transcript_text', "")
        if 'file_text' in locals() and file_text:
             context_text = file_text
        
        text_question = st.text_input("Type your question here about the content:")
        
        if st.button("Ask Question", key="ask_btn"):
             if text_question:
                 with st.spinner("Thinking..."):
                     answer = utils.ask_question(context_text, text_question, client)
                     st.subheader("Answer")
                     st.write(answer)
             else:
                 st.warning("Please enter a question.")
