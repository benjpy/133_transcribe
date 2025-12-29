# AI Transcription & Summarization App 🎙️

A powerful Streamlit application that transcribes video and audio files using OpenAI's Whisper model, summarizes the content, and allows you to ask questions about the transcript.

## Features

-   **Transcription**: Upload `mp3`, `mp4`, `wav`, `m4a` files. Large files are automatically converted and chunked to handle API limits.
-   **Summarization**: Generate concise summaries and extract key ideas/bullet points from your transcripts or uploaded text files.
-   **Q&A**: Ask questions about your transcript using **Text** or **Voice**. The app uses the transcript context to provide accurate answers.
-   **Optimization**: Automatically converts video to audio (MP3) before processing to save bandwidth and API costs.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/benjpy/transcribe-streamlit-app.git
    cd transcribe-streamlit-app
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Set up your OpenAI API Key:
    -   Create a `.env` file in the root directory:
        ```
        OPENAI_API_KEY=sk-your-api-key
        ```

4.  Run the app:
    ```bash
    streamlit run app.py
    ```

## deployment on Streamlit Cloud

1.  Push this code to your GitHub repository.
2.  Go to [Streamlit Cloud](https://share.streamlit.io/).
3.  Connect your GitHub account and select this repository.
4.  **Important**: In the specific app settings on Streamlit Cloud, go to **Secrets** and add your API key:
    ```toml
    OPENAI_API_KEY = "sk-your-api-key"
    ```
5.  Click **Deploy**!

## Requirements

-   Python 3.8+
-   `ffmpeg` (Required for audio processing. Included in `packages.txt` for Streamlit Cloud).
