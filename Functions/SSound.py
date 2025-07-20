import streamlit as st
from gtts import gTTS
import os
import uuid
import base64

def SSound(text, lang='en'):
    # Generate a unique filename
    filename = f"tts_audio_{uuid.uuid4().hex}.mp3"

    # Convert text to speech using gTTS
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filename)

    # Read the audio file for both playback and download
    with open(filename, 'rb') as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')

        # Generate download link
        b64 = base64.b64encode(audio_bytes).decode()
        href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">📥 Download Audio</a>'
        st.markdown(href, unsafe_allow_html=True)

    # Optionally delete file afterwards
    os.remove(filename)
