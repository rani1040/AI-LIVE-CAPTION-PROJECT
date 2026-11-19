
import mediapipe as mp
import streamlit as st
import numpy as np

from scipy.io import wavfile
from io import BytesIO

# STEP 1: STREAMLIT CONFIGURATION

st.set_page_config(
    page_title="Live Captions",
    layout="centered"
)

st.title("🎙️ Live Audio Transcriber")

# STEP 2: MICROPHONE INPUT
# it will give you file with name and id and its type--audio wav
audio_file = st.audio_input(
    "Click to record",
    sample_rate=16000
)

# STEP 3: MEDIAPIPE SETUP

BaseOptions = mp.tasks.BaseOptions

AudioTranscriber = mp.tasks.audio.AudioTranscriber

AudioTranscriberOptions = mp.tasks.audio.AudioTranscriberOptions

options = AudioTranscriberOptions(
    base_options=BaseOptions(
        model_asset_path="parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm",
        delegate=BaseOptions.Delegate.CPU,
    ),
    sample_rate_hz=16000,
)

# STEP 4: CHECK AUDIO

if audio_file:

    st.success("Audio recorded successfully!")

    st.audio(audio_file)

    if st.button("Transcribe!!", type="primary"):

        # Read audio file -- samplerate--16000 and audio_data=[ 0 0 0 ... -1026 1772 2108]
        sample_rate, audio_data = wavfile.read(
            BytesIO(audio_file.getvalue())
        )

        
       
        st.warning(audio_data.ndim)
        # Convert stereo to mono
        if audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=1)

        # Convert integer audio into float32
        if np.issubdtype(audio_data.dtype, np.integer):

            info = np.iinfo(audio_data.dtype)

            audio_data = audio_data.astype(np.float32) / max(
                abs(info.min), info.max
            )

        else:
            audio_data = audio_data.astype(np.float32)

        # Check sample rate
        if sample_rate != 16000:

            st.error(
                f"Expected 16000 Hz, received {sample_rate} Hz."
            )

            st.stop()

        # STEP 5: TRANSCRIBE AUDIO

        with st.spinner("Transcribing audio..."):

            with AudioTranscriber.create_from_options(options) as transcriber:

                result = transcriber.transcribe(
                    audio_data,
                    end_of_stream=True
                )

            # STEP 6: DISPLAY RESULT

            st.subheader("📝 Transcription")

            st.write(result.text)

else:

    st.info("Please record some audio first.")
