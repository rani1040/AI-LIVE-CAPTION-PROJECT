# 🎙️ AI Live Audio Transcriber

An AI-powered speech-to-text application built using **Python, Streamlit, MediaPipe, and Google's Parakeet TDT 0.6B V3 model**.

This application allows users to record audio through their microphone, process the recorded audio, and convert spoken words into text using an AI audio transcription model.

The project demonstrates how to integrate a locally running AI model into a simple web application and understand the basics of audio processing.

## ✨ Features

- 🎤 Record audio directly through your microphone.
- 🔊 Listen to recorded audio before transcription.
- 🤖 Convert speech into text using the Parakeet AI model.
- 🎧 Process stereo and mono audio.
- 🔄 Convert integer audio samples into normalized Float32 values.
- ⚡ Run model inference locally using CPU.
- 🖥️ Simple and interactive Streamlit interface.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Building the web interface |
| MediaPipe | Loading and running the audio transcription model |
| NumPy | Audio processing and normalization |
| SciPy | Reading WAV audio files |
| Hugging Face | Downloading the pretrained model |
| Parakeet TDT 0.6B V3 | Speech-to-text transcription |

## 🤖 AI Model

This project uses **Parakeet TDT 0.6B V3**, a pretrained audio transcription model available through Hugging Face's LiteRT Community.

**Model file:**

`parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm`

**Download the model:**

[Download Parakeet TDT 0.6B V3 from Hugging Face](https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/blob/main/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm)

After downloading, place the model file in the root directory of your project.

## 📂 Project Structure

```text
AI-LIVE-CAPTION/
│
├── app.py
├── parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd AI-LIVE-CAPTION
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**Mac/Linux:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the AI Model

Download the `.litertlm` model file from Hugging Face:

[Parakeet TDT 0.6B V3](https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/blob/main/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm)

Keep the downloaded file inside the project directory.

## 🚀 Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔄 How the Project Works

The application follows these steps:

### Step 1: Record Audio

Streamlit provides the `st.audio_input()` component to record audio through the microphone.

```python
audio_file = st.audio_input(
    "Click to record",
    sample_rate=16000
)
```

The recorded audio is captured as a WAV audio file.

### Step 2: Read the Audio File

We use `BytesIO()` and `wavfile.read()` to extract the audio data and sample rate.

```python
sample_rate, audio_data = wavfile.read(
    BytesIO(audio_file.getvalue())
)
```

- `audio_file.getvalue()` retrieves the audio file as bytes.
- `BytesIO()` converts the bytes into a file-like object.
- `wavfile.read()` extracts the sample rate and audio samples.

### Step 3: Convert Stereo Audio into Mono

Audio can have one or two channels.

| Mono | Stereo |
|---|---|
| One audio channel | Two audio channels |
| Single audio signal | Left and right audio signals |
| Shape: `(samples,)` | Shape: `(samples, 2)` |

The project converts stereo audio into mono by calculating the mean of both channels.

```python
if audio_data.ndim == 2:
    audio_data = audio_data.mean(axis=1)
```

This reduces two audio channels into one.

### Step 4: Convert Integer Audio into Float32

Audio samples can be stored as integers, but audio-processing models may expect floating-point values.

We use NumPy to convert integer audio into `float32`.

```python
if np.issubdtype(audio_data.dtype, np.integer):

    info = np.iinfo(audio_data.dtype)

    audio_data = audio_data.astype(np.float32) / max(
        abs(info.min), info.max
    )

else:
    audio_data = audio_data.astype(np.float32)
```

**Understanding the NumPy functions:**

| Function | Purpose |
|---|---|
| `np.issubdtype()` | Checks whether the data belongs to a particular data type |
| `np.integer` | Represents integer data types |
| `np.iinfo()` | Provides information about integer limits |
| `info.min` | Minimum value of the integer data type |
| `info.max` | Maximum value of the integer data type |
| `abs()` | Converts a negative number into a positive number |
| `max()` | Selects the larger value |
| `.astype(np.float32)` | Converts audio samples into Float32 |
| `/` | Divides values to normalize the audio |

**Example:**

For a signed 16-bit integer:

- Minimum value = -32768
- Maximum value = 32767
- Normalization divisor = 32768

Dividing the audio samples by 32768 scales them approximately into the range `-1.0` to `1.0`.

### Step 5: Check the Sample Rate

The model is configured to work with a sample rate of 16,000 Hz.

```python
if sample_rate != 16000:

    st.error(
        f"Expected 16000 Hz, received {sample_rate} Hz."
    )

    st.stop()
```

A sample rate of 16,000 Hz means that 16,000 audio samples are recorded every second.

### Step 6: Load the AI Model

MediaPipe provides the AudioTranscriber API to initialize and run the transcription model.

```python
options = AudioTranscriberOptions(
    base_options=BaseOptions(
        model_asset_path="parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm",
        delegate=BaseOptions.Delegate.CPU,
    ),
    sample_rate_hz=16000,
)
```

- `model_asset_path`: Specifies the downloaded model location.
- `delegate`: Specifies CPU execution.
- `sample_rate_hz`: Configures the audio sample rate.

### Step 7: Generate Transcription

Once the audio is processed, it is passed to the AI model.

```python
with AudioTranscriber.create_from_options(options) as transcriber:

    result = transcriber.transcribe(
        audio_data,
        end_of_stream=True
    )
```

- `create_from_options()` initializes the transcription model.
- `transcriber.transcribe()` processes the audio samples.
- `audio_data` contains the preprocessed audio.
- `end_of_stream=True` indicates that the audio input has ended.
- `result.text` contains the recognized speech.

### Step 8: Display the Transcription

Finally, Streamlit displays the generated text.

```python
st.subheader("📝 Transcription")
st.write(result.text)
```

## 🧠 What I Learned

While building this project, I learned:

- How to explore and select pretrained AI models.
- How to download models from Hugging Face.
- How to use MediaPipe Tasks for audio transcription.
- How microphone input works in Streamlit.
- How to read WAV audio files using SciPy.
- The difference between stereo and mono audio.
- How to process audio using NumPy.
- How to convert integer audio into Float32.
- Why audio normalization is important.
- How to integrate an AI model into a Python web application.

## 🔮 Future Improvements

- Add continuous microphone recording.
- Display live captions while speaking.
- Add support for uploading audio files.
- Support multiple languages.
- Add an option to download the generated transcript.
- Improve the user interface.

## 👩‍💻 Author

**Rani Rathore**

Built with Python, AI, and a curiosity to learn how speech recognition works!

---

⭐ If you find this project interesting, consider giving this repository a star!
