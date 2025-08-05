import streamlit as st
import subprocess
import base64

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

st.set_page_config(page_title="SignSpeak", layout="centered")

st.markdown("""
    <style>
        .main-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .logo {
            margin-bottom: 20px;
        }

        .title {
            font-size: 2.8em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
            color: #212121;
        }

        .subtitle {
            font-size: 1.2em;
            text-align: center;
            margin-bottom: 30px;
        }

        .button-style {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }

        .footer {
            margin-top: 10px;
            text-align: center;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-container">', unsafe_allow_html=True)


image_base64 = get_image_base64("SignSpeak_logo.jpg")
st.markdown(
    f"""
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{image_base64}' width='200'/>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">SignSpeak - ISL Hand Sign Recognizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle"> AI based Real-time Indian Sign Language interpreter</div>', unsafe_allow_html=True)

st.markdown('<div class="button-style">', unsafe_allow_html=True)
if st.button("Start Webcam"):
    try:
        subprocess.Popen(["python", "handrecogniser.py"])
        st.success("Webcam Opening. Please wait for few seconds. Press 'q' to quit.")
    except Exception as e:
        st.error(f"Failed to launch webcam: {e}")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <strong>Made by : </strong><br>
    Deepak | Diksha | Shreya | Shifa
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
