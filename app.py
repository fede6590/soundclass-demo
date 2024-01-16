import streamlit as st
import torch
import torchaudio
import requests
import os
import tempfile

from BEATs.BEATs import BEATs, BEATsConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
url = "http://tinyurl.com/4mz5ydpj"


def download_model(url, to_path):
    data = requests.get(url)
    with open(to_path, 'wb')as file:
        file.write(data.content)


def load_model(location):
    global model
    if model is None:
        checkpoint = torch.load(location)
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    return model.to(device)


def pre_process(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    return waveform.to(device)


def inference(model, data):
    with torch.no_grad():
        pred = model.extract_features(data, padding_mask=None)[0]
    return pred


def main():
    st.title('Audio Classification App')

    download_model(url, 'model.pt')
    while not os.path.isfile('model.pt'):
        st.write('Downloading the model...')

    st.write("Upload an audio file for classification")
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            audio_bytes = temp_file.read()
            temp_file.write(audio_bytes)

        st.audio(temp_file, format="audio/wav", start_time=0)

        if st.button('Classify'):
            model = load_model('model.pt')
            audio_data = pre_process(temp_file.name)
            prediction = inference(model, audio_data)
            st.write(f'Predicted class: {prediction}')
            os.unlink(temp_file.name)

if __name__ == "__main__":
    main()
