import streamlit as st
import torch
import torchaudio
import requests
import json

from BEATs.BEATs import BEATs, BEATsConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
url = "http://tinyurl.com/5a3b7ec5"
# url = "http://tinyurl.com/4mz5ydpj"


def download_model(url, to_path):
    data = requests.get(url)
    with open(to_path, 'wb') as file:
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


def post_process(pred, k, thresh):
    topk_pred = pred.topk(k=k)
    mask = (topk_pred.values >= thresh)
    if True not in mask:
        mask[0][0] = True
    probs = topk_pred.values[mask].tolist()
    preds = topk_pred.indices[mask].tolist()
    return [probs, preds]


def inference(model, data):
    with torch.no_grad():
        pred = model.extract_features(data, padding_mask=None)[0]
    return pred


def main():
    st.title('Sound Classification app')

    download_model(url, 'model.pt')

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
    k = int(st.number_input("K (number of wanted results): ", value=5))
    thresh = int(st.number_input("Treshold (percentage between 0 and 100)**: ", value=0)) / 100
    st.write("** if no prob is greater than the treshold, it returns the one most probable result")
    if uploaded_file:

        st.audio(uploaded_file, format="audio/wav", start_time=0)

        if st.button('Predict'):
            model = load_model('model.pt')
            audio_data = pre_process(uploaded_file)
            pred = inference(model, audio_data)
            label_pred = post_process(pred, k=k, thresh=thresh)

            with open('labels.json', 'r') as file:
                labels = json.load(file)
                for i in range(len(label_pred[0])):
                    st.write(f'Label: {labels[str(label_pred[1][i])]} ({label_pred[1][i]}) - Prob: {int(round(label_pred[0][i], 2) * 100)}%')


if __name__ == "__main__":
    main()
