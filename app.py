import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template,send_file
from tensorflow.keras.models import load_model
import joblib
import pickle
import librosa
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

def extract_features(filename):
    y,sr = librosa.load(filename,duration =30)
    length = 0.0
    chroma_stft_mean= librosa.feature.chroma_stft(y=y,sr =sr).mean()
    chroma_stft_var= librosa.feature.chroma_stft(y=y,sr =sr).var()
    rms_mean = librosa.feature.rms(y=y).mean()
    rms_var = librosa.feature.rms(y=y).var()
    spectral_centroid_mean = librosa.feature.spectral_centroid(y=y,sr =sr).mean()
    spectral_centroid_var =librosa.feature.spectral_centroid(y=y,sr =sr).var()
    spectral_bandwidth_mean =librosa.feature.spectral_bandwidth(y=y,sr =sr).mean()
    spectral_bandwidth_var =librosa.feature.spectral_bandwidth(y=y,sr =sr).var()
    rolloff_mean = librosa.feature.spectral_rolloff(y=y,sr =sr).mean()
    rolloff_var = librosa.feature.spectral_rolloff(y=y,sr =sr).var()
    zero_crossing_rate_mean = librosa.feature.zero_crossing_rate(y=y).mean()
    zero_crossing_rate_var = librosa.feature.zero_crossing_rate(y=y).mean()
    y_harmonic = librosa.effects.harmonic(y = y)
    harmony_mean =  np.mean(y_harmonic)
    harmony_var =  np.var(y_harmonic)
    y_percussive = librosa.effects.percussive(y)
    perceptr_mean = np.mean(y_percussive)
    perceptr_var = np.var(y_percussive)
    tempo, _ = librosa.beat.beat_track(y=y,sr = sr)
    mfccs = librosa.feature.mfcc(y = y, sr = sr,n_mfcc =20)
    
    mfcc1_mean = mfccs[0].mean()
    mfcc1_var = mfccs[0].var()
    mfcc2_mean = mfccs[1].mean()
    mfcc2_var = mfccs[1].var()
    mfcc3_mean = mfccs[2].mean()
    mfcc3_var = mfccs[2].var()
    mfcc4_mean = mfccs[3].mean()
    mfcc4_var = mfccs[3].var()
    mfcc5_mean = mfccs[4].mean()
    mfcc5_var = mfccs[4].var()
    mfcc6_mean = mfccs[5].mean()
    mfcc6_var = mfccs[5].var()
    mfcc7_mean = mfccs[6].mean()
    mfcc7_var = mfccs[6].var()
    mfcc8_mean = mfccs[7].mean()
    mfcc8_var = mfccs[7].var()
    mfcc9_mean = mfccs[8].mean()
    mfcc9_var = mfccs[8].var()
    mfcc10_mean = mfccs[9].mean()
    mfcc10_var = mfccs[9].var()
    mfcc11_mean = mfccs[10].mean()
    mfcc11_var = mfccs[10].var()
    mfcc12_mean = mfccs[11].mean()
    mfcc12_var = mfccs[11].var()
    mfcc13_mean = mfccs[12].mean()
    mfcc13_var = mfccs[12].var()
    mfcc14_mean = mfccs[13].mean()
    mfcc14_var = mfccs[13].var()
    mfcc15_mean = mfccs[14].mean()
    mfcc15_var = mfccs[14].var()
    mfcc16_mean = mfccs[15].mean()
    mfcc16_var = mfccs[15].var()
    mfcc17_mean = mfccs[16].mean()
    mfcc17_var = mfccs[16].var()
    mfcc18_mean = mfccs[17].mean()
    mfcc18_var = mfccs[17].var()
    mfcc19_mean = mfccs[18].mean()
    mfcc19_var = mfccs[18].var()
    mfcc20_mean = mfccs[19].mean()
    mfcc20_var = mfccs[19].var()
    
    features = np.array([length, chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, spectral_centroid_mean, spectral_centroid_var
                       , spectral_bandwidth_mean, spectral_bandwidth_var, rolloff_mean, rolloff_var, zero_crossing_rate_mean, 
                       zero_crossing_rate_var, harmony_mean, harmony_var, perceptr_mean, perceptr_var, tempo, mfcc1_mean, 
                       mfcc1_var, mfcc2_mean, mfcc2_var, mfcc3_mean, mfcc3_var, mfcc4_mean, mfcc4_var, mfcc5_mean, mfcc5_var, 
                       mfcc6_mean, mfcc6_var, mfcc7_mean, mfcc7_var, mfcc8_mean, mfcc8_var, mfcc9_mean, mfcc9_var, mfcc10_mean, 
                       mfcc10_var,mfcc11_mean, mfcc11_var,mfcc12_mean, mfcc12_var, mfcc13_mean,  mfcc13_var, mfcc14_mean,mfcc14_var, 
                       mfcc15_mean, mfcc15_var, mfcc16_mean,  mfcc16_var, mfcc17_mean, mfcc17_var, mfcc18_mean, mfcc18_var, mfcc19_mean,
                       mfcc19_var, mfcc20_mean, mfcc20_var])                      
    return features

# for wav files
# Load the trained model and label encoder
model = joblib.load("./classifier.pickle")  # Make sure to have the correct path to your model file
label_encoder = joblib.load("label_encoder.pickle")
labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


# Load the saved CNN model
model1 = load_model("model1.h5")

# Define class labels (replace with your actual class labels)
CLASS_NAMES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


@app.route('/', methods=['GET'])
def index():
    # Render the HTML file located in the "templates" folder
    return render_template('index.html')

def preprocess_image(image_path):
    image = cv2.imread(image_path)  # original image
    # resizing image
    new_size = (400, 400)
    resized_image = cv2.resize(image, new_size)
    # converting into array and expanding dimensions
    img_array = np.expand_dims(resized_image, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the file to a temporary location
        image_path = 'temp_image.png'
        file.save(image_path)
        
        # Preprocess the image
        img_array = preprocess_image(image_path)
        
        # Make predictions
        predictions = model1.predict(img_array)
        predicted_label = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_label]
        
        os.remove(image_path)  # Remove the temporary image file
        
        return jsonify({'predicted_class': predicted_class})

@app.route('/upload/audio', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded file to a temporary location
        temp_path = "temp_audio.wav"
        file.save(temp_path)

        # Extract features from the uploaded audio file
        features = extract_features(temp_path)
        features = features.reshape(1, -1)
        scaler_path = "normalized.pickle"
        with open(scaler_path, 'rb') as file:
              scaler = joblib.load(file)

        scaled_array_data = scaler.transform(features)

        # Make predictions
        numerical_preds = model.predict(scaled_array_data)
        print(numerical_preds)
        predicted_label = labels[numerical_preds[0]]
        print(predicted_label)
        # Return the predicted label along with the audio file
        return jsonify({'predicted_label': predicted_label, 'audio_url': '/audio'})


@app.route('/audio')
def get_audio():
    # Return the audio file as a response
    return send_file("temp_audio.wav", mimetype="audio/wav", as_attachment=True, download_name="audio.wav")


if __name__ == '__main__':
    app.run(debug=True)


