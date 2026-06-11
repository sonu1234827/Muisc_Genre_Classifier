import os
import uuid
import sys
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import numpy as np
import librosa
import soundfile as sf

# ensure unicode prints fine on Windows consoles
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Matplotlib non-GUI backend to avoid Tkinter thread errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

# TensorFlow
import tensorflow as tf
from tensorflow.image import resize

# -------------------------
# Flask app config
# -------------------------
app = Flask(__name__)
app.secret_key = "change_this_to_a_random_secret_for_prod"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
SPEC_DIR = os.path.join(BASE_DIR, "static", "spectrograms")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SPEC_DIR, exist_ok=True)

# -------------------------
# Load model once
# -------------------------
MODEL_PATH = os.path.join(BASE_DIR, "model", "Trained_Music_Genre_Classifier.h5")
print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("[OK] Model loaded.")

# -------------------------
# Labels & mappings
# -------------------------
classes = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
mood_map = {
    "rock": "Energetic ⚡",
    "jazz": "Chill 🎷",
    "pop": "Upbeat 💃",
    "classical": "Calm 🎻",
    "hiphop": "Groovy 🎤",
    "metal": "Intense 🔥",
    "country": "Relaxed 🤠",
    "disco": "Vibrant 🕺",
    "blues": "Melancholic 💙",
    "reggae": "Laid-back 🌴"
}

# -------------------------
# Helpers
# -------------------------
def unique_name(orig_name):
    name = f"{uuid.uuid4().hex[:8]}_{orig_name.replace(' ', '_')}"
    return name

def save_uploaded_file(file_storage, dest_folder=UPLOAD_DIR):
    fn = unique_name(file_storage.filename)
    out_path = os.path.join(dest_folder, fn)
    file_storage.save(out_path)
    return fn, out_path

def generate_spectrogram_image(audio_path, out_png_path):
    """
    Generates mel-spectrogram PNG (log scaled) saved at out_png_path.
    max_duration: shorten for speed
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(8, 3.6))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(out_png_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        return True
    except Exception as e:
        print("Spectrogram generation failed:", e)
        return False

def preprocess_for_model(file_path, target_shape=(150, 150)):
    """
    Loads the audio (max_duration seconds), splits into overlapping 4s chunks with 2s overlap,
    computes mel spectrograms, normalizes and resizes to target_shape.
    Returns a numpy array of shape (n_chunks, h, w, 1).
    """
    y, sr = librosa.load(file_path, sr=None)
    chunk_duration = 4  # seconds
    overlap = 2  # seconds
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)

    # pad if shorter than chunk
    if len(y) < chunk_samples:
        y = np.pad(y, (0, chunk_samples - len(y)), mode='constant')

    if len(y) <= chunk_samples:
        num_chunks = 1
    else:
        num_chunks = int(np.ceil((len(y) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    arr = []
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = y[start:end]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, max(0, chunk_samples - len(chunk))), mode='constant')
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # normalize to 0-1
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        mel_norm = np.expand_dims(mel_norm, axis=-1).astype(np.float32)  # (freq, time, 1)
        mel_resized = resize(mel_norm, target_shape).numpy()
        arr.append(mel_resized)
    if arr:
        return np.array(arr)
    return np.empty((0, *target_shape, 1), dtype=np.float32)

# -------------------------
# Model Prediction Function
# -------------------------
def model_prediction(X_test):
    y_pred = model.predict(X_test, batch_size=4)
    predicted_categories = np.argmax(y_pred, axis=1)
    
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    total = np.sum(counts)
    
    percentages = (counts / total) * 100
    sorted_indices = np.argsort(percentages)[::-1]
    unique_elements = unique_elements[sorted_indices]
    counts = counts[sorted_indices]
    percentages = percentages[sorted_indices]
    
    # print detailed distribution in console
    print("\n🎵 Genre Prediction Confidence Distribution (Descending Order):")
    for label, pct, count in zip(unique_elements, percentages, counts):
        print(f"  {classes[label]:<15} --> {pct:.2f}% ({count} samples)")
    
    predicted_genre = classes[unique_elements[0]]
    print("\n✅ Final Decision:")
    print(f"  The model predicts **{predicted_genre.upper()}** "
          f"as the most likely genre ({percentages[0]:.2f}% confidence).")
    
    return unique_elements, percentages
# -------------------------
# Session History Management
def add_history(entry):
    history = session.get('history', [])
    history.insert(0, entry)   # newest first
    session['history'] = history[:40]  # limit history length

# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    # clear any current upload pointer (optional)
    session.pop('current_upload', None)
    return render_template('index.html')

@app.route('/upload_preview', methods=['POST'])
def upload_preview():
    """
    Endpoint used when user picks a file and clicks 'Upload & Preview'.
    Saves the file, generates spectrogram image (fast), and returns JSON with URLs.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file sent"}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    saved_name, saved_path = save_uploaded_file(f, UPLOAD_DIR)
    spec_name = f"spec_{saved_name.rsplit('.',1)[0]}.png"
    spec_path = os.path.join(SPEC_DIR, spec_name)
    ok = generate_spectrogram_image(saved_path, spec_path)
    # store current upload filename in session so /predict can use it
    session['current_upload'] = saved_name

    audio_url = url_for('static', filename=f"uploads/{saved_name}")
    spec_url = url_for('static', filename=f"spectrograms/{spec_name}") if ok else None
    return jsonify({"audio_url": audio_url, "spectrogram_url": spec_url, "filename": saved_name})

@app.route('/predict', methods=['POST'])
def predict():
    filename = session.get('current_upload', None)
    if 'file' in request.files:
        f = request.files['file']
        filename, saved_path = save_uploaded_file(f, UPLOAD_DIR)
        session['current_upload'] = filename

    if not filename:
        return jsonify({"error": "No file available to predict"}), 400

    audio_path = os.path.join(UPLOAD_DIR, filename)
    spec_name = f"spec_{filename.rsplit('.',1)[0]}.png"
    spec_path = os.path.join(SPEC_DIR, spec_name)
    if not os.path.exists(spec_path):
        generate_spectrogram_image(audio_path, spec_path)

    X = preprocess_for_model(audio_path, target_shape=(150,150))
    if X.shape[0] == 0:
        return jsonify({"error": "Audio preprocessing failed"}), 500

    # ----------------------------
    # Use new model_prediction logic
    # ----------------------------
    unique_elements, percentages = model_prediction(X)

    results = []
    for i, label_idx in enumerate(unique_elements):
        results.append({
            "genre": classes[label_idx],
            "percentage": float(percentages[i]),
            "count": int((percentages[i] / 100) * X.shape[0])
        })

    final_genre = classes[unique_elements[0]]
    mood = mood_map.get(final_genre, "Neutral")

    audio_url = url_for('static', filename=f"uploads/{filename}")
    spectrogram_url = url_for('static', filename=f"spectrograms/{spec_name}")
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
    entry = {
        "id": uuid.uuid4().hex[:8],
        "filename": filename,
        "audio_url": audio_url,
        "spectrogram_url": spectrogram_url,
        "final_genre": final_genre,
        "mood": mood,
        "results": results,
        "timestamp": timestamp
    }
    session['last_prediction'] = entry
    add_history(entry)

    return jsonify({
        "ok": True,
        "audio_url": audio_url,
        "spectrogram_url": spectrogram_url,
        "final_genre": final_genre,
        "mood": mood,
        "results": results,
        "timestamp": timestamp
    })

@app.route('/results')
def results_page():
    pred = session.get('last_prediction')
    if not pred:
        return redirect(url_for('index'))
    return render_template('results.html', pred=pred)

@app.route('/history')
def history_page():
    hist = session.get('history', [])
    return render_template('history.html', history=hist)

# simple API to fetch history JSON (optional)
@app.route('/api/history')
def api_history():
    return jsonify(session.get('history', []))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)