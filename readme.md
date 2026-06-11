A Deep Learning-based Music Genre Classification System using CNNs, Mel Spectrograms, TensorFlow, and Flask.

GenreLens is a Deep Learning-based Music Genre Classification System that analyzes audio files, generates Mel Spectrograms, and predicts the music genre using a Convolutional Neural Network (CNN).

The project provides a modern Flask-based web application where users can upload MP3/WAV files, visualize spectrograms, classify genres, and view prediction history.

---

## 🚀 Features

* Upload MP3 or WAV audio files
* Automatic Mel Spectrogram generation
* Deep Learning-based genre prediction
* Confidence distribution visualization
* Interactive Bar Chart and Pie Chart
* Session History Tracking
* Modern responsive UI (GenreLens)
* Audio Preview Support
* Spectrogram Visualization

---

## 🎯 Supported Genres

The model classifies music into the following 10 genres:

* Blues
* Classical
* Country
* Disco
* Hip-Hop
* Jazz
* Metal
* Pop
* Reggae
* Rock

---

## 🧠 Deep Learning Approach

### Audio Preprocessing

The uploaded audio is:

1. Loaded using Librosa
2. Split into overlapping 4-second chunks
3. Converted into Mel Spectrograms
4. Normalized and resized to 150 × 150
5. Passed to the CNN model

### Model Architecture

The system uses a Convolutional Neural Network (CNN) trained on Mel Spectrogram images.

Layers include:

* Conv2D
* MaxPooling2D
* Dropout
* Dense Layers
* Softmax Output Layer

The final output predicts the probability distribution across all 10 genres.

---

## 📊 Dataset

Dataset Used:

**GTZAN Music Genre Dataset**

Dataset Characteristics:

* 1000 Audio Samples
* 10 Genres
* 100 Songs per Genre
* 30 Seconds per Audio Sample

Genres included:

* Blues
* Classical
* Country
* Disco
* Hip-Hop
* Jazz
* Metal
* Pop
* Reggae
* Rock

---

## 🛠️ Tech Stack

### Backend

* Python
* Flask

### Deep Learning

* TensorFlow
* Keras

### Audio Processing

* Librosa
* SoundFile

### Visualization

* Matplotlib
* Chart.js

### Frontend

* HTML
* CSS
* JavaScript

---

## 📂 Project Structure

```text
MUSIC GENRE/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   └── Trained_Music_Genre_Classifier.h5
│
├── notebooks/
│   ├── train_music_genre_classifier.ipynb
│   └── music_genre_testing.ipynb
│
├── scripts/
│   ├── train.py
│   └── test.py
│
├── templates/
│   ├── index.html
│   ├── results.html
│   └── history.html
│
├── static/
│   ├── uploads/
│   └── spectrograms/
│
└── screenshots/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/music-genre-classifier.git
cd music-genre-classifier
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

in your browser.

---

## 📈 Workflow

```text
Audio File
      ↓
Librosa Audio Loading
      ↓
Chunk Generation
      ↓
Mel Spectrogram Creation
      ↓
Image Resizing
      ↓
CNN Model
      ↓
Genre Prediction
      ↓
Confidence Distribution
      ↓
Result Visualization
```

---

## 📸 Screenshots

### Home Page

Add screenshot here:

```text
screenshots/home.png
```

### Prediction Result

Add screenshot here:

```text
screenshots/result.png
```

### Session History

Add screenshot here:

```text
screenshots/history.png
```

---

## 🔮 Future Improvements

* Real-time music genre detection
* Additional music genres
* Spotify integration
* Mobile application support
* Transformer-based audio models
* Multi-label genre classification

---

## 👨‍💻 Author

**Sonu Dalai**

B.Tech Computer Science & Engineering

Deep Learning | Machine Learning | Web Development

---

## 📜 License

This project is developed for educational and research purposes.
