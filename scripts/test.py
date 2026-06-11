# %% [markdown]
# ### Importing Libraries

# %%
import os
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.image import resize

# %%
#Loading Model
model = tf.keras.models.load_model("./model/Trained_Music_Genre_Classifier.h5")

# %%
classes = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

# %% [markdown]
# ### Single Audio Preprocessing

# %%
# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
                
    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    # Iterate over each chunk
    for i in range(num_chunks):
                    # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
                    
                    # Extract the chunk of audio
        chunk = audio_data[start:end]
                    
                    # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                    
                #mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)

# %% [markdown]
# ### Playing a sound

# %%
from IPython.display import Audio
file_path = "./dataset/bensound-jazz-lefacteur.mp3"
y, sr = librosa.load(file_path, sr=44100)
Audio(data=y, rate=sr)

# %%
#Processing Test File
X_test = load_and_preprocess_data(file_path)

# %% [markdown]
# ### Model Prediction

# %%
# Model Prediction with Percentage Confidence (Sorted in Descending Order)
def model_prediction(X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    total = np.sum(counts)
    
    # Compute percentage for each genre
    percentages = (counts / total) * 100
    
    # Sort by descending percentage
    sorted_indices = np.argsort(percentages)[::-1]
    unique_elements = unique_elements[sorted_indices]
    counts = counts[sorted_indices]
    percentages = percentages[sorted_indices]
    
    # Print the detailed percentage distribution
    print("\n🎵 Genre Prediction Confidence Distribution (Descending Order):")
    for label, pct, count in zip(unique_elements, percentages, counts):
        print(f"  {classes[label]:<15} --> {pct:.2f}% ({count} samples)")
    
    # Determine the genre with the highest percentage
    predicted_genre = classes[unique_elements[0]]
    
    print("\n✅ Final Decision:")
    print(f"  The model predicts **{predicted_genre.upper()}** "
          f"as the most likely genre ({percentages[0]:.2f}% confidence).")
    
    return unique_elements[0]

# %%
# Model Prediction
c_index = model_prediction(X_test)
print(f"\n🎧 Model Final Prediction :: Music Genre --> {classes[c_index]}")


