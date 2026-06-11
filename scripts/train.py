# %% [markdown]
# Import Libraries

# %%
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# %% [markdown]
# Visualizing single audio

# %%
random_file_name = "../dataset/audio/blues/blues.00000.wav"

# %%
y,sr = librosa.load(random_file_name, sr=44100)
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)

# %% [markdown]
# Playing a sound

# %%
from IPython.display import Audio
Audio(data=y, rate=sr)

# %% [markdown]
# Doing visualization on chunks of audio

# %%
audio_path = "../dataset/audio/blues/blues.00000.wav"
y, sr = librosa.load(random_file_name, sr=None) #sr=None to preserve the original sampling rate

#Define the duration of each chunk and overlap
chunk_duration = 4
overlap_duration = 2

#Convert durations to samples
chunk_samples = chunk_duration * sr
overlap_samples = overlap_duration * sr

#Calculate the number of chunks
num_chunks = int(np.ceil((len(y) - overlap_samples) / (chunk_samples - overlap_samples)))+1

#Iterate over each chunks
for i in range(num_chunks):
    #Start and end sample indices
    start = i*(chunk_samples - overlap_samples)
    end = start+chunk_samples
    #Extract the chunk audio
    chunk = y[start:end]
    plt.figure(figsize=(4, 2))
    librosa.display.waveshow(chunk, sr=sr)
    plt.show()

# %% [markdown]
# Melspectrogram Visualization

# %%
#Plotting Melspectrogram of entire audio
def plot_melspectrogram(y, sr):
    #Compute spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    #Convert to decibels (log scale)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    #Visualize the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%2.0f dB')
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.show()

# %%
random_file_name = "../dataset/audio/blues/blues.00000.wav"
y,sr = librosa.load(random_file_name, sr=44100)
plot_melspectrogram(y, sr)

# %%
def plot_melspectrogram_chunks(y,sr):
    #Define the duration of each chunk and overlap
    chunk_duration = 4
    overlap_duration = 2

    #Convert durations to samples
    chunk_samples = chunk_duration * sr
    overlap_samples = overlap_duration * sr

    #Calculate the number of chunks
    num_chunks = int(np.ceil((len(y) - overlap_samples) / (chunk_samples - overlap_samples)))+1

    #Iterate over each chunks
    for i in range(num_chunks):
        #Start and end sample indices
        start = i*(chunk_samples - overlap_samples)
        end = start+chunk_samples
        #Extract the chunk audio
        chunk = y[start:end]
        #Melspectrogram part
        spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sr)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        #Visualize the spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%2.0f dB')
        plt.title("Spectrogram")
        plt.tight_layout()
        plt.show()


# %%
random_file_name = "../dataset/audio/blues/blues.00000.wav"
y,sr = librosa.load(random_file_name, sr=44100)
plot_melspectrogram_chunks(y, sr)

# %% [markdown]
# Audio Preprocessing - Final

# %%
#define your folder structure
data_dir = "../dataset/audio"
classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# %%
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.image import resize

# Extended function to save spectrograms as images
def load_and_preprocess_data(data_dir, classes, target_shape=(150,150), output_dir="../dataset/spectrograms"):
    data = []
    labels = []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i_class, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        save_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(save_class_dir, exist_ok=True)

        print("Processing--", class_name)

        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir, filename)
                audio_data, sample_rate = librosa.load(file_path, sr=None)

                # Chunking
                chunk_duration = 4
                overlap_duration = 2
                chunk_samples = chunk_duration * sample_rate
                overlap_samples = overlap_duration * sample_rate
                num_chunks = int(np.ceil((len(audio_data)-chunk_samples)/(chunk_samples-overlap_samples)))+1

                for i in range(num_chunks):
                    start = i*(chunk_samples-overlap_samples)
                    end = start+chunk_samples
                    chunk = audio_data[start:end]

                    # Mel-spectrogram
                    mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)

                    # Resize for model input
                    mel_spectrogram_resized = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)

                    # Add to memory dataset
                    data.append(mel_spectrogram_resized)
                    labels.append(i_class)

                    # --- Save as image ---
                    plt.figure(figsize=(2.56, 2.56), dpi=100)   # 256 x 256 pixels
                    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), sr=sample_rate, x_axis=None, y_axis=None)
                    plt.axis("off")

                    # Build filename
                    img_filename = f"{os.path.splitext(filename)[0]}_chunk{i}.png"
                    img_path = os.path.join(save_class_dir, img_filename)

                    plt.savefig(img_path, bbox_inches=None, pad_inches=0)
                    plt.close()

    data = np.array(data)
    labels = np.array(labels, dtype=np.int32)

    print("Final data shape:", data.shape)     # (num_samples, 150, 150, 1)
    print("Final labels shape:", labels.shape) # (num_samples,)

    # Save as .npy
    np.save("data.npy", data)
    np.save("labels.npy", labels)
    print("✅ Saved data.npy and labels.npy")

    return data, labels


# %%
data,labels = load_and_preprocess_data(data_dir,classes)

# %%
data = np.load("data.npy", allow_pickle=True)
data.shape

# %%
labels  = np.load("labels.npy", allow_pickle=True)
labels.shape

# %%
from tensorflow.keras.utils import to_categorical
labels = to_categorical(labels,num_classes = len(classes)) # Converting labels to one-hot encoding
labels

# %%
labels.shape

# %% [markdown]
# Splitting of Dataset into Training and Test set

# %%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(data,labels,test_size=0.2,random_state=42)
X_train = X_train.astype("float16")
X_test  = X_test.astype("float16")

# %% [markdown]
# Building Model

# %%
model = tf.keras.models.Sequential()

# %%
X_train[0].shape

# %%
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout

model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=X_train[0].shape))
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

# %%
model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

# %%
model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

# %%
model.add(Dropout(0.3))

# %%
model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

# %%
model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

# %%
model.add(Dropout(0.3))

# %%
model.add(Flatten())

# %%
model.add(Dense(units=1200,activation='relu'))

# %%
model.add(Dropout(0.45))

# %%
#Output layer
model.add(Dense(units=len(classes),activation='softmax'))

# %%
model.summary()

# %%
#Compile the model
from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

# %%
#Training Model
training_history = model.fit(X_train,Y_train,epochs=30,batch_size=32,validation_data=(X_test,Y_test))

# %%
model.save("Trained_Music_Genre_Classifier.h5")

# %%
training_history.history

# %%
#Recording History in json
import json
with open('training_history.json', 'w') as f:
    json.dump(training_history.history, f)

# %%
X_train

# %%
#Reloading model variable
model = tf.keras.models.load_model("Trained_Music_Genre_Classifier.h5")
model.summary()

# %%



