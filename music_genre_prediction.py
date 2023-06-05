import numpy as np
import pandas as pd
import os
import scipy.io.wavfile as wavfile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    odd = fft(x[0::2])
    even = fft(x[1::2])
    factor = np.exp(-1j *2* np.pi * np.arange(N//2) / N)
            #Xt + W*Xc  <->  Xt - W*Xc
    return np.concatenate([odd + factor[:N // 2] * even, odd - factor[:N // 2] * even])

def stft(signal, window_size, hop_size, windowing_function):
    num_frames = (len(signal) - window_size) // hop_size + 1
    stft_matrix = np.zeros((window_size, num_frames+1), dtype=np.complex128)
    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size
        frame = signal[start:end]
        windowed_frame = frame * windowing_function
        stft_matrix[:, i] = fft(windowed_frame)
    
    #son windowa geldiğinde bu kalan kısım windowsi<e'dan küçükse 0 ile doldurulur
    if start<len(signal):
        i+=1
        start = i * hop_size
        end = len(signal)
        frame = signal[start:end]
        pad_length = window_size - len(frame)
        frame = np.pad(frame, (0, pad_length), 'constant')
        windowed_frame = frame * windowing_function
        stft_matrix[:, i] = fft(windowed_frame)
    return stft_matrix


# Özellik vektörünü hesapla
def calculate_features(stft_matrix, sample_rate):
    num_frames = stft_matrix.shape[1]
    #sample_rate pencere bsayısına bölerek frekans düzlemini çıkarır
    frequency_range = np.linspace(0, sample_rate, num_frames)
    power_mean = []
    amplitude_mean = []
    weighted_frequency_mean = []

    for i in range(num_frames):
        #sütunu al
        frame = stft_matrix[:, i]
        #1.	Frekans Gücü (Power)
        power = np.abs(frame) ** 2
        power_mean.append(np.mean(power))
        #2.	Frekans Düzleminde Genlik Ortalaması
        amplitude = np.mean(np.abs(frame))
        amplitude_mean.append(np.mean(amplitude))
        #3.	Genliğe göre ağırlıklı frekans ortalaması
        weighted_frequency = np.sum(frequency_range * np.abs(stft_matrix), axis=1) / np.sum(np.abs(stft_matrix), axis=1)
        weighted_frequency_mean.append(weighted_frequency)

    features = [ #hepsinin ortalama, standart sapma ve medyanını bul
        np.mean(power_mean), np.std(power_mean), np.median(power_mean),
        np.mean(amplitude_mean), np.std(amplitude_mean), np.median(amplitude_mean),
        np.mean(weighted_frequency_mean), np.std(weighted_frequency_mean), np.median(weighted_frequency_mean)
    ]
    return features

def get_features():
    window_size = 4096
    hanning_window = np.hanning(window_size)
    hamming_window = np.hamming(window_size)
    blackman_window = np.blackman(window_size)

    save_features_to_csv(hanning_window, "Hanning","train")
    """save_features_to_csv(hamming_window, "Hamming","train")
    save_features_to_csv(blackman_window, "Blackman","train")

    save_features_to_csv(hanning_window, "Hanning","test")
    save_features_to_csv(hamming_window, "Hamming","test")
    save_features_to_csv(blackman_window, "Blackman","test")"""

    return

def save_features_to_csv(windowing_function, function_name,type):
    #dosyanın bulundugu konum
    base_dir = os.path.dirname(os.path.abspath(__file__))

    columns = ["Power_Mean", "Power_Std", "Power_Median", "Amplitude_Mean_Mean", "Amplitude_Mean_Std", "Amplitude_Mean_Median", "Weighted_Frequency_Mean_Mean", "Weighted_Frequency_Mean_Std", "Weighted_Frequency_Mean_Median","Genre"]
    #csv'ye dönüştürülecek fature dataframe'ini aç
    data = pd.DataFrame(columns=columns)

    #gerekli bazı nitelikler
    window_size = 4096
    overlap_ratio = 0.2
    hop_size = int(window_size * (1 - overlap_ratio))
    
    # Sınıfların adlarını tanımla
    genres = ["classical", "disco", "hiphop", "jazz", "metal"]
    factor = 20
    if type=="test":
        factor =10
    

    for i,genre_name in enumerate(genres):
        #müzik türü dosyasına gir
        genre_folder_path = os.path.join(base_dir, genre_name)
        print(genre_folder_path)
        
        # Her bir türün altındaki train/test klasörünü dolaş
        folder_path = os.path.join(genre_folder_path, type)
        files = os.listdir(folder_path)

        for j,file_name in enumerate(files):
            #her wav dosyasını dolaş
            file_path = os.path.join(folder_path, file_name)

            sample_rate, signal = wavfile.read(file_path)

            #stft sonucunu kaydet
            stft_result = stft(signal, window_size, hop_size, windowing_function)
            #stft sonucundan çıkarılan featureları dataframe'e kaydet
            data.loc[i*factor+j] = calculate_features(stft_result,sample_rate) + [i]
        
    print(data)
    #tüm featureları csv'ye kaydet
    data.to_csv(function_name+"_"+type+".csv")
    return

#aşağıdaki satır get_features() methodu  yeni feature çıkarılıp csvye kaydetmek istenildiği zaman yorumdan çıkarılmalıdır.
#get_features()

# Train verilerini okur
train_hamming_df = pd.read_csv('Hamming_train.csv')
train_hanning_df = pd.read_csv('Hanning_train.csv')
train_blackman_df = pd.read_csv('Blackman_train.csv')

# Test verilerini okur
test_hamming_df = pd.read_csv('Hamming_test.csv')
test_hanning_df = pd.read_csv('Hanning_test.csv')
test_blackman_df = pd.read_csv('Blackman_test.csv')


# Train verilerini özellik (feature) ve hedef (target) olarak ayırır
train_hamming_features = train_hamming_df.drop('Genre', axis=1)
train_hamming_target = train_hamming_df['Genre']

train_hanning_features = train_hanning_df.drop('Genre', axis=1)
train_hanning_target = train_hanning_df['Genre']

train_blackman_features = train_blackman_df.drop('Genre', axis=1)
train_blackman_target = train_blackman_df['Genre']

# Test verilerini özellik (feature) ve hedef (target) olarak ayırır
test_hamming_features = test_hamming_df.drop('Genre', axis=1)
test_hamming_target = test_hamming_df['Genre']

test_hanning_features = test_hanning_df.drop('Genre', axis=1)
test_hanning_target = test_hanning_df['Genre']

test_blackman_features = test_blackman_df.drop('Genre', axis=1)
test_blackman_target = test_blackman_df['Genre']


# Hamming için knn algoritmasını çalışıtırp sonuçları kaydet
knn = KNeighborsClassifier(n_neighbors=1)  # K-NN sınıflandırıcı n_neighbors=1 olarak tanımlanmış
knn.fit(train_hamming_features, train_hamming_target)
hamming_predictions1 = pd.DataFrame(knn.predict(test_hamming_features),columns=["Prediction"])

knn = KNeighborsClassifier(n_neighbors=3)  # K-NN sınıflandırıcı n_neighbors=3 olarak tanımlanmış
knn.fit(train_hamming_features, train_hamming_target)
hamming_predictions3 = pd.DataFrame(knn.predict(test_hamming_features),columns=["Prediction"])

knn = KNeighborsClassifier(n_neighbors=5)  # K-NN sınıflandırıcı n_neighbors=5 olarak tanımlanmış
knn.fit(train_hamming_features, train_hamming_target)
hamming_predictions5 = pd.DataFrame(knn.predict(test_hamming_features),columns=["Prediction"])

# Hanning için knn algoritmasını çalışıtırp sonuçları kaydet
knn = KNeighborsClassifier(n_neighbors=1)  # K-NN sınıflandırıcı n_neighbors=1 olarak tanımlanmış
knn.fit(train_hanning_features, train_hanning_target)
hanning_predictions1 = pd.DataFrame(knn.predict(test_hanning_features),columns=["Prediction"])

knn = KNeighborsClassifier(n_neighbors=3)  # K-NN sınıflandırıcı n_neighbors=3 olarak tanımlanmış
knn.fit(train_hanning_features, train_hanning_target)
hanning_predictions3 = pd.DataFrame(knn.predict(test_hanning_features),columns=["Prediction"])

knn = KNeighborsClassifier(n_neighbors=5)  # K-NN sınıflandırıcı n_neighbors=5 olarak tanımlanmış
knn.fit(train_hanning_features, train_hanning_target)
hanning_predictions5 = pd.DataFrame(knn.predict(test_hanning_features),columns=["Prediction"])

# Blackman için knn algoritmasını çalışıtırp sonuçları kaydet
knn = KNeighborsClassifier(n_neighbors=1)  # K-NN sınıflandırıcı n_neighbors=1 olarak tanımlanmış
knn.fit(train_blackman_features, train_blackman_target)
blackman_predictions1 = pd.DataFrame(knn.predict(test_blackman_features),columns=["Prediction"])

knn = KNeighborsClassifier(n_neighbors=3)  # K-NN sınıflandırıcı n_neighbors=3 olarak tanımlanmış
knn.fit(train_blackman_features, train_blackman_target)
blackman_predictions3 = pd.DataFrame(knn.predict(test_blackman_features),columns=["Prediction"])

knn = KNeighborsClassifier(n_neighbors=5)  # K-NN sınıflandırıcı n_neighbors=5 olarak tanımlanmış
knn.fit(train_blackman_features, train_blackman_target)
blackman_predictions5 = pd.DataFrame(knn.predict(test_blackman_features),columns=["Prediction"])


        #Sonuçların birlikte ekrana yazdırılması ve accuracy hesaplama
#Hamming
frames = [test_hamming_target, hamming_predictions1]
hamming_result1 = pd.concat(frames, axis=1)
print(hamming_result1)
hamming_accuracy1 = accuracy_score(test_hamming_target, hamming_predictions1)
print(f"Hamming 1 accuracy: {hamming_accuracy1}")

frames = [test_hamming_target, hamming_predictions3]
hamming_result3 = pd.concat(frames, axis=1)
print(hamming_result3)
hamming_accuracy3 = accuracy_score(test_hamming_target, hamming_predictions3)
print(f"Hamming 3 accuracy: {hamming_accuracy3}")

frames = [test_hamming_target, hamming_predictions5]
hamming_result5 = pd.concat(frames, axis=1)
print(hamming_result5)
hamming_accuracy5 = accuracy_score(test_hamming_target, hamming_predictions5)
print(f"Hamming 5 accuracy: {hamming_accuracy5}")

#Hanning
frames = [test_hanning_target, hanning_predictions1]
hanning_result1 = pd.concat(frames, axis=1)
print(hanning_result1)
hanning_accuracy1 = accuracy_score(test_hanning_target, hanning_predictions1)
print(f"Hanning 1 accuracy: {hanning_accuracy1}")

frames = [test_hanning_target, hanning_predictions3]
hanning_result3 = pd.concat(frames, axis=1)
print(hanning_result3)
hanning_accuracy3 = accuracy_score(test_hanning_target, hanning_predictions3)
print(f"Hanning 3 accuracy: {hanning_accuracy3}")

frames = [test_hanning_target, hanning_predictions5]
hanning_result5 = pd.concat(frames, axis=1)
print(hanning_result5)
hanning_accuracy5 = accuracy_score(test_hanning_target, hanning_predictions5)
print(f"Hanning 5 accuracy: {hanning_accuracy5}")

#Blackman
frames = [test_blackman_target, blackman_predictions1]
blackman_result1 = pd.concat(frames, axis=1)
print(blackman_result1)
blackman_accuracy1 = accuracy_score(test_blackman_target, blackman_predictions1)
print(f"Blackman 1 accuracy: {blackman_accuracy1}")

frames = [test_blackman_target, blackman_predictions3]
blackman_result3 = pd.concat(frames, axis=1)
print(blackman_result3)
blackman_accuracy3 = accuracy_score(test_blackman_target, blackman_predictions3)
print(f"Blackman 3 accuracy: {blackman_accuracy3}")

frames = [test_blackman_target, blackman_predictions5]
blackman_result5 = pd.concat(frames, axis=1)
print(blackman_result5)
blackman_accuracy5 = accuracy_score(test_blackman_target, blackman_predictions5)
print(f"Blackman 5 accuracy: {blackman_accuracy5}")

# özet olarak tüm accuracy sonuçlarını bastır
print("--------------------------")
print(f"Hamming 1 accuracy: {hamming_accuracy1}")
print(f"Hamming 3 accuracy: {hamming_accuracy3}")
print(f"Hamming 5 accuracy: {hamming_accuracy5}")
print(f"Hanning 1 accuracy: {hanning_accuracy1}")
print(f"Hanning 3 accuracy: {hanning_accuracy3}")
print(f"Hanning 5 accuracy: {hanning_accuracy5}")
print(f"Blackman 1 accuracy: {blackman_accuracy1}")
print(f"Blackman 3 accuracy: {blackman_accuracy3}")
print(f"Blackman 5 accuracy: {blackman_accuracy5}")
