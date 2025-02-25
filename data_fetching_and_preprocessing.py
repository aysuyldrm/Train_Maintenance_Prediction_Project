import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.api.models import Sequential, load_model
from keras.api.optimizers import Adam
from keras.api.layers import LSTM, Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import keras_tuner as kt
import numpy as np


def veritabanı_baglantısı_ve_veri_cekme(host_adi, veritabanı_adi, kullanici_adi, parola, tablo_adi, port):
    """
    MySQL veritabanına bağlanır ve belirtilen tablodan verileri çeker.

    Args:
        host_adi(str):
        veritabani_adi(str):
        kullanici_adi(str):
        parola(str):
        tablo_adi(str):
        port (int):
    """
    try:
        mydb = mysql.connector.connect(
            host = host_adi,
            user = kullanici_adi,
            port = port,
            password = parola,
            database = veritabanı_adi
        )
        if mydb.is_connected():
            cursor = mydb.cursor(dictionary=True)
            cursor.execute(f'SELECT * FROM {tablo_adi}')
            veriler = cursor.fetchall()
            df = pd.DataFrame(veriler)
        return df
    except mysql.connector.Error as e:
        print(f'Veritabanı bağlantı hatası: {e}')
        return None
    finally:
        if mydb and mydb.is_connected:
            cursor.close()
            mydb.close()

def veri_on_isleme(df, hedef_kolon = 'maintenance'):

    # 1. Handle missing values
    # Fill with mean for numerical data
    for col in df.select_dtypes(include = ['number']).columns:
        df[col].fillna(df[col].mean(), inplace = True)
    # Drop rows with missing values
    # df.dropna(inplace = True)

    # 2. Convert Timestamp to Numerical Feature (important for time series)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp_seconds'] = df['timestamp'].astype('int64')//10**9 # Seconds since epoch

    # 3. Feature Engineering (examples)

    # Create rolling average (if you have enough data)
    # df['engine_temp_rolling_mean'] = df['eng_temp'].rolling(window = 10).mean() # Example window size

    # Create time-based features (day of week, hour of day, etc.)

    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour_of_day'] = df['timestamp'].dt.hour
    df.drop('timestamp', axis = 1, inplace = True) # Drop the original timestamp column

    # Assuming your dataframe is named df
    columns_to_convert = [
        'engine_temp', 'engine_vibration', 'wheel_temp', 'wheel_wear', 
        'track_health', 'brake_pad_wear', 'brake_temp', 'brake_pressure',
        'brake_force', 'fuel_usage', 'battery_health', 'humidity', 
        'air_temp', 'latitude', 'longitude', 'speed', 'weather_temp',
        'wind_speed', 'precipitation', 'cargo_load', 'cargo_temp'
    ]

    # Convert these columns to float type
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. One-Hot Encoding

    df = pd.get_dummies(df, columns = ["door_status", "window_status", \
                                       "emergency_system", "diagnostic_data", \
                                        "error_codes"], dtype='int')
    
    X = df.drop(["train_id", "id", hedef_kolon], axis = 1)
    y = df[hedef_kolon]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    numeric_cols = X_train.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test, scaler

def mlp_model_olustur(input_shape):
    """
        Çok Katmanlı Algılayıcı (MLP) modelini oluşturur.

        Args:
            input_shape (tuple): Giriş verisinin şekli
        
        Returns:
            tensorflow.keras.models.Sequential: Derlenmiş MLP modeli
    """

    model = Sequential([
        Dense(128, activation = 'relu', input_shape=[input_shape]),
        Dropout(0.2), # Dropout katmanı
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)  # Regresyon için çıktı katmanı, aktivasyon fonksiyonu yok
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def rnn_model_olustur(input_shape):
    """
    Tekrarlayan Sinir Ağı (RNN) modeli (LSTM ile) oluşturulur.

    Args:
        input_shape (tuple): Giriş verisinin şekli (örneğin, zaman serisi için)

    Results:
        tensorflow.keras.models.Sequential: Derlenmiş RNN modeli
    """

    model = Sequential([
        LSTM(128, activation = 'relu', input_shape = input_shape),
        Dropout(0.2), # Dropout katmanı
        Dense(64, activation = 'relu'),
        Dropout(0.2), # Dropout katmanı
        Dense(1)
    ])
    model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'mse')
    return model

def cnn_model_olustur(input_shape):
    """
    1D Evrişimli Sinir Ağı (CNN) modeli oluşturur.

    Args:
        input_shape (tuple): Giriş verisinin şekli (örneğin, zaman serisi için)
    Results:
        tensorflow.keras.models.Sequential: Derlemiş CNN modeli
    """

    model = Sequential([
        Conv1D(filters=64, kernel_size = 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size = 2),
        Dropout(0.2), # Dropout katmanı
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2), # Dropout katmanı
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss= 'mse')
    return model

def model_egit_ve_degerlendir(model, model_adi, X_train, y_train, X_test, y_test, epochs = 100, batch_size = 32):
    """
    Verilen modeli egitir ve değerlendirme metriklerini hesaplar

    Args:
        model (tensorflow.keras.models.Sequential): Eğitilecek model
        model_adı (str): Modelin adı (grafik başlıkları için).
        X_train (pandas.DataFrame): Eğitim özellikleri
        y_train (pandas.DataFrame): Eğitim hedef kolonu
        X_test (pandas.DataFrame): Test özellikleri
        y_test (pandas.DataFrame) : Test hedef kolonu
        epochs (int): Eğitimin yaklaşım sayısı
        batch_size (int): Batch büyüklüğü
    
        Results:
            tuple: Modelin eğitim geçmişi, MSE, MAE, R^2 skorları
    """

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return history, mse, mae, r2

def performans_grafigi_cizdir(history, model_adi):
    """
    Modelin eğitim ve doğrulama kaybı grafiklerini çizdirir.

    Args:
        history (tensorflow.keras.callbacks.History): Modelin eğitim geçmişi.
        modelin_adi (str) : Modelin adı (grafik başlıkları için)
    """

    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title(f'{model_adi} - Eğitim ve Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    plt.grid(True)
    plt.show()

def en_iyi_modeli_keras_tuner_ile_bul(X_train, y_train, input_shape, tuner_epochs=20):
    """
    Keras Tuner kullanarak en iyi MLP modelini bulur.

    Args:
        X_train (pandas.DataFrame): Eğitim özelliklerini
        y_train (pandas.DataFrame): Eğitim hedef kolonu
        input_shape (tuple): Giriş verisinin şekli
        tuner_epochs (int): Keras Tuner deneme epoch sayısı

    Returns:
        tensorflow.keras.models.Sequentials: En iyi hiperparametrelerle eğitilmiş MLP modeli
    """

    def model_builder(hp):
        model = Sequential([
            Dense(hp.Int('units_layer1', min_value=32, max_value = 256, step=32), activation = 'relu', input_shape= [input_shape]),
            Dense(hp.Int('units_layer2', min_value=32, max_value = 128, step=32), activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                      loss='mse')
        return model

    tuner = kt.Hyperband(
        model_builder,
        objective='val_loss',
        max_epochs=10,
        factor=3,
        directory='your_directory_path', # (C:\\Users\\username\\Desktop\\yourfolder)
        project_name = 'tren_bakim_tahmini_tuner'
    )

    tuner.search(X_train, y_train, epochs = tuner_epochs, validation_split=0.20, verbose=0)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    return best_model, best_hps

def ortalama_tahmin_hesaplama(model, X_test):
    """
    Modelin test verisi üzerindeki tahminlerinin ortalamasını hesaplar.

    Args:
        model (tensorflow.keras.models.Sequential) : Kullanılacak eğitilmiş model
        X_test (pandas.DataFrame): Test özellikleri

    Results:
        float: Ortalama tahmin değeri
    """

    tahminler = model.predict(X_test).flatten()
    ortalama_tahmin = np.mean(tahminler)
    return ortalama_tahmin

def model_kaydet(model, model_kayıt_yolu):
    """
    Eğitilmiş modeli belirten yola kaydedilir.

    Args:
        model (tensorflow.keras.model.Sequential): Kaydedilecek model
        model_kayıt_yolu (str): Modelin kaydedileceği yol
    """

    model.save(model_kayıt_yolu)
    print(f'Model başarıyla {model_kayıt_yolu} yoluna kaydedildi.')


# Veritabanı bağlantı bilğileri
# MySQL Configuration

host_adi = "your_local_ip_address" # Or your MySQL server address (127.0.0.1)
kullanici_adi = "your_database_user_name" # Your MYSQL username (root)
parola = "your_database_password" # Your MYSQL password (12345)
veritabani_adi = "your_database_name" # Your database name (train_maintenance)
port = 3306 # Default MySQL port
tablo_adi = 'your_Database_table_name' # Your databse table name (sensor_table)

if __name__ == '__main__':

    #1. Veritabanından Veri Çekme
    df = veritabanı_baglantısı_ve_veri_cekme(host_adi=host_adi, \
                                            veritabanı_adi=veritabani_adi, \
                                            kullanici_adi=kullanici_adi, \
                                            parola=parola, \
                                            tablo_adi=tablo_adi, \
                                            port=port)
    if df is None:
        print('Dataframe is empty. Check the database data fecthing algorithm!')
        exit()


    #2. Veri ön işleme
    X_train, X_test, y_train, y_test, scaler = veri_on_isleme(df)
    input_shape = X_train.shape[1]

    ##3. model oluşturma
    #mlp_model = mlp_model_olustur(input_shape=input_shape)
    #rnn_model = rnn_model_olustur((X_train.shape[1], 1)) # RNN için giriş şekli (zaman adımı, özellik sayısı)
    #cnn_model = cnn_model_olustur((X_train.shape[1], 1)) # CNN için giriş şekli (zaman adımı, özellik sayısı)
#
    ## Veriyi RNN ve CNN için yeniden şekillendirme (3D tesörlere)
    #X_train_rnn = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
    #X_test_rnn = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))
    #X_train_cnn = X_train_rnn
    #X_test_cnn = X_test_rnn
#
    ##4. Model eğitme ve değerlendirme
    #modeller = {
    #    "MLP" : mlp_model,
    #    "RNN" : rnn_model,
    #    "CNN" : cnn_model
    #}
#
    #model_sonuclari = {}
#
#
    #for model_adi, model in modeller.items():
    #    if model_adi == "MLP":
    #        history, mse, mae, r2 = model_egit_ve_degerlendir(model, model_adi, X_train, y_train, X_test, y_test)
    #    elif model_adi == "RNN":
    #        history, mse, mae, r2 = model_egit_ve_degerlendir(model, model_adi, X_train_rnn, y_train, X_test_rnn, y_test)
    #    elif model_adi == "CNN":
    #        history, mse, mae, r2 = model_egit_ve_degerlendir(model, model_adi, X_train_cnn, y_train, X_test_cnn, y_test)
    #
#
    #    model_sonuclari[model_adi] = {
    #        'history' : history,
    #        'mse' : mse,
    #        'mae' : mae,
    #        'r2' : r2
    #    }
#
#
    ##5. En iyi modeli seçme
    #en_iyi_model_adi = min(model_sonuclari, key = lambda k: model_sonuclari[k]['mse'])
    #print(f'\nEn İyi Model : {en_iyi_model_adi}')
#
#
    ##6. En iyi modeli keras tuner ile iyileştirme
    #en_iyi_model_ham = modeller[en_iyi_model_adi] # Tuner'a ham mdodeli gönderiyoruz, en iyi model adını kullanarak değil
    #if en_iyi_model_adi == 'CNN': # Sadece CNN için tuner uyguluyoruz, diğer modeller için genişletilebilir.
    #    en_iyi_model_tuned, best_hps = en_iyi_modeli_keras_tuner_ile_bul(X_train, y_train, input_shape)
    #    history_tuned, mse_tuned, mae_tuned, r2_tuned = model_egit_ve_degerlendir(en_iyi_model_tuned, \
    #                                                                              f"{en_iyi_model_adi} (Tuned)", \
    #                                                                              X_train, y_train, \
    #                                                                              X_test, y_test, epochs = 200) # Tuned modeli daha çok epoch ile eğitelim.
    #    performans_grafigi_cizdir(history_tuned, f"{en_iyi_model_adi} (Tuned)")
    #    print("\nEn İyi Hiperparametreler (Keras Tuner): ")
    #    print(best_hps.values)
    #    en_iyi_model_nihai = en_iyi_model_tuned # Nihai model olarak tuned modeli belirliyoruz
    #else:
    #    en_iyi_model_nihai = en_iyi_model_ham # Tuner uygulanmayan modeller için ham modeli kullanıyoruz
    #    print("\nKeras Tuner iyileştirmesi sadece {} modeline uygulandı.".format(en_iyi_model_adi)) 
#
    ##7. Ortalama tahmin hesaplama
    #ortalama_tahmin = ortalama_tahmin_hesaplama(en_iyi_model_nihai, X_test)
    #print(f"\n{en_iyi_model_adi} Modeli Ortalama Tahmin Bakım Gün Değeri (Test Seti) : {ortalama_tahmin:.2f} gün")
#
    ##8. En iyi modeli kaydetme
    #model_kaydet(en_iyi_model_nihai, model_kayıt_yolu="your_saving_path")
    #print(f"\n{en_iyi_model_adi} 'your_saving_path' adresine kaydedildi.")


    #9. Model yükleme ve tahminleme
    best_model = load_model("your_Saving_path", custom_objects={'mse': mean_squared_error})
    ortalama_tahmin = ortalama_tahmin_hesaplama(best_model, X_test)
    print(f"\nCNN Modeli Ortalama Tahmin Bakım Gün Değeri (Test Seti) : {ortalama_tahmin:.2f} gün")
    print("\nProje tamamlandı.")
    