import pandas as pd
import numpy as np

# VERİYİ YÜKLEME
dosya_yolu = r'E:\ILKAYOKSUZ\energymarket\data\Turkish_data_2013_2016 (9).csv'
df = pd.read_csv(dosya_yolu)

df = df.fillna(method='bfill')

#  ZAMAN SERİSİ ÖZELLİKLERİ EKLE (Feature Engineering), geçmişe bakma olayı
df['Price_Lag24'] = df['Day-Ahead Price PTF (TL/MWh)'].shift(24)    # 1 gün öncesi
df['Price_Lag168'] = df['Day-Ahead Price PTF (TL/MWh)'].shift(168)  # 1 hafta öncesi

df = df.dropna()

# GİRDİ VE ÇIKTI DEĞİŞKENLERİNİ SEÇ
# Modele neleri öğreteceğiz?
ozellikler = ['Month', 'Hour1', 'Weekday', 'Holiday', 'Temperature', 
              'Forecast Demand/Supply', 'Price_Lag24', 'Price_Lag168']
hedef = 'Day-Ahead Price PTF (TL/MWh)'

# VERİYİ BÖLME (Train-Validation-Test)
train_df = df[df['Year'].isin([2013, 2014])]
val_df   = df[df['Year'] == 2015]
test_df  = df[df['Year'] == 2016]

# X: Girdiler (Özellikler), y: Tahmin edilecek değer (Fiyat)
X_train, y_train = train_df[ozellikler], train_df[hedef]
X_val,   y_val   = val_df[ozellikler],   val_df[hedef]
X_test,  y_test  = test_df[ozellikler],  test_df[hedef]

print("--- 1. ADIM TAMAMLANDI ---")
print(f"Eğitim verisi (2013-14): {len(X_train)} satır")
print(f"Doğrulama verisi (2015): {len(X_val)} satır")
print(f"Test verisi (2016): {len(X_test)} satır")




from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

print("\n--- 2. ADIM: MODELLER EĞİTİLİYOR ---")

# LASSO MODELİ (Basit/Doğrusal Model)
# alpha: Düzenlileştirme gücü (küçük değerler daha karmaşık model demektir)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_preds = lasso_model.predict(X_test)
lasso_mae = mean_absolute_error(y_test, lasso_preds)

#XGBOOST MODELİ (Gelişmiş/Doğrusal Olmayan Model)
# Burada validation (2015) verisini kullanarak aşırı öğrenmeyi engelliyoruz
xgb_model = XGBRegressor(
    n_estimators=1000, 
    learning_rate=0.05, 
    max_depth=6, 
    early_stopping_rounds=50
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_preds = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_preds)

print(f"LASSO Test MAE: {lasso_mae:.2f} TL")
print(f"XGBoost Test MAE: {xgb_mae:.2f} TL")

# Başarıyı karşılaştırma
if xgb_mae < lasso_mae:
    print("\nAnaliz: XGBoost, LASSO'dan daha iyi performans gösterdi.")




from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

print("\n--- 3. ADIM: DERİN ÖĞRENME MODELLERİ HAZIRLANIYOR ---")

# VERİYİ ÖLÇEKLENDİR (Scaling)
# Derin öğrenme için verileri 0-1 arasına çekiyoruz
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_x.fit_transform(X_train)
X_val_scaled = scaler_x.transform(X_val)
X_test_scaled = scaler_x.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))

# DNN MODELİ (Deep Neural Network)
dnn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
dnn_model.compile(optimizer='adam', loss='mae')
print("DNN eğitiliyor...")
dnn_model.fit(X_train_scaled, y_train_scaled, epochs=20, batch_size=32, verbose=0)

# LSTM MODELİ (Recurrent Neural Network)
# LSTM veriyi 3 boyutlu ister: [Örnek sayısı, Zaman adımı, Özellik sayısı]
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mae')
print("LSTM eğitiliyor...")
lstm_model.fit(X_train_lstm, y_train_scaled, epochs=20, batch_size=32, verbose=0, 
               validation_data=(X_val_lstm, y_val_scaled))

# TAHMİNLERİ GERİ DÖNÜŞTÜR (Unscaling)
dnn_preds_scaled = dnn_model.predict(X_test_scaled)
dnn_preds = scaler_y.inverse_transform(dnn_preds_scaled)

lstm_preds_scaled = lstm_model.predict(X_test_lstm)
lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled)

dnn_mae = mean_absolute_error(y_test, dnn_preds)
lstm_mae = mean_absolute_error(y_test, lstm_preds)

print(f"DNN Test MAE: {dnn_mae:.2f} TL")
print(f"LSTM Test MAE: {lstm_mae:.2f} TL")


import matplotlib.pyplot as plt

print("\n--- 4. ADIM: GÖRSELLEŞTİRME VE ANALİZ ---")

# MAE KARŞILAŞTIRMA GRAFİĞİ
modeller = ['LASSO', 'XGBoost', 'DNN', 'LSTM']
hatalar = [lasso_mae, xgb_mae, dnn_mae, lstm_mae]

plt.figure(figsize=(10, 6))
plt.bar(modeller, hatalar, color=['gray', 'blue', 'green', 'orange'])
plt.ylabel('MAE (Hata Payı - TL)')
plt.title('Modellerin Performans Karşılaştırması (Düşük olan daha iyi)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('plots/model_karsilastirma.png') # Grafiği bilgisayarına kaydeder
print("Performans grafiği 'model_karsilastirma.png' olarak kaydedildi.")

# GERÇEK VS TAHMİN GRAFİĞİ (Son 100 Saat)
plt.figure(figsize=(15, 6))
plt.plot(y_test.values[-100:], label='Gerçek Fiyat', color='black', linewidth=2)
plt.plot(dnn_preds[-100:], label='DNN Tahmini (En İyi)', color='green', linestyle='--')
plt.plot(xgb_preds[-100:], label='XGBoost Tahmini', color='blue', alpha=0.6)

plt.title('2016 Sonu: Gerçek Fiyatlar vs Model Tahminleri')
plt.xlabel('Saat')
plt.ylabel('Fiyat (TL/MWh)')
plt.legend()
plt.savefig('plots/tahmin_test.png')
print("Tahmin grafiği 'tahmin_test.png' olarak kaydedildi.")

# FİNAL SONUÇ TABLOSU
print("\n" + "="*30)
print("     FİNAL PERFORMANS ÖZETİ")
print("="*30)
for i in range(len(modeller)):
    print(f"{modeller[i]:<10}: {hatalar[i]:>6.2f} TL")
print("="*30)


# Modellerin sonuçlarını tablo gibi yazdırma
results_df = pd.DataFrame({
    'Model': ['LASSO', 'XGBoost', 'DNN', 'LSTM'],
    'MAE (TL)': [lasso_mae, xgb_mae, dnn_mae, lstm_mae]
})
print("\n--- SONUÇ TABLOSU ---")
print(results_df.to_markdown(index=False)) 