# Gerekli kütüphaneleri import et
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Veri setini yükle
data = pd.read_csv('vgsales.csv')

# NaN değerleri temizle
data = data.dropna()

# Özellik (X) ve hedef (y) değişkenlerini belirle
X = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]  # Giriş özellikleri
y = data['Global_Sales']  # Tahmin etmeye çalıştığımız değer

# Veriyi eğitim ve test setlerine ayır (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Test seti ile tahmin yap
y_pred = model.predict(X_test)

# Sonuçları değerlendir
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Gerçek ve Tahmin Değerlerini Aynı Grafikte Görselleştir
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Gerçek Değerler', color='blue', marker='o', linestyle='dashed', alpha=0.7)
plt.plot(y_pred, label='Tahmin Edilen Değerler', color='red', marker='x', linestyle='dashed', alpha=0.7)
plt.title('Gerçek vs Tahmin Edilen Değerler')
plt.xlabel('Örnek Index')
plt.ylabel('Satış Değerleri')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_test)), y_test.values, label='Gerçek Değerler', color='blue', alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, label='Tahmin Edilen Değerler', color='red', alpha=0.6)
plt.title('Gerçek vs Tahmin Edilen Değerler')
plt.xlabel('Örnek Index')
plt.ylabel('Satış Değerleri')
plt.legend()
plt.grid()
plt.show()

