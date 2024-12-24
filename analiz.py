import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini oku
data = pd.read_csv('vgsales.csv')  # Dosya adını burada doğru belirt.

top_games = data.sort_values(by='Global_Sales', ascending=False).head(10)
print(top_games[['Name', 'Global_Sales']])

platform_sales = data.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)
print(platform_sales)

genre_sales = data.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
print(genre_sales)

top_publishers = data.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(10)
print(top_publishers)

# Yıllara göre toplam satışlar
yearly_sales = data.groupby('Year')['Global_Sales'].sum()

# Çizim
plt.figure(figsize=(10, 6))
plt.plot(yearly_sales.index, yearly_sales.values, marker='o', color='blue')
plt.title('Yıllara Göre Global Oyun Satışları')
plt.xlabel('Yıl')
plt.ylabel('Satış (Milyonlar)')
plt.grid()
plt.show()

# Yıllara göre satışlar
yearly_sales = data.groupby('Year')['Global_Sales'].sum().reset_index()

# Çizim
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_sales, x='Year', y='Global_Sales', marker='o', color='green')
plt.title('Yıllara Göre Global Oyun Satışları')
plt.xlabel('Yıl')
plt.ylabel('Satış (Milyonlar)')
plt.grid()
plt.show()