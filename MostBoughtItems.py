import ReadingData
import matplotlib.pyplot as plt
#getting the top 10 most bought items
print(ReadingData.buys_df['item_id'].value_counts()[:10])

ReadingData.buys_df['item_id'].value_counts()[:10].plot(kind='barh', figsize=(7, 6), rot=0)
plt.xlabel("Frequency", labelpad=0.3)
plt.ylabel("Label Id", labelpad=0.3)
plt.title("Item that was bought the Most", y=1.02);
plt.show()