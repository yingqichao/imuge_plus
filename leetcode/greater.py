import pandas as pd
import matplotlib.pyplot as plt

file_path = "./athlete_events.csv"
athlete_events_df = pd.read_csv(file_path)

age_distribution = athlete_events_df[athlete_events_df['Age'].notna()]

plt.figure(figsize=(12, 8))
age_distribution.boxplot(column='Age', by='Sport', vert=False, patch_artist=True, showfliers=False)
plt.title('Age Distribution of Athletes by Sport')
plt.xlabel('Age')
plt.ylabel('Sport')
plt.show()