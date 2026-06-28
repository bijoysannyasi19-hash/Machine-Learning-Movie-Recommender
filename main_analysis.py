import os
import matplotlib.pyplot as plt
os.makedirs('visualizations', exist_ok=True)
plot_counter = 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')




df = pd.read_csv('mymoviedb.csv', lineterminator='\n')
df.head()



df.info()



# Missing values & duplicates
print("Nulls per column:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())



df.describe()



# Extract release year
df['Release_Date'] = pd.to_datetime(df['Release_Date']).dt.year
df.rename(columns={'Release_Date': 'Release_Year'}, inplace=True)
df['Release_Year'].head()



# Drop columns not needed for this analysis
df.drop(columns=['Overview', 'Original_Language', 'Poster_Url'], inplace=True)
df.head()



def categorize_column(df, col, labels):
    """Bucket a numeric column into quartile-based categories."""
    edges = [df[col].min(), df[col].quantile(.25), df[col].quantile(.5),
             df[col].quantile(.75), df[col].max()]
    return pd.cut(df[col], edges, labels=labels, duplicates='drop')

df['Vote_Average_Category'] = categorize_column(
    df, 'Vote_Average', ['not_popular', 'below_avg', 'average', 'popular']
)
df['Vote_Average_Category'].value_counts()



df.dropna(inplace=True)

# Split comma-separated genres into one row per genre
df['Genre'] = df['Genre'].str.split(', ')
df = df.explode('Genre').reset_index(drop=True)
df['Genre'] = df['Genre'].astype('category')
df.head()



print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
df.nunique()



plt.figure(figsize=(8, 6))
sns.countplot(y='Genre', data=df, order=df['Genre'].value_counts().index, color='#4287f5')
plt.title('Genre distribution')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig(f'visualizations/plot_{plot_counter}.png')
print(f'Saved plot_{plot_counter}.png')
plot_counter += 1
plt.close()

top_genre = df['Genre'].value_counts().idxmax()
pct = df['Genre'].value_counts(normalize=True).max() * 100
print(f"Most frequent genre: {top_genre} ({pct:.1f}% of all genre tags)")



plt.figure(figsize=(7, 5))
sns.countplot(y='Vote_Average_Category', data=df,
              order=df['Vote_Average_Category'].value_counts().index, color='#4287f5')
plt.title('Vote average category distribution')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig(f'visualizations/plot_{plot_counter}.png')
print(f'Saved plot_{plot_counter}.png')
plot_counter += 1
plt.close()



top = df[df['Popularity'] == df['Popularity'].max()]
print(top[['Title', 'Popularity', 'Vote_Count', 'Genre']].to_string(index=False))



bottom = df[df['Popularity'] == df['Popularity'].min()]
print(bottom[['Title', 'Popularity', 'Vote_Count', 'Genre']].to_string(index=False))



plt.figure(figsize=(9, 5))
df['Release_Year'].plot(kind='hist', bins=30, color='#4287f5')
plt.title('Films by release year')
plt.xlabel('Year')
plt.tight_layout()
plt.savefig(f'visualizations/plot_{plot_counter}.png')
print(f'Saved plot_{plot_counter}.png')
plot_counter += 1
plt.close()

peak_year = df.drop_duplicates('Title')['Release_Year'].value_counts().idxmax()
print(f"Year with most films: {peak_year}")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


# Load data
df_raw = pd.read_csv('mymoviedb.csv', lineterminator='\n')
df_raw['Release_Year'] = pd.to_datetime(df_raw['Release_Date']).dt.year

# Create movie-level dataframe
# Some movies might have identical titles in different years, so we group by both.
df_movies = df_raw.groupby(['Title', 'Release_Year']).agg({
    'Popularity': 'first',
    'Vote_Count': 'first',
    'Vote_Average': 'first',
    'Genre': lambda x: list(set(x.iloc[0].split(', '))) # keep as list
}).reset_index()

print(f"Total unique movies: {len(df_movies)}")
df_movies.head()


corr_cols = ['Popularity', 'Vote_Count', 'Vote_Average']
corr_matrix = df_movies[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap: Popularity vs Votes')
plt.savefig(f'visualizations/plot_{plot_counter}.png')
print(f'Saved plot_{plot_counter}.png')
plot_counter += 1
plt.close()


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw Popularity
sns.histplot(df_movies['Popularity'], bins=50, ax=axes[0], color='skyblue')
axes[0].set_title('Raw Popularity Distribution')
axes[0].set_xlabel('Popularity')

# Log1p Popularity
sns.histplot(np.log1p(df_movies['Popularity']), bins=50, ax=axes[1], color='salmon')
axes[1].set_title('Log1p(Popularity) Distribution')
axes[1].set_xlabel('Log1p(Popularity)')

plt.tight_layout()
plt.savefig(f'visualizations/plot_{plot_counter}.png')
print(f'Saved plot_{plot_counter}.png')
plot_counter += 1
plt.close()


# Explode the genres so each row is one movie-genre pair
df_genre = df_movies.explode('Genre')

# Aggregate
genre_agg = df_genre.groupby('Genre').agg(
    Mean_Popularity=('Popularity', 'mean'),
    Mean_Vote_Average=('Vote_Average', 'mean'),
    Count=('Title', 'count')
).sort_values(by='Mean_Popularity', ascending=False)

genre_agg


# Visualizing mean popularity per genre
plt.figure(figsize=(10, 8))
sns.barplot(x='Mean_Popularity', y=genre_agg.index, data=genre_agg, palette='viridis')
plt.title('Mean Popularity per Genre')
plt.xlabel('Mean Popularity')
plt.ylabel('Genre')
plt.savefig(f'visualizations/plot_{plot_counter}.png')
print(f'Saved plot_{plot_counter}.png')
plot_counter += 1
plt.close()


# Get top 5 most common genres
top_5_genres = genre_agg.nlargest(5, 'Count').index.tolist()
print("Top 5 genres by count:", top_5_genres)

# Filter for these genres
df_top5 = df_genre[df_genre['Genre'].isin(top_5_genres)]

# Group by year and genre to get counts
yearly_genre_counts = df_top5.groupby(['Release_Year', 'Genre']).size().unstack(fill_value=0)

# Calculate proportions (share of the top 5 total for that year)
yearly_genre_share = yearly_genre_counts.div(yearly_genre_counts.sum(axis=1), axis=0)

# Filter years to remove noisy early years with very few movies (e.g., pre-1960 if necessary)
# Let's keep from 1980 onwards for a cleaner chart where most data lies
yearly_genre_share = yearly_genre_share[yearly_genre_share.index >= 1980]

# Plot stacked area chart
yearly_genre_share.plot.area(figsize=(12, 6), colormap='Set2', alpha=0.8)
plt.title('Share of Top 5 Genres Over Time (1980 - Present)')
plt.xlabel('Release Year')
plt.ylabel('Proportion (within Top 5)')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'visualizations/plot_{plot_counter}.png')
print(f'Saved plot_{plot_counter}.png')
plot_counter += 1
plt.close()


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def categorize_column(df, col, labels):
    """Bucket a numeric column into quartile-based categories."""
    edges = [df[col].min() - 0.001, df[col].quantile(.25), df[col].quantile(.5),
             df[col].quantile(.75), df[col].max() + 0.001]
    return pd.cut(df[col], bins=edges, labels=labels, duplicates='drop')

print("Loading data...")
df_raw = pd.read_csv('mymoviedb.csv', lineterminator='\n')

df_raw['Release_Year'] = pd.to_datetime(df_raw['Release_Date']).dt.year

# Deduplicate to movie-level
df_movies = df_raw.groupby(['Title', 'Release_Year']).agg({
    'Vote_Count': 'first',
    'Vote_Average': 'first',
    'Genre': lambda x: list(set(x.iloc[0].split(', ')))
}).reset_index()

labels = ['not_popular', 'below_avg', 'average', 'popular']
df_movies['Vote_Average_Category'] = categorize_column(
    df_movies, 'Vote_Average', labels
)

df_movies.dropna(subset=['Vote_Average_Category'], inplace=True)
df_movies.head()


mlb = MultiLabelBinarizer()
genre_features = pd.DataFrame(mlb.fit_transform(df_movies['Genre']), columns=mlb.classes_, index=df_movies.index)

X = pd.concat([df_movies[['Vote_Count', 'Release_Year']], genre_features], axis=1)
y = df_movies['Vote_Average_Category']

# 80/20 train/test split with fixed random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


# Train Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Train Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Evaluation Function
def evaluate(model_name, y_true, y_pred):
    print(f"--- {model_name} ---")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))
    return report['macro avg']['f1-score']

lr_f1 = evaluate("Logistic Regression", y_test, lr_preds)
rf_f1 = evaluate("Random Forest Classifier", y_test, rf_preds)


if rf_f1 > lr_f1:
    winner_name = "Random Forest"
    winner_model = rf_model
    winner_preds = rf_preds
else:
    winner_name = "Logistic Regression"
    winner_model = lr_model
    winner_preds = lr_preds

print(f"Winner: {winner_name}")

cm = confusion_matrix(y_test, winner_preds, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f'Confusion Matrix: {winner_name}')
plt.ylabel('True Category')
plt.xlabel('Predicted Category')
plt.savefig(f'visualizations/plot_{plot_counter}.png')
print(f'Saved plot_{plot_counter}.png')
plot_counter += 1
plt.close()


os.makedirs('models', exist_ok=True)
joblib.dump(winner_model, 'models/vote_category_classifier.pkl')
print("Model saved to models/vote_category_classifier.pkl")


from recommender import MovieRecommender

# Initialize the recommender (this loads data and computes the similarity matrix)
recommender = MovieRecommender(data_path='mymoviedb.csv')
print("Recommender initialized.")


title1 = "Spider-Man: No Way Home"
print(f"Recommendations for '{title1}':")
print(recommender.recommend(title1, n=5))


title2 = "Encanto"
print(f"Recommendations for '{title2}':")
print(recommender.recommend(title2, n=5))


title3 = "The Batman"
print(f"Recommendations for '{title3}':")
print(recommender.recommend(title3, n=5))


typo_title = "The Batmen" # intentional typo
print(f"Trying to find recommendations for '{typo_title}'...")
print(recommender.recommend(typo_title, n=5))


typo_title_2 = "Spder-Man No Way Home"
print(f"Trying to find recommendations for '{typo_title_2}'...")
print(recommender.recommend(typo_title_2, n=5))


import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sns.set_style('whitegrid')


print("Loading data...")
df_raw = pd.read_csv('mymoviedb.csv', lineterminator='\n')

# Explode genres
df_raw['Genre'] = df_raw['Genre'].str.split(', ')
df_exploded = df_raw.explode('Genre')

# Select top 5 genres to compare
top_genres = df_exploded['Genre'].value_counts().head(5).index.tolist()
print(f"Analyzing posters for genres: {top_genres}")

# Sample 30 movies per top genre
samples = []
for genre in top_genres:
    genre_movies = df_exploded[df_exploded['Genre'] == genre].drop_duplicates('Title')
    # Use random state for reproducibility
    sampled = genre_movies.sample(n=min(30, len(genre_movies)), random_state=42)
    samples.append(sampled)

df_sample = pd.concat(samples).drop_duplicates('Title').reset_index(drop=True)
print(f"Total posters to download: {len(df_sample)}")


def get_dominant_color(url, timeout=5):
    try:
        response = requests.get(url, timeout=timeout, verify=False)
        response.raise_for_status()
        # Open image and resize to speed up KMeans
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((50, 75))
        
        # Reshape data to (pixels, 3)
        pixels = np.array(img).reshape(-1, 3)
        
        # KMeans to find dominant color
        kmeans = KMeans(n_clusters=1, random_state=42, n_init=1)
        kmeans.fit(pixels)
        
        # Get dominant color as RGB tuple
        dominant_color = kmeans.cluster_centers_[0]
        # Convert to integer
        return tuple(map(int, dominant_color))
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

# Process posters
print("Downloading and processing posters. This may take a minute or two...")
start_time = time.time()

dominant_colors = []
brightness_vals = []

for idx, row in df_sample.iterrows():
    color = get_dominant_color(row['Poster_Url'])
    if color:
        dominant_colors.append(color)
        # Calculate perceived brightness (standard formula)
        # Brightness = sqrt(0.299*R^2 + 0.587*G^2 + 0.114*B^2)
        r, g, b = color
        brightness = np.sqrt(0.299*(r**2) + 0.587*(g**2) + 0.114*(b**2))
        brightness_vals.append(brightness)
    else:
        dominant_colors.append(None)
        brightness_vals.append(None)
        
    # Rate limit print
    if (idx + 1) % 25 == 0:
        print(f"Processed {idx + 1}/{len(df_sample)} posters...")

df_sample['Dominant_Color'] = dominant_colors
df_sample['Brightness'] = brightness_vals

# Drop failures
df_sample.dropna(subset=['Brightness'], inplace=True)
print(f"Successfully processed {len(df_sample)} posters in {time.time() - start_time:.1f}s.")


# Explode the successfully processed sample
df_sample_exploded = df_sample.explode('Genre')

# Filter to only the top genres we care about
df_analysis = df_sample_exploded[df_sample_exploded['Genre'].isin(top_genres)]

genre_brightness = df_analysis.groupby('Genre')['Brightness'].mean().sort_values(ascending=False).reset_index()

genre_brightness


# Visualize average brightness per genre
plt.figure(figsize=(10, 6))
sns.barplot(x='Brightness', y='Genre', data=genre_brightness, palette='magma')
plt.title('Average Poster Brightness by Genre')
plt.xlabel('Perceived Brightness (0-255)')
plt.ylabel('Genre')
plt.savefig(f'visualizations/plot_{plot_counter}.png')
print(f'Saved plot_{plot_counter}.png')
plot_counter += 1
plt.close()

