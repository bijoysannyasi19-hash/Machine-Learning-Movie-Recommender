import streamlit as st
import pandas as pd
import plotly.express as px

# Setup page
st.set_page_config(page_title="Netflix EDA Dashboard", page_icon="🍿", layout="wide")
st.title("🍿 Netflix Movie EDA Dashboard")

# Caching data loading
@st.cache_data
def load_data():
    df_raw = pd.read_csv('mymoviedb.csv', lineterminator='\n')
    df_raw['Release_Year'] = pd.to_datetime(df_raw['Release_Date']).dt.year
    
    # Explode genre for filtering
    df_raw['Genre_List'] = df_raw['Genre'].apply(lambda x: x.split(', '))
    df_exploded = df_raw.explode('Genre_List')
    
    # We also keep a movie-level dataframe for scatter plot, deduplicating
    df_movies = df_raw.groupby('Title').first().reset_index()
    return df_raw, df_exploded, df_movies

with st.spinner("Loading Data..."):
    df_raw, df_exploded, df_movies = load_data()

# Sidebar filters
st.sidebar.header("Filters")
all_genres = sorted(df_exploded['Genre_List'].unique())
selected_genres = st.sidebar.multiselect("Select Genres", all_genres, default=all_genres[:5])

min_year = int(df_raw['Release_Year'].min())
max_year = int(df_raw['Release_Year'].max())
selected_years = st.sidebar.slider("Select Release Years", min_year, max_year, (2000, max_year))

min_votes = int(df_movies['Vote_Count'].min())
max_votes = int(df_movies['Vote_Count'].max())
selected_min_votes = st.sidebar.slider("Minimum Vote Count", min_votes, 10000, 100)

# Filter logic
def filter_data(df, genres, years, min_v):
    # Filter by year and votes first
    mask = (df['Release_Year'] >= years[0]) & (df['Release_Year'] <= years[1]) & (df['Vote_Count'] >= min_v)
    filtered = df[mask]
    
    # Then filter by genre: keep movies that have AT LEAST ONE of the selected genres
    # To do this, we can check if the intersection of the movie's genres and selected genres is not empty
    if genres:
        filtered = filtered[filtered['Genre_List'].apply(lambda x: any(g in genres for g in x))]
    return filtered

df_filtered = filter_data(df_movies, selected_genres, selected_years, selected_min_votes)

# KPI Cards
st.markdown("### Key Metrics (Filtered)")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Movies", f"{len(df_filtered):,}")
with col2:
    avg_pop = df_filtered['Popularity'].mean()
    st.metric("Average Popularity", f"{avg_pop:.1f}" if not pd.isna(avg_pop) else "0.0")
with col3:
    avg_vote = df_filtered['Vote_Average'].mean()
    st.metric("Average Vote Score", f"{avg_vote:.1f}" if not pd.isna(avg_vote) else "0.0")

st.divider()

# Charts
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Genre Distribution")
    # For genre distribution, we need the exploded version of the filtered data
    if not df_filtered.empty:
        # Explode only the filtered movies to get correct genre counts
        df_filtered_exploded = df_filtered.explode('Genre_List')
        # Only count selected genres to match filter intent, or count all genres present in these movies
        genre_counts = df_filtered_exploded['Genre_List'].value_counts().reset_index()
        genre_counts.columns = ['Genre', 'Count']
        # Filter to top 10 for readability
        genre_counts = genre_counts.head(10)
        
        fig_bar = px.bar(genre_counts, x='Count', y='Genre', orientation='h', color='Count', color_continuous_scale='Viridis')
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

with col_chart2:
    st.subheader("Popularity vs Vote Average")
    if not df_filtered.empty:
        # Pick the first genre as the color for simplicity
        df_filtered['Primary_Genre'] = df_filtered['Genre_List'].apply(lambda x: x[0] if x else 'None')
        fig_scatter = px.scatter(df_filtered, x='Vote_Average', y='Popularity', color='Primary_Genre', hover_data=['Title'])
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

st.divider()

# Top 10 Table
st.subheader("Top 10 Movies by Popularity")
if not df_filtered.empty:
    top_10 = df_filtered.sort_values(by='Popularity', ascending=False).head(10)
    # Clean up table for display
    display_df = top_10[['Title', 'Release_Year', 'Genre', 'Popularity', 'Vote_Average', 'Vote_Count']].reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True)
else:
    st.info("No data available for the selected filters.")
