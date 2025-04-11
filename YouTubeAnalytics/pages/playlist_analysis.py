"""
Playlist/Series Analysis Page
Analyze performance by playlist/series to identify trends and top-performing content.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="Playlist Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #FF0000;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
    }
    .stat-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background: #1E1E1E;
        border-left: 4px solid #FF0000;
        margin-bottom: 1rem;
    }
    .card-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .card-label {
        color: #CCCCCC;
        margin-top: 0;
    }
    .highlight {
        background: #FF000022;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📊 Playlist/Series Analysis</h1>', unsafe_allow_html=True)
st.markdown("Analyze performance by playlist/series to identify trends and top-performing content")

# Check if API is initialized
if 'api' not in st.session_state:
    st.error("Please go to the home page and set up your API key and channel ID first.")
    st.stop()

# Cache functions to improve performance
@st.cache_data(ttl=3600)
def load_playlist_videos(_api, playlist_id):
    """Load all videos from a playlist and get detailed stats"""
    videos = _api.get_playlist_videos(playlist_id)
    if not videos:
        return pd.DataFrame()
        
    video_ids = [v['id'] for v in videos]
    details_df = _api.get_videos_details(video_ids)
    
    # Add playlist position to each video
    position_map = {v['id']: v['position'] for v in videos}
    details_df['position'] = details_df['id'].apply(lambda x: position_map.get(x))
    details_df['playlist_id'] = playlist_id
    
    return details_df

@st.cache_data
def analyze_titles(df, metric='viewCount', min_count=2):
    """Analyze common words in top vs bottom performing videos by a metric"""
    if len(df) < 4:  # Need at least 4 videos for meaningful analysis
        return []
        
    # Sort by the specified metric
    df_sorted = df.sort_values(by=metric, ascending=False)
    
    # Split into top performers (top 25%) and low performers (bottom 25%)
    cutoff_high = max(int(len(df_sorted) * 0.25), 1)
    cutoff_low = min(int(len(df_sorted) * 0.75), len(df_sorted) - 1)
    
    top_performers = df_sorted.iloc[:cutoff_high]
    low_performers = df_sorted.iloc[cutoff_low:]
    
    # Extract words from titles
    def extract_words(title_series):
        # Join all titles and convert to lowercase
        all_text = ' '.join(title_series).lower()
        # Extract words using regex
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        # Remove common stop words
        stop_words = {"the", "and", "you", "for", "this", "that", "with", "from", "your", 
                    "have", "are", "not", "was", "but", "they", "will", "what", "when",
                    "can", "all", "get", "just", "been", "like", "how"}
        return [w for w in words if w not in stop_words]
    
    top_words = extract_words(top_performers['title'])
    low_words = extract_words(low_performers['title'])
    
    # Count frequencies
    top_word_counts = Counter(top_words)
    low_word_counts = Counter(low_words)
    
    # Calculate impact scores
    impact_words = []
    
    for word, count in top_word_counts.items():
        if count < min_count:
            continue
            
        # Calculate frequency in top videos
        top_freq = count / len(top_performers)
        # Calculate frequency in bottom videos (avoid division by zero)
        low_freq = low_word_counts.get(word, 0) / max(len(low_performers), 1)
        
        # Calculate impact ratio (how much more common in top videos)
        if low_freq == 0:
            impact_ratio = top_freq * 10  # Arbitrary high number if word doesn't appear in bottom videos
        else:
            impact_ratio = top_freq / low_freq
            
        if impact_ratio > 1.2:  # Only include words that appear more frequently in top videos
            impact_words.append({
                'word': word,
                'count': count,
                'impact_ratio': impact_ratio,
                'top_freq': top_freq,
                'low_freq': low_freq
            })
    
    # Sort by impact ratio
    impact_words.sort(key=lambda x: x['impact_ratio'], reverse=True)
    return impact_words[:10]  # Return top 10 words by impact

@st.cache_data
def generate_title_wordcloud(titles):
    """Generate a word cloud from video titles"""
    if not titles or len(titles) == 0:
        return None
        
    # Join all titles and convert to lowercase
    all_text = ' '.join(titles).lower()
    
    # Remove common stop words
    stop_words = {"the", "and", "you", "for", "this", "that", "with", "from", "your", 
                 "have", "are", "not", "was", "but", "they", "will", "what", "when",
                 "can", "all", "get", "just", "been", "like", "how"}
                 
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='black',
        colormap='Reds',
        stopwords=stop_words,
        min_font_size=10,
        max_font_size=150,
        random_state=42
    ).generate(all_text)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    return buf

# Load playlists from session state or fetch them
if 'playlists' not in st.session_state:
    with st.spinner("Loading playlists..."):
        playlists = st.session_state.api.get_playlists(st.session_state.channel_id)
        if playlists:
            st.session_state.playlists = playlists
        else:
            st.warning("No playlists found for this channel.")
            st.stop()

# Convert playlists to DataFrame for easier handling
playlists_df = pd.DataFrame(st.session_state.playlists)

# Select a playlist to analyze
st.subheader("Select a Playlist/Series to Analyze")
selected_playlist = st.selectbox(
    "Choose a playlist",
    options=playlists_df['id'].tolist(),
    format_func=lambda x: playlists_df[playlists_df['id'] == x]['title'].iloc[0]
)

# Load videos for selected playlist
with st.spinner("Loading playlist videos..."):
    playlist_name = playlists_df[playlists_df['id'] == selected_playlist]['title'].iloc[0]
    videos_df = load_playlist_videos(st.session_state.api, selected_playlist)
    
    if videos_df.empty:
        st.warning(f"No videos found in the '{playlist_name}' playlist.")
        st.stop()

# Display basic playlist stats
st.markdown(f"<h2 class='sub-header'>Analysis: {playlist_name}</h2>", unsafe_allow_html=True)

video_count = len(videos_df)
total_views = videos_df['viewCount'].sum()
avg_views = videos_df['viewCount'].mean()
total_duration = videos_df['duration_minutes'].sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <p class="card-value">{video_count}</p>
        <p class="card-label">Videos</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <p class="card-value">{total_views:,}</p>
        <p class="card-label">Total Views</p>
    </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.markdown(f"""
    <div class="stat-card">
        <p class="card-value">{int(avg_views):,}</p>
        <p class="card-label">Avg. Views</p>
    </div>
    """, unsafe_allow_html=True)
    
with col4:
    st.markdown(f"""
    <div class="stat-card">
        <p class="card-value">{int(total_duration)} min</p>
        <p class="card-label">Total Duration</p>
    </div>
    """, unsafe_allow_html=True)

# Performance metrics over time
st.subheader("Performance Trends")

# Convert published date to datetime
videos_df['publishedAt'] = pd.to_datetime(videos_df['publishedAt'])

# Sort by published date
videos_df_sorted = videos_df.sort_values('publishedAt')

# Visualize views over time
fig = px.line(
    videos_df_sorted,
    x=np.array(videos_df_sorted['publishedAt']),
    y='viewCount',
    markers=True,
    title="Views Over Time",
    hover_data=['title'],
)
fig.update_layout(xaxis_title="Published Date", yaxis_title="Views")
st.plotly_chart(fig, use_container_width=True)

# High-performing videos
st.subheader("Top Performing Videos")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### Most Viewed Videos")
    top_views = videos_df.nlargest(5, 'viewCount')
    fig = px.bar(
        top_views,
        x='viewCount',
        y='title',
        orientation='h',
        title="Top 5 Videos by Views",
        color='engagementScore',
        color_continuous_scale=px.colors.sequential.Reds
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Views", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("##### Most Engaging Videos")
    top_engagement = videos_df.nlargest(5, 'engagementScore')
    fig = px.bar(
        top_engagement,
        x='engagementScore',
        y='title',
        orientation='h',
        title="Top 5 Videos by Engagement Score",
        color='engagementScore',
        color_continuous_scale=px.colors.sequential.Reds
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Engagement Score", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# Low-performing videos
st.subheader("Lowest Performing Videos")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### Least Viewed Videos")
    bottom_views = videos_df.nsmallest(5, 'viewCount')
    fig = px.bar(
        bottom_views,
        x='viewCount',
        y='title',
        orientation='h',
        title="Bottom 5 Videos by Views",
        color='engagementScore',
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Views", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("##### Least Engaging Videos")
    bottom_engagement = videos_df.nsmallest(5, 'engagementScore')
    fig = px.bar(
        bottom_engagement,
        x='engagementScore',
        y='title',
        orientation='h',
        title="Bottom 5 Videos by Engagement Score",
        color='engagementScore',
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Engagement Score", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# Title analysis
st.markdown('<h2 class="sub-header">Title Analysis</h2>', unsafe_allow_html=True)
st.markdown("Understanding what titles perform better in this series")

col1, col2 = st.columns(2)

with col1:
    # Generate word cloud from titles
    st.markdown("##### Title Word Cloud")
    wordcloud_buffer = generate_title_wordcloud(videos_df['title'].tolist())
    if wordcloud_buffer:
        st.image(wordcloud_buffer, use_column_width=True)
    else:
        st.info("Not enough title data to generate word cloud.")
        
with col2:
    # Title length analysis
    st.markdown("##### Title Length Analysis")
    
    # Add title length column
    videos_df['title_length'] = videos_df['title'].apply(len)
    
    # Create scatter plot of title length vs views
    fig = px.scatter(
        videos_df,
        x='title_length',
        y='viewCount',
        hover_data=['title'],
        title="Title Length vs Views",
        trendline="ols",
        labels={
            "title_length": "Title Length (characters)",
            "viewCount": "Views"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

# Analyze high-performing title words
title_impact_words = analyze_titles(videos_df, 'viewCount')

if title_impact_words:
    st.subheader("Words Associated with Higher Views")
    
    # Create DataFrame for visualization
    impact_df = pd.DataFrame(title_impact_words)
    
    # Create bar chart of impact words
    fig = px.bar(
        impact_df,
        x='impact_ratio',
        y='word',
        orientation='h',
        title="Title Words Associated with Higher View Counts",
        color='count',
        color_continuous_scale=px.colors.sequential.Reds,
        labels={
            "impact_ratio": "Impact Ratio (higher = better)",
            "word": "Word",
            "count": "Frequency"
        }
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation
    st.markdown("""
    <div class="highlight">
    <p><strong>What This Means:</strong> Words with higher impact ratios appear more frequently in your top-performing videos compared to lower-performing ones. Consider using these words in future titles for this series.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Not enough data to analyze title performance patterns.")

# Video length analysis
st.markdown('<h2 class="sub-header">Video Duration Analysis</h2>', unsafe_allow_html=True)

# Create duration categories
duration_bins = [0, 5, 10, 15, 20, 30, 60, float('inf')]
duration_labels = ['0-5 min', '5-10 min', '10-15 min', '15-20 min', '20-30 min', '30-60 min', '60+ min']
videos_df['duration_category'] = pd.cut(videos_df['duration_minutes'], bins=duration_bins, labels=duration_labels)

# Calculate average views by duration category
duration_performance = videos_df.groupby('duration_category').agg({
    'viewCount': 'mean',
    'likeCount': 'mean',
    'engagementScore': 'mean',
    'id': 'count'
}).reset_index()
duration_performance.columns = ['Duration', 'Avg Views', 'Avg Likes', 'Avg Engagement', 'Video Count']

# Only include categories with at least one video
duration_performance = duration_performance[duration_performance['Video Count'] > 0]

# Plot average views by duration
if not duration_performance.empty:
    fig = px.bar(
        duration_performance,
        x='Duration',
        y='Avg Views',
        title="Average Views by Video Duration",
        color='Video Count',
        color_continuous_scale=px.colors.sequential.Viridis,
        text='Video Count'
    )
    fig.update_traces(texttemplate='%{text} videos', textposition='outside')
    fig.update_layout(xaxis_title="Video Duration", yaxis_title="Average Views")
    st.plotly_chart(fig, use_container_width=True)
    
    # Find optimal duration
    if len(duration_performance) > 1:
        best_duration = duration_performance.loc[duration_performance['Avg Views'].idxmax()]
        st.markdown(f"""
        <div class="highlight">
        <p><strong>Optimal Duration:</strong> Videos in the <b>{best_duration['Duration']}</b> range perform best with an average of {int(best_duration['Avg Views']):,} views.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Not enough duration data to analyze performance.")

# Compare with other playlists
st.markdown('<h2 class="sub-header">Compare with Other Series</h2>', unsafe_allow_html=True)

# Select playlists to compare
playlists_to_compare = st.multiselect(
    "Select other playlists/series to compare",
    options=[p for p in playlists_df['id'].tolist() if p != selected_playlist],
    format_func=lambda x: playlists_df[playlists_df['id'] == x]['title'].iloc[0],
    max_selections=3
)

# Add current playlist to the comparison
playlists_to_compare.insert(0, selected_playlist)

if len(playlists_to_compare) > 1:
    comparison_data = []
    
    with st.spinner("Loading comparison data..."):
        for playlist_id in playlists_to_compare:
            playlist_name = playlists_df[playlists_df['id'] == playlist_id]['title'].iloc[0]
            
            # Only load if it's not the already loaded playlist
            if playlist_id == selected_playlist:
                current_videos_df = videos_df
            else:
                current_videos_df = load_playlist_videos(st.session_state.api, playlist_id)
            
            if not current_videos_df.empty:
                comparison_data.append({
                    'playlist_id': playlist_id,
                    'title': playlist_name,
                    'video_count': len(current_videos_df),
                    'total_views': current_videos_df['viewCount'].sum(),
                    'avg_views': current_videos_df['viewCount'].mean(),
                    'avg_likes': current_videos_df['likeCount'].mean(),
                    'avg_engagement': current_videos_df['engagementScore'].mean(),
                    'avg_duration': current_videos_df['duration_minutes'].mean()
                })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Compare average views
            fig = px.bar(
                comparison_df,
                x='title',
                y='avg_views',
                title="Average Views per Video by Playlist/Series",
                color='avg_views',
                color_continuous_scale=px.colors.sequential.Reds,
                text_auto='.2s'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(xaxis_title="", yaxis_title="Average Views")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Compare engagement
            fig = px.bar(
                comparison_df,
                x='title',
                y='avg_engagement',
                title="Average Engagement Score by Playlist/Series",
                color='avg_engagement',
                color_continuous_scale=px.colors.sequential.Reds,
                text_auto='.3f'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(xaxis_title="", yaxis_title="Engagement Score")
            st.plotly_chart(fig, use_container_width=True)
        
        # Display comparison table
        st.subheader("Series Comparison Table")
        
        # Format table for display
        display_df = comparison_df[['title', 'video_count', 'total_views', 'avg_views', 'avg_likes', 'avg_engagement', 'avg_duration']]
        display_df.columns = ['Series Name', 'Videos', 'Total Views', 'Avg Views', 'Avg Likes', 'Avg Engagement', 'Avg Duration (min)']
        
        # Format numeric columns
        format_dict = {
            'Total Views': '{:,.0f}',
            'Avg Views': '{:,.0f}',
            'Avg Likes': '{:,.0f}',
            'Avg Engagement': '{:.4f}',
            'Avg Duration (min)': '{:.1f}'
        }
        
        # Display styled dataframe
        st.dataframe(
            display_df.style.format(format_dict).background_gradient(
                subset=['Avg Views', 'Avg Engagement'], 
                cmap='Reds'
            ),
            use_container_width=True
        )
        
        # Identify best performing series
        best_series = comparison_df.loc[comparison_df['avg_views'].idxmax()]
        st.markdown(f"""
        <div class="highlight">
        <p><strong>Best Performing Series:</strong> "{best_series['title']}" has the highest average views ({int(best_series['avg_views']):,} per video) among the compared series.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Select at least one other playlist/series to compare.")

# Conclusions and recommendations
st.markdown('<h2 class="sub-header">Insights & Recommendations</h2>', unsafe_allow_html=True)

# Generate insights based on the data
insights = []

# Insight 1: Best performing videos
if not videos_df.empty:
    best_video = videos_df.loc[videos_df['viewCount'].idxmax()]
    insights.append(f"Your best performing video in this series is \"{best_video['title']}\" with {best_video['viewCount']:,} views.")

# Insight 2: Optimal duration
if 'best_duration' in locals():
    insights.append(f"Videos with a duration of {best_duration['Duration']} tend to perform best in this series.")

# Insight 3: Title patterns
if title_impact_words:
    top_words = ", ".join([word['word'] for word in title_impact_words[:3]])
    insights.append(f"Titles containing words like \"{top_words}\" tend to perform better in this series.")

# Display insights
for i, insight in enumerate(insights):
    st.markdown(f"**Insight {i+1}:** {insight}")

# Recommendations
st.subheader("Recommendations")

recommendations = [
    "Continue producing content for your best-performing series to maximize engagement.",
    "Consider optimizing your video titles based on the high-performing keywords identified.",
    "Aim for the optimal video duration identified for this series."
]

for rec in recommendations:
    st.markdown(f"• {rec}")

# Export options
st.markdown("---")
st.subheader("Export Data")

# Create formatted Excel
if st.button("Generate Excel Report"):
    with st.spinner("Generating report..."):
        # Create Excel buffer
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Playlist Name', 'Video Count', 'Total Views', 'Average Views', 'Average Engagement'],
                'Value': [
                    playlist_name,
                    video_count,
                    total_views,
                    avg_views,
                    videos_df['engagementScore'].mean()
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Videos sheet
            export_cols = ['title', 'publishedAt', 'viewCount', 'likeCount', 'commentCount', 
                          'duration_minutes', 'engagementScore']
                          
            # Make a copy and convert timezone-aware datetime to timezone-naive
            export_df = videos_df[export_cols].copy()
            if 'publishedAt' in export_df.columns:
                export_df['publishedAt'] = pd.DatetimeIndex(export_df['publishedAt']).tz_localize(None)
                
            export_df.to_excel(writer, sheet_name='Videos', index=False)
            
            # Top Videos sheet
            top_export_df = videos_df.nlargest(10, 'viewCount')[export_cols].copy()
            if 'publishedAt' in top_export_df.columns:
                top_export_df['publishedAt'] = pd.DatetimeIndex(top_export_df['publishedAt']).tz_localize(None)
                
            top_export_df.to_excel(writer, sheet_name='Top Videos', index=False)
            
        # Offer download
        output.seek(0)
        st.download_button(
            label="Download Excel Report",
            data=output,
            file_name=f"{playlist_name}_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
