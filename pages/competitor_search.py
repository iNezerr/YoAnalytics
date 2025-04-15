"""
Competitor Analysis Page
Search for and analyze similar content across YouTube to identify opportunities.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="Competitor Analysis",
    page_icon="🔍",
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
        margin-bottom: 1rem;
    }
    .opportunity {
        background: #00FF0022;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .recommendation-section {
        background: #0066CC22;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .recommendation-title {
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .recommendation-point {
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🔍 Competitor Analysis</h1>', unsafe_allow_html=True)
st.markdown("Search for similar content across YouTube to identify opportunities and trends")

# Check if API is initialized
if 'api' not in st.session_state:
    st.error("Please go to the home page and set up your API key and channel ID first.")
    st.stop()

# Cache functions to improve performance
@st.cache_data(ttl=3600)
def search_youtube(_api, query, max_results=50, published_after=None, published_before=None):
    """Search YouTube and get video details with date filtering"""
    search_results = _api.search_videos(query, max_results)
    
    if not search_results:
        return pd.DataFrame()
    
    # Extract video IDs
    video_ids = [video['id'] for video in search_results]
    
    # Get detailed stats
    details_df = _api.get_videos_details(video_ids)
    
    # Add channel info from search results
    channel_info = {video['id']: {'channelTitle': video['channelTitle'], 
                               'channelId': video['channelId']} 
                 for video in search_results}
    
    # Map channel info to details DataFrame
    if not details_df.empty:
        details_df['channelTitle'] = details_df['id'].map(lambda x: channel_info.get(x, {}).get('channelTitle', ''))
        details_df['channelId'] = details_df['id'].map(lambda x: channel_info.get(x, {}).get('channelId', ''))
        
        # Apply date filters if provided
        if published_after or published_before:
            details_df['publishedAt'] = pd.to_datetime(details_df['publishedAt'])
            
            if published_after:
                details_df = details_df[details_df['publishedAt'] >= published_after]
            if published_before:
                details_df = details_df[details_df['publishedAt'] <= published_before]
    
    return details_df

@st.cache_data(ttl=3600)
def search_my_channel(_api, channel_id, query, max_results=10, published_after=None, published_before=None):
    """Search for videos on my channel matching the query with date filtering"""
    # Add channel filter to the query
    channel_query = f"{query} channelId:{channel_id}"
    
    # Search YouTube
    search_results = _api.search_videos(channel_query, max_results)
    
    if not search_results:
        return pd.DataFrame()
    
    # Extract video IDs
    video_ids = [video['id'] for video in search_results]
    
    # Get detailed stats
    details_df = _api.get_videos_details(video_ids)
    
    # Apply date filters if provided
    if not details_df.empty and (published_after or published_before):
        details_df['publishedAt'] = pd.to_datetime(details_df['publishedAt'])
        
        if published_after:
            details_df = details_df[details_df['publishedAt'] >= published_after]
        if published_before:
            details_df = details_df[details_df['publishedAt'] <= published_before]
    
    return details_df

@st.cache_data
def analyze_title_keywords(df, min_count=2):
    """Extract and analyze keywords from video titles"""
    if df.empty or len(df) < 2:
        return []
    
    # Combine all titles
    all_titles = ' '.join(df['title'].str.lower())
    
    # Extract words using regex
    words = re.findall(r'\b[a-z]{3,}\b', all_titles)
    
    # Remove common stop words
    stop_words = {"the", "and", "you", "for", "this", "that", "with", "from", "your", 
                "have", "are", "not", "was", "but", "they", "will", "what", "when",
                "can", "all", "get", "just", "been", "like", "how", "video", "watch"}
    
    filtered_words = [w for w in words if w not in stop_words]
    
    # Count frequencies
    word_counts = Counter(filtered_words)
    
    # Sort by frequency
    popular_keywords = [{'keyword': word, 'count': count} 
                      for word, count in word_counts.most_common(20)
                      if count >= min_count]
    
    return popular_keywords

@st.cache_data
def analyze_descriptions(df, min_count=2):
    """Analyze common phrases in descriptions of top performing videos"""
    if df.empty or len(df) < 4:  # Need at least 4 videos for meaningful analysis
        return []
        
    # Get top performers (top 25%)
    df_sorted = df.sort_values(by='viewCount', ascending=False)
    cutoff = max(int(len(df_sorted) * 0.25), 1)
    top_performers = df_sorted.iloc[:cutoff]
    
    # Extract common phrases from descriptions
    all_descriptions = ' '.join(top_performers['description'].fillna('').str.lower())
    
    # Look for common patterns in descriptions
    url_pattern = re.compile(r'https?://\S+')
    urls = url_pattern.findall(all_descriptions)
    url_counter = Counter(urls)
    
    # Look for sentence patterns (sentences that appear in multiple descriptions)
    sentences = re.split(r'[.!?]', all_descriptions)
    # Filter out very short sentences and trim whitespace
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    sentence_counter = Counter(sentences)
    
    # Find common phrases (3+ words)
    phrases = []
    words = all_descriptions.split()
    for i in range(len(words)-2):
        phrase = ' '.join(words[i:i+3])
        if len(phrase) > 10:  # Minimum length to avoid common words
            phrases.append(phrase)
    phrase_counter = Counter(phrases)
    
    # Combine insights
    description_patterns = []
    
    # Add common URLs (might be channel links, social media, etc.)
    for url, count in url_counter.most_common(5):
        if count >= min_count:
            description_patterns.append({
                'type': 'url',
                'content': url,
                'count': count
            })
    
    # Add common sentences
    for sentence, count in sentence_counter.most_common(3):
        if count >= min_count:
            description_patterns.append({
                'type': 'sentence',
                'content': sentence,
                'count': count
            })
    
    # Add common phrases
    for phrase, count in phrase_counter.most_common(5):
        if count >= min_count:
            description_patterns.append({
                'type': 'phrase',
                'content': phrase,
                'count': count
            })
    
    return description_patterns

@st.cache_data
def extract_hashtags(df, min_count=2):
    """Extract hashtags from titles and descriptions of top videos"""
    if df.empty or len(df) < 4:
        return []
    
    # Sort by performance
    df_sorted = df.sort_values(by='viewCount', ascending=False)
    
    # Get top performers (top 33%)
    cutoff = max(int(len(df_sorted) * 0.33), 1)
    top_performers = df_sorted.iloc[:cutoff]
    
    # Combine titles and descriptions
    all_text = ' '.join(top_performers['title'] + ' ' + top_performers['description'].fillna(''))
    
    # Extract hashtags
    hashtag_pattern = re.compile(r'#\w+')
    hashtags = hashtag_pattern.findall(all_text.lower())
    
    # Count frequencies
    hashtag_counter = Counter(hashtags)
    
    # Return most common hashtags
    common_hashtags = [
        {'hashtag': tag, 'count': count}
        for tag, count in hashtag_counter.most_common(10)
        if count >= min_count
    ]
    
    return common_hashtags

@st.cache_data
def identify_opportunities(competitor_videos, my_videos=None):
    """Identify content opportunities based on competitor analysis"""
    if competitor_videos.empty:
        return []
    
    opportunities = []
    
    # 1. Find popular topics/themes
    keywords = analyze_title_keywords(competitor_videos.nlargest(10, 'viewCount'))
    if keywords:
        top_keywords = ", ".join([k['keyword'] for k in keywords[:5]])
        opportunities.append({
            'type': 'keywords',
            'title': 'Popular Keywords',
            'description': f"Top performing videos frequently use these keywords: {top_keywords}",
        })
    
    # 2. Extract description patterns
    description_patterns = analyze_descriptions(competitor_videos)
    if description_patterns:
        # Find a representative pattern
        pattern_example = ""
        for pattern in description_patterns:
            if pattern['type'] == 'sentence' and len(pattern['content']) < 100:
                pattern_example = pattern['content']
                break
        
        if pattern_example:
            opportunities.append({
                'type': 'description',
                'title': 'Description Strategy',
                'description': f"Top videos use similar description patterns, example: \"{pattern_example}\"",
            })
    
    # 3. Extract hashtags
    hashtags = extract_hashtags(competitor_videos)
    if hashtags:
        top_hashtags = ", ".join([h['hashtag'] for h in hashtags[:5]])
        opportunities.append({
            'type': 'hashtags',
            'title': 'Popular Hashtags',
            'description': f"Top videos use these hashtags: {top_hashtags}",
        })
    
    # 4. Check if we have similar content
    if my_videos is not None and not my_videos.empty:
        # Compare average views
        my_avg_views = my_videos['viewCount'].mean()
        competitor_avg_views = competitor_videos['viewCount'].mean()
        
        if my_avg_views < competitor_avg_views:
            gap = (competitor_avg_views - my_avg_views) / competitor_avg_views * 100
            opportunities.append({
                'type': 'performance',
                'title': 'Views Gap',
                'description': f"Your similar videos get {gap:.1f}% fewer views on average. Consider optimizing titles and thumbnails.",
            })
        
        # Check engagement gap
        my_avg_engagement = my_videos['engagementScore'].mean()
        competitor_avg_engagement = competitor_videos['engagementScore'].mean()
        
        if my_avg_engagement < competitor_avg_engagement:
            opportunities.append({
                'type': 'engagement',
                'title': 'Engagement Gap',
                'description': "Your videos have lower engagement. Focus on improving content quality and viewer interaction.",
            })
            
        # Check video frequency
        if len(my_videos) < len(competitor_videos) / 3:
            opportunities.append({
                'type': 'frequency',
                'title': 'Content Frequency',
                'description': f"Top channels publish more frequently on this topic. Consider increasing your publishing cadence.",
            })
    else:
        # We don't have content in this category
        opportunities.append({
            'type': 'new_content',
            'title': 'Content Gap',
            'description': "You don't have content on this topic yet. This could be an opportunity to expand.",
        })
    
    # 5. Identify optimal duration
    duration_analysis = competitor_videos.groupby(pd.cut(
        competitor_videos['duration_minutes'], 
        bins=[0, 5, 10, 15, 20, 30, 60, float('inf')],
        labels=['0-5 min', '5-10 min', '10-15 min', '15-20 min', '20-30 min', '30-60 min', '60+ min']
    )).agg({
        'viewCount': 'mean',
        'id': 'count'
    }).reset_index()
    
    duration_analysis.columns = ['Duration', 'Avg Views', 'Video Count']
    
    # Filter to durations with enough videos
    duration_analysis = duration_analysis[duration_analysis['Video Count'] >= 3]
    
    if not duration_analysis.empty:
        best_duration = duration_analysis.loc[duration_analysis['Avg Views'].idxmax()]
        opportunities.append({
            'type': 'duration',
            'title': 'Optimal Duration',
            'description': f"Videos between {best_duration['Duration']} perform best with an average of {int(best_duration['Avg Views']):,} views.",
        })
    
    # 6. Identify top channels
    top_channels = competitor_videos.groupby('channelTitle').agg({
        'viewCount': 'mean',
        'id': 'count'
    }).reset_index()
    
    top_channels = top_channels[top_channels['id'] >= 2]  # At least 2 videos
    
    if not top_channels.empty:
        top_channels = top_channels.nlargest(3, 'viewCount')
        channel_list = ", ".join(top_channels['channelTitle'].tolist())
        opportunities.append({
            'type': 'channels',
            'title': 'Channels to Study',
            'description': f"These channels perform well on this topic: {channel_list}. Study their content strategy."
        })
    
    return opportunities

# Main page content - Search interface
st.subheader("Search for Videos")

with st.form("youtube_search_form"):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Enter search query (e.g., 'Christian sermon', 'Bible study')")
        
        # Add date range filter
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            default_start_date = datetime.now() - timedelta(days=365)  # Default to last year
            published_after = st.date_input("Published After", value=default_start_date)
        
        with date_col2:
            published_before = st.date_input("Published Before", value=datetime.now())
    
    with col2:
        max_results = st.slider("Max results", 10, 100, 50)
        include_my_channel = st.checkbox("Include my channel", value=True)
    
    submit_search = st.form_submit_button("Search YouTube")

# Process search when submitted
if submit_search and search_query:
    # Convert dates to datetime
    published_after_dt = datetime.combine(published_after, datetime.min.time())
    published_before_dt = datetime.combine(published_before, datetime.max.time())
    
    if published_after > published_before:
        st.error("Start date must be before end date")
    else:
        with st.spinner(f"Searching YouTube for: {search_query}"):
            # Perform search with date filtering
            competitor_videos = search_youtube(
                st.session_state.api, 
                search_query, 
                max_results, 
                published_after_dt, 
                published_before_dt
            )
            
            # Get my channel's videos on this topic if requested
            my_videos = None
            if include_my_channel:
                my_videos = search_my_channel(
                    st.session_state.api, 
                    st.session_state.channel_id, 
                    search_query, 
                    published_after=published_after_dt, 
                    published_before=published_before_dt
                )
            
            # Store results in session state
            st.session_state.competitor_videos = competitor_videos
            st.session_state.my_videos = my_videos
            st.session_state.search_query = search_query
            st.session_state.date_range = f"{published_after} to {published_before}"

# Display search results if available
if 'competitor_videos' in st.session_state and not st.session_state.competitor_videos.empty:
    competitor_df = st.session_state.competitor_videos
    my_videos_df = st.session_state.my_videos if 'my_videos' in st.session_state else None
    search_query = st.session_state.search_query
    date_range = st.session_state.get('date_range', '')
    
    # Show search summary
    st.markdown(f"<h2 class='sub-header'>Results for: {search_query}</h2>", unsafe_allow_html=True)
    if date_range:
        st.markdown(f"Showing videos published during **{date_range}**")
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <p class="card-value">{len(competitor_df)}</p>
            <p class="card-label">Videos Found</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_views = int(competitor_df['viewCount'].mean())
        st.markdown(f"""
        <div class="stat-card">
            <p class="card-value">{avg_views:,}</p>
            <p class="card-label">Avg. Views</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_likes = int(competitor_df['likeCount'].mean())
        st.markdown(f"""
        <div class="stat-card">
            <p class="card-value">{avg_likes:,}</p>
            <p class="card-label">Avg. Likes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        unique_channels = competitor_df['channelTitle'].nunique()
        st.markdown(f"""
        <div class="stat-card">
            <p class="card-value">{unique_channels}</p>
            <p class="card-label">Unique Channels</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display comparison if we have our channel's videos
    if my_videos_df is not None and not my_videos_df.empty:
        st.subheader("Your Channel vs. Competitors")
        
        # Create comparison data
        comparison_data = {
            'Metric': ['Videos', 'Avg. Views', 'Avg. Likes', 'Avg. Comments', 'Avg. Engagement'],
            'Your Channel': [
                len(my_videos_df),
                int(my_videos_df['viewCount'].mean()),
                int(my_videos_df['likeCount'].mean()),
                int(my_videos_df['commentCount'].mean()),
                float(my_videos_df['engagementScore'].mean())
            ],
            'Competitors': [
                len(competitor_df),
                int(competitor_df['viewCount'].mean()),
                int(competitor_df['likeCount'].mean()),
                int(competitor_df['commentCount'].mean()),
                float(competitor_df['engagementScore'].mean())
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display as a table
        st.dataframe(
            comparison_df.style.format({
                'Your Channel': lambda x: f"{x:,}" if isinstance(x, (int, float)) else x,
                'Competitors': lambda x: f"{x:,}" if isinstance(x, (int, float)) else x
            }, subset=['Your Channel', 'Competitors']),
            use_container_width=True
        )
        
        # Visual comparison
        metrics_to_plot = ['Avg. Views', 'Avg. Likes', 'Avg. Comments', 'Avg. Engagement']
        plot_df = comparison_df[comparison_df['Metric'].isin(metrics_to_plot)].copy()
        
        # Melt for easier plotting
        plot_df = pd.melt(
            plot_df,
            id_vars=['Metric'],
            value_vars=['Your Channel', 'Competitors'],
            var_name='Source',
            value_name='Value'
        )
        
        # Create plot
        fig = px.bar(
            plot_df,
            x='Metric',
            y='Value',
            color='Source',
            barmode='group',
            title='Performance Comparison',
            color_discrete_map={
                'Your Channel': '#FF0000',
                'Competitors': '#0066CC'
            },
            text_auto='.2s'
        )
        
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Competition comparison recommendations
        view_diff_pct = ((comparison_data['Competitors'][1] - comparison_data['Your Channel'][1]) / 
                        max(1, comparison_data['Your Channel'][1])) * 100
        
        view_comparison = (
            "significantly outperform your content" if view_diff_pct > 50 else
            "perform better than your content" if view_diff_pct > 10 else
            "perform similarly to your content" if abs(view_diff_pct) <= 10 else
            "perform worse than your content"
        )
        
        engagement_diff_pct = ((comparison_data['Competitors'][4] - comparison_data['Your Channel'][4]) / 
                            max(0.001, comparison_data['Your Channel'][4])) * 100
        
        engagement_comparison = (
            "much higher engagement" if engagement_diff_pct > 30 else
            "better engagement" if engagement_diff_pct > 10 else
            "similar engagement" if abs(engagement_diff_pct) <= 10 else
            "lower engagement"
        )
        
        st.markdown(f"""
        <div class="recommendation-section">
            <p class="recommendation-title">🔍 Competitive Analysis Recommendations:</p>
            <p class="recommendation-point">1. Competitor videos {view_comparison} on this topic with {abs(view_diff_pct):.1f}% {'more' if view_diff_pct > 0 else 'fewer'} views on average.</p>
            <p class="recommendation-point">2. Competitors have {engagement_comparison} rates compared to your content.</p>
            <p class="recommendation-point">3. {'Increase your posting frequency on this topic to build audience interest.' if len(my_videos_df) < 5 else 'Continue maintaining a consistent publishing schedule to retain audience interest.'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display our videos matching the search
        st.subheader("Your Videos on This Topic")
        
        # Format for display
        display_df = my_videos_df[['title', 'viewCount', 'likeCount', 'commentCount', 'publishedAt', 'duration_minutes']].copy()
        display_df.columns = ['Title', 'Views', 'Likes', 'Comments', 'Published Date', 'Duration (min)']
        display_df['Published Date'] = pd.to_datetime(display_df['Published Date']).dt.strftime('%Y-%m-%d')
        
        # Sort by views (highest first)
        display_df = display_df.sort_values('Views', ascending=False)
        
        st.dataframe(display_df, use_container_width=True)
    
    # Top performing videos
    st.subheader("Top Performing Videos on This Topic")
    
    # Get top 10 videos by views
    top_videos = competitor_df.nlargest(10, 'viewCount').copy()
    
    # Format for display
    display_top = top_videos[['title', 'channelTitle', 'viewCount', 'likeCount', 'commentCount', 'publishedAt', 'duration_minutes']].copy()
    display_top.columns = ['Title', 'Channel', 'Views', 'Likes', 'Comments', 'Published Date', 'Duration (min)']
    display_top['Published Date'] = pd.to_datetime(display_top['Published Date']).dt.strftime('%Y-%m-%d')
    
    # Sort by views (highest first)
    display_top = display_top.sort_values('Views', ascending=False)
    
    # Show table
    st.dataframe(display_top, use_container_width=True)
    
    # Visualization of top videos
    fig = px.bar(
        top_videos,
        x='viewCount',
        y='title',
        orientation='h',
        color='channelTitle',
        title="Top 10 Videos by Views",
        hover_data=['likeCount', 'commentCount', 'duration_minutes']
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top videos recommendations
    top_video = top_videos.iloc[0] if not top_videos.empty else None
    top_title_words = analyze_title_keywords(top_videos)
    top_keywords = ", ".join([k['keyword'] for k in top_title_words[:3]]) if top_title_words else "N/A"
    
    st.markdown(f"""
    <div class="recommendation-section">
        <p class="recommendation-title">🏆 Top Content Strategy Recommendations:</p>
        <p class="recommendation-point">1. Study the format and presentation style of top-performing videos like "{top_video['title'] if top_video is not None else 'top videos'}".</p>
        <p class="recommendation-point">2. Incorporate high-performing keywords like "{top_keywords}" in your titles and descriptions.</p>
        <p class="recommendation-point">3. {'Create thumbnail styles similar to top performers to improve click-through rates.' if top_video is not None else 'Create eye-catching thumbnails with clear text overlays.'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Channel analysis
    st.subheader("Top Channels in This Category")
    
    # Aggregate by channel
    channel_stats = competitor_df.groupby('channelTitle').agg({
        'id': 'count',
        'viewCount': ['mean', 'sum'],
        'likeCount': ['mean', 'sum'],
        'commentCount': ['mean', 'sum'],
        'engagementScore': 'mean'
    }).reset_index()
    
    # Flatten multi-index columns
    channel_stats.columns = [
        'Channel', 'Videos', 'Avg Views', 'Total Views',
        'Avg Likes', 'Total Likes', 'Avg Comments', 
        'Total Comments', 'Avg Engagement'
    ]
    
    # Sort by average views
    channel_stats = channel_stats.sort_values('Avg Views', ascending=False)
    
    # Show only channels with at least 2 videos
    channel_stats = channel_stats[channel_stats['Videos'] >= 2]
    
    # Create styled dataframe
    format_dict = {
        'Videos': '{:,.0f}',
        'Avg Views': '{:,.0f}',
        'Total Views': '{:,.0f}',
        'Avg Likes': '{:,.0f}',
        'Total Likes': '{:,.0f}',
        'Avg Comments': '{:,.0f}',
        'Total Comments': '{:,.0f}',
        'Avg Engagement': '{:.4f}'
    }
    
    st.dataframe(
        channel_stats.style.format(format_dict).background_gradient(
            subset=['Avg Views', 'Avg Engagement'], 
            cmap='Reds'
        ),
        use_container_width=True
    )
    
    # Visualize top channels
    top_channels = channel_stats.head(10).copy()
    
    fig = px.bar(
        top_channels,
        x='Avg Views',
        y='Channel',
        orientation='h',
        color='Avg Engagement',
        color_continuous_scale=px.colors.sequential.Reds,
        title="Top 10 Channels by Average Views",
        hover_data=['Videos', 'Total Views']
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Top channel recommendations
    top_channel = channel_stats.iloc[0]['Channel'] if not channel_stats.empty else "top channels"
    
    st.markdown(f"""
    <div class="recommendation-section">
        <p class="recommendation-title">📡 Channel Strategy Recommendations:</p>
        <p class="recommendation-point">1. Study {top_channel}'s content strategy, posting frequency, and audience engagement techniques.</p>
        <p class="recommendation-point">2. Analyze how top channels structure their video introductions to hook viewers quickly.</p>
        <p class="recommendation-point">3. Examine their community engagement practices in comments and channel community posts.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Description and hashtag analysis
    st.subheader("Content Strategy Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Description patterns analysis
        st.markdown("##### Description Patterns in Top Videos")
        description_patterns = analyze_descriptions(competitor_df)
        
        if description_patterns:
            for pattern in description_patterns:
                pattern_type = pattern['type']
                content = pattern['content']
                count = pattern['count']
                
                if pattern_type == 'url':
                    st.markdown(f"🔗 **Common URL** ({count} videos): `{content}`")
                elif pattern_type == 'sentence':
                    st.markdown(f"📝 **Common Text** ({count} videos): '{content}'")
                elif pattern_type == 'phrase':
                    st.markdown(f"🔤 **Common Phrase** ({count} videos): '{content}'")
        else:
            st.info("Not enough data to identify common description patterns.")
    
    with col2:
        # Hashtag analysis
        st.markdown("##### Common Hashtags in Top Videos")
        hashtags = extract_hashtags(competitor_df)
        
        if hashtags:
            # Create DataFrame for visualization
            hashtags_df = pd.DataFrame(hashtags)
            
            # Create bar chart of hashtags
            fig = px.bar(
                hashtags_df,
                x='count',
                y='hashtag',
                orientation='h',
                title="Common Hashtags in Top Performing Videos",
                color='count',
                color_continuous_scale=px.colors.sequential.Reds
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No common hashtags found in top videos.")
    
    # Description and hashtag recommendations
    common_hashtags = ", ".join([h['hashtag'] for h in hashtags[:5]]) if hashtags else "#NoHashtagsFound"
    has_urls = any(p['type'] == 'url' for p in description_patterns) if description_patterns else False
    
    st.markdown(f"""
    <div class="recommendation-section">
        <p class="recommendation-title">📋 Description & Hashtag Strategy:</p>
        <p class="recommendation-point">1. {'Include common hashtags like ' + common_hashtags + ' in your descriptions for better discoverability.' if hashtags else 'Add relevant hashtags to improve discoverability in search results.'}</p>
        <p class="recommendation-point">2. {'Include key links in your descriptions as top performers do.' if has_urls else 'Add links to your social media profiles and related content in descriptions.'}</p>
        <p class="recommendation-point">3. Create a standardized description template that includes:
           <br>• A compelling first 2-3 lines that summarize the video content
           <br>• Links to your social media and related content
           <br>• Relevant hashtags to improve discoverability
           <br>• A call-to-action encouraging viewers to subscribe and engage</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Keyword analysis
    st.subheader("Popular Keywords in Titles")
    
    # Extract keywords from titles
    keywords = analyze_title_keywords(competitor_df)
    
    if keywords:
        # Create dataframe
        keywords_df = pd.DataFrame(keywords)
        
        # Visualize keywords
        fig = px.bar(
            keywords_df,
            x='count',
            y='keyword',
            orientation='h',
            color='count',
            color_continuous_scale=px.colors.sequential.Reds,
            title="Most Common Keywords in Video Titles",
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate word cloud
        wordcloud_text = ' '.join([f"{k['keyword']} " * k['count'] for k in keywords])
        
        if wordcloud_text:
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='black',
                colormap='Reds',
                min_font_size=10,
                max_font_size=150,
                random_state=42
            ).generate(wordcloud_text)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            plt.tight_layout(pad=0)
            
            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            
            st.image(buf, use_column_width=True)
            
        # Title optimization recommendations
        top_keywords = ", ".join([k['keyword'] for k in keywords[:5]])
        
        st.markdown(f"""
        <div class="recommendation-section">
            <p class="recommendation-title">✏️ Title Optimization Strategy:</p>
            <p class="recommendation-point">1. Include high-performing keywords like "{top_keywords}" in your titles.</p>
            <p class="recommendation-point">2. Create title templates that follow successful patterns from competitors:
               <br>• Include numbers when appropriate (e.g., "5 Ways to...", "3 Tips for...")
               <br>• Use power words that evoke emotion
               <br>• Keep titles between 40-60 characters for optimal display</p>
            <p class="recommendation-point">3. A/B test different title formats to see what works best with your audience.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Not enough data to generate keyword analysis.")
    
    # Video duration analysis
    st.subheader("Video Length Analysis")
    
    # Create duration bins
    bins = [0, 5, 10, 15, 20, 30, 60, float('inf')]
    labels = ['0-5 min', '5-10 min', '10-15 min', '15-20 min', '20-30 min', '30-60 min', '60+ min']
    
    competitor_df['duration_category'] = pd.cut(
        competitor_df['duration_minutes'], 
        bins=bins, 
        labels=labels
    )
    
    # Aggregate by duration category
    duration_stats = competitor_df.groupby('duration_category').agg({
        'id': 'count',
        'viewCount': 'mean',
        'likeCount': 'mean',
        'engagementScore': 'mean'
    }).reset_index()
    
    # Rename columns
    duration_stats.columns = [
        'Duration', 'Video Count', 'Avg Views', 'Avg Likes', 'Avg Engagement'
    ]
    
    # Visualize
    fig = px.bar(
        duration_stats,
        x='Duration',
        y='Avg Views',
        color='Video Count',
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Average Views by Video Duration",
        text='Video Count'
    )
    fig.update_traces(texttemplate='%{text} videos', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # Find best performing duration category
    if not duration_stats.empty and len(duration_stats[duration_stats['Video Count'] >= 3]) > 0:
        # Only consider categories with at least 3 videos
        valid_durations = duration_stats[duration_stats['Video Count'] >= 3]
        best_duration = valid_durations.loc[valid_durations['Avg Views'].idxmax()]
        
        st.markdown(f"""
        <div class="highlight">
        <p><strong>Optimal Duration:</strong> Videos in the <b>{best_duration['Duration']}</b> range perform best with an average of {int(best_duration['Avg Views']):,} views.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Duration recommendations
        st.markdown(f"""
        <div class="recommendation-section">
            <p class="recommendation-title">⏱️ Video Duration Strategy:</p>
            <p class="recommendation-point">1. Target a duration of {best_duration['Duration']} for optimal performance on this topic.</p>
            <p class="recommendation-point">2. For longer topics, consider breaking content into multiple videos in this optimal range.</p>
            <p class="recommendation-point">3. Pay special attention to viewer retention in the first 30-60 seconds, as this is when most viewers decide whether to continue watching.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Content opportunities
    st.subheader("Content Opportunities")
    
    # Generate opportunities
    opportunities = identify_opportunities(competitor_df, my_videos_df)
    
    if opportunities:
        for opportunity in opportunities:
            st.markdown(f"""
            <div class="opportunity">
            <p><strong>{opportunity['title']}:</strong> {opportunity['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Comprehensive strategy recommendations
        st.markdown("""
        <div class="recommendation-section">
            <p class="recommendation-title">🚀 Comprehensive Content Strategy:</p>
            
            <p style="font-weight:bold; margin-top:10px; margin-bottom:5px;">Content Creation:</p>
            <p class="recommendation-point">• Focus on the topics and formats that perform best in this niche.</p>
            <p class="recommendation-point">• Aim for the optimal video duration identified in the analysis.</p>
            <p class="recommendation-point">• Create a consistent publishing schedule to build audience expectations.</p>
            
            <p style="font-weight:bold; margin-top:10px; margin-bottom:5px;">SEO & Discoverability:</p>
            <p class="recommendation-point">• Use the identified high-performing keywords in titles, descriptions, and tags.</p>
            <p class="recommendation-point">• Include relevant hashtags in both video titles and descriptions.</p>
            <p class="recommendation-point">• Create eye-catching thumbnails with clear text that incorporates keywords.</p>
            
            <p style="font-weight:bold; margin-top:10px; margin-bottom:5px;">Audience Engagement:</p>
            <p class="recommendation-point">• Include clear calls-to-action in your videos to boost engagement metrics.</p>
            <p class="recommendation-point">• Respond to comments promptly to build community and improve engagement signals.</p>
            <p class="recommendation-point">• Use end screens and cards to keep viewers on your channel longer.</p>
            
            <p style="font-weight:bold; margin-top:10px; margin-bottom:5px;">Competitive Advantage:</p>
            <p class="recommendation-point">• Study top-performing channels but find ways to differentiate your content.</p>
            <p class="recommendation-point">• Fill content gaps that competitors aren't addressing.</p>
            <p class="recommendation-point">• Consider collaborations with complementary channels to expand your audience.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No clear opportunities identified. Try a more specific search query.")
    
    # Export options
    st.markdown("---")
    st.subheader("Export Data")
    
    # Create Excel report
    if st.button("Generate Excel Report"):
        with st.spinner("Generating report..."):
            # Create Excel buffer
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Metric': ['Search Query', 'Date Range', 'Videos Found', 'Unique Channels', 
                              'Average Views', 'Average Likes', 'Average Engagement'],
                    'Value': [
                        search_query,
                        date_range,
                        len(competitor_df),
                        competitor_df['channelTitle'].nunique(),
                        competitor_df['viewCount'].mean(),
                        competitor_df['likeCount'].mean(),
                        competitor_df['engagementScore'].mean()
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Recommendations sheet
                if opportunities:
                    recommendations_data = {
                        'Category': [opp['title'] for opp in opportunities],
                        'Recommendation': [opp['description'] for opp in opportunities]
                    }
                    pd.DataFrame(recommendations_data).to_excel(writer, sheet_name='Recommendations', index=False)
                
                # Top Videos sheet
                export_cols = ['title', 'channelTitle', 'viewCount', 'likeCount', 'commentCount', 
                              'publishedAt', 'duration_minutes', 'engagementScore']
                
                # Make a copy and convert timezone-aware datetime to timezone-naive
                export_df = top_videos[export_cols].copy()
                if 'publishedAt' in export_df.columns:
                    export_df['publishedAt'] = pd.DatetimeIndex(export_df['publishedAt']).tz_localize(None)
                
                export_df.to_excel(writer, sheet_name='Top Videos', index=False)
                
                # Channel Analysis sheet
                channel_stats.to_excel(writer, sheet_name='Channel Analysis', index=False)
                
                # Keywords sheet
                if keywords:
                    pd.DataFrame(keywords).to_excel(writer, sheet_name='Keywords', index=False)
                
                # Duration Analysis sheet
                duration_stats.to_excel(writer, sheet_name='Duration Analysis', index=False)
                
                # My Videos sheet (if available)
                if my_videos_df is not None and not my_videos_df.empty:
                    my_export_df = my_videos_df[export_cols].copy()
                    if 'publishedAt' in my_export_df.columns:
                        my_export_df['publishedAt'] = pd.DatetimeIndex(my_export_df['publishedAt']).tz_localize(None)
                    my_export_df.to_excel(writer, sheet_name='My Videos', index=False)
            
            # Offer download
            output.seek(0)
            st.download_button(
                label="Download Excel Report",
                data=output,
                file_name=f"competitor_analysis_{search_query.replace(' ', '_')}_{date_range.replace(' to ', '-').replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    if submit_search and search_query:
        st.warning("No videos found for this search query. Try different keywords or check your API key.")
    elif submit_search:
        st.warning("Please enter a search query.")
