"""
Competitor Analysis Page
Search for and analyze similar content across YouTube to identify opportunities.
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
def search_youtube(_api, query, max_results=50):
    """Search YouTube and get video details"""
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
    
    return details_df

@st.cache_data(ttl=3600)
def search_my_channel(_api, channel_id, query, max_results=10):
    """Search for videos on my channel matching the query"""
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
    
    # 2. Check if we have similar content
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
    else:
        # We don't have content in this category
        opportunities.append({
            'type': 'new_content',
            'title': 'Content Gap',
            'description': "You don't have content on this topic yet. This could be an opportunity to expand.",
        })
    
    # 3. Identify optimal duration
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
    
    # 4. Identify top channels
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
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search_query = st.text_input("Enter search query (e.g., 'Joshua Heward-Mills sermon')")
    
    with col2:
        max_results = st.slider("Max results", 10, 100, 50)
    
    with col3:
        include_my_channel = st.checkbox("Include my channel", value=True)
    
    submit_search = st.form_submit_button("Search YouTube")

# Process search when submitted
if submit_search and search_query:
    with st.spinner(f"Searching YouTube for: {search_query}"):
        # Perform search
        competitor_videos = search_youtube(st.session_state.api, search_query, max_results)
        
        # Get my channel's videos on this topic if requested
        my_videos = None
        if include_my_channel:
            my_videos = search_my_channel(st.session_state.api, st.session_state.channel_id, search_query)
        
        # Store results in session state
        st.session_state.competitor_videos = competitor_videos
        st.session_state.my_videos = my_videos
        st.session_state.search_query = search_query

# Display search results if available
if 'competitor_videos' in st.session_state and not st.session_state.competitor_videos.empty:
    competitor_df = st.session_state.competitor_videos
    my_videos_df = st.session_state.my_videos if 'my_videos' in st.session_state else None
    search_query = st.session_state.search_query
    
    # Show search summary
    st.markdown(f"<h2 class='sub-header'>Results for: {search_query}</h2>", unsafe_allow_html=True)
    
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
                    'Metric': ['Search Query', 'Videos Found', 'Unique Channels', 
                              'Average Views', 'Average Likes', 'Average Engagement'],
                    'Value': [
                        search_query,
                        len(competitor_df),
                        competitor_df['channelTitle'].nunique(),
                        competitor_df['viewCount'].mean(),
                        competitor_df['likeCount'].mean(),
                        competitor_df['engagementScore'].mean()
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
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
                file_name=f"competitor_analysis_{search_query.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    if submit_search and search_query:
        st.warning("No videos found for this search query. Try different keywords or check your API key.")
    elif submit_search:
        st.warning("Please enter a search query.")
