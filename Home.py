"""
YouTube Analytics Dashboard - Main Application
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv

from youtube_api import YouTubeAPI

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF0000;
        text-align: center;
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
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .card-label {
        color: #CCCCCC;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("YOUTUBE_API_KEY", "")
    
if 'channel_id' not in st.session_state:
    st.session_state.channel_id = os.getenv("CHANNEL_ID", "")
    
if 'api_initialized' not in st.session_state:
    st.session_state.api_initialized = False

# Sidebar
st.sidebar.markdown('## YouTube Analytics')
st.sidebar.markdown('Analyze channel performance by playlist/series and compare with competitors')

# API Key and Channel ID input
with st.sidebar.expander("API Settings", expanded=not bool(st.session_state.api_key)):
    api_key = st.text_input("YouTube API Key", value=st.session_state.api_key, type="password")
    channel_id = st.text_input("Channel ID", value=st.session_state.channel_id)
    
    if st.button("Save Settings"):
        st.session_state.api_key = api_key
        st.session_state.channel_id = channel_id
        st.session_state.api_initialized = True
        st.rerun()
        
# Initialize the YouTube API client once we have the key
if st.session_state.api_initialized and st.session_state.api_key:
    try:
        st.session_state.api = YouTubeAPI(st.session_state.api_key)
    except Exception as e:
        st.error(f"Failed to initialize YouTube API: {str(e)}")
        st.session_state.api_initialized = False

# Main content
st.markdown('<h1 class="main-header">YouTube Analytics Dashboard</h1>', unsafe_allow_html=True)

# If API is not initialized, show welcome screen
if not st.session_state.api_initialized or not st.session_state.api_key:
    st.markdown("""
    ## Welcome to YouTube Analytics Dashboard
    
    This tool helps you analyze your YouTube channel performance by:
    
    1. Tracking performance by playlist/series to identify top-performing content
    2. Analyzing what content performs best based on views, likes, and engagement
    3. Comparing with competitors to identify content opportunities
    
    To get started:
    1. Get a YouTube Data API key from the [Google Cloud Console](https://console.cloud.google.com/)
    2. Enter your YouTube channel ID
    3. Save your settings in the sidebar
    """)
    st.stop()

# If we have API key and channel ID, load channel data
with st.spinner("Loading channel data..."):
    channel_info = st.session_state.api.get_channel_info(st.session_state.channel_id)
    
    if not channel_info:
        st.error("Failed to load channel data. Please check your Channel ID.")
        st.stop()
        
# Display channel information
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.image(channel_info['thumbnail'], width=150)
    
with col2:
    st.markdown(f"### {channel_info['title']}")
    st.markdown(f"*{channel_info.get('customUrl', '')}*")
    
    # Display key metrics in a single row
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.markdown(f"""
        <div class="stat-card">
            <p class="card-value">{channel_info['subscriberCount']:,}</p>
            <p class="card-label">Subscribers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown(f"""
        <div class="stat-card">
            <p class="card-value">{channel_info['viewCount']:,}</p>
            <p class="card-label">Total Views</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        st.markdown(f"""
        <div class="stat-card">
            <p class="card-value">{channel_info['videoCount']:,}</p>
            <p class="card-label">Videos</p>
        </div>
        """, unsafe_allow_html=True)
        
with col3:
    st.markdown(f"**Published:** {pd.to_datetime(channel_info['publishedAt']).strftime('%b %d, %Y')}")
    st.markdown(f"**Country:** {channel_info.get('country', 'Not specified')}")
    
    # Button to visit channel on YouTube
    st.link_button("Visit Channel", f"https://www.youtube.com/channel/{st.session_state.channel_id}")

st.markdown("---")

# Load playlists
with st.spinner("Loading playlists..."):
    playlists = st.session_state.api.get_playlists(st.session_state.channel_id)
    
    if not playlists:
        st.warning("No playlists found for this channel.")
    else:
        st.session_state.playlists = playlists
        
# Load uploads playlist videos
with st.spinner("Loading recent videos..."):
    uploads_playlist = channel_info.get('uploads')
    if uploads_playlist:
        videos = st.session_state.api.get_playlist_videos(uploads_playlist)
        
        # Get the first 10 videos for quick overview
        recent_videos = videos[:10]
        if recent_videos:
            video_ids = [video['id'] for video in recent_videos]
            video_details = st.session_state.api.get_videos_details(video_ids)
            
            if not video_details.empty:
                st.session_state.recent_videos = video_details

# Display recent videos performance
st.markdown('<h2 class="sub-header">Recent Videos Performance</h2>', unsafe_allow_html=True)

if 'recent_videos' in st.session_state and not st.session_state.recent_videos.empty:
    videos_df = st.session_state.recent_videos
    
    # Format DataFrame for display
    display_df = videos_df[['title', 'viewCount', 'likeCount', 'commentCount', 'publishedAt']].copy()
    display_df.columns = ['Title', 'Views', 'Likes', 'Comments', 'Published Date']
    display_df['Published Date'] = pd.to_datetime(display_df['Published Date']).dt.strftime('%Y-%m-%d')
    
    # Sort by published date (newest first)
    display_df = display_df.sort_values('Published Date', ascending=False)
    
    # Show the table
    st.dataframe(display_df, use_container_width=True)
    
    # Show a bar chart of views for recent videos
    fig = px.bar(
        videos_df.sort_values('publishedAt', ascending=False),
        x='title',
        y='viewCount',
        title='Views for Recent Videos',
        color='engagementScore',
        color_continuous_scale=px.colors.sequential.Reds,
        hover_data=['likeCount', 'commentCount', 'publishedAt']
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(xaxis_title='', yaxis_title='Views')
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No recent videos available to display.")

# Display playlist overview
st.markdown('<h2 class="sub-header">Playlists Overview</h2>', unsafe_allow_html=True)

if 'playlists' in st.session_state and st.session_state.playlists:
    # Create a DataFrame from playlists
    playlists_df = pd.DataFrame(st.session_state.playlists)
    
    # Format for display
    display_playlists = playlists_df[['title', 'video_count', 'publishedAt']].copy()
    display_playlists.columns = ['Playlist Name', 'Videos', 'Created Date']
    display_playlists['Created Date'] = pd.to_datetime(display_playlists['Created Date']).dt.strftime('%Y-%m-%d')
    
    # Sort by video count (most videos first)
    display_playlists = display_playlists.sort_values('Videos', ascending=False)
    
    # Show table
    st.dataframe(display_playlists, use_container_width=True)
    
    # Show pie chart of video distribution across playlists
    fig = px.pie(
        playlists_df,
        names='title',
        values='video_count',
        title='Video Distribution Across Playlists',
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display CTA for detailed analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Analyze Playlists/Series
        Dive deeper into your playlists to understand which series perform best and what content resonates with your audience.
        """)
        st.page_link("pages/playlist_analysis.py", label="Go to Playlist Analysis", icon="📊")
    
    with col2:
        st.markdown("""
        ### Compare with Competitors
        Search for similar content across YouTube to see what performs well and identify content opportunities.
        """)
        st.page_link("pages/competitor_search.py", label="Go to Competitor Analysis", icon="🔍")
else:
    st.info("No playlists available to display.")
