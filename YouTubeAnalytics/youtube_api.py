"""
YouTube API wrapper for accessing YouTube Data API v3.
Handles authentication and provides methods for accessing channel, playlist, and video data.
"""
import os
import googleapiclient.discovery
import pandas as pd
import re
from typing import Dict, List, Optional, Any, Union


class YouTubeAPI:
    """Wrapper for the YouTube Data API v3."""
    
    def __init__(self, api_key: str):
        """
        Initialize the YouTube API client.
        
        Args:
            api_key: YouTube Data API key
        """
        self.api_key = api_key
        self.youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=api_key
        )
    
    def get_channel_info(self, channel_id: str) -> Dict[str, Any]:
        """
        Get basic information about a YouTube channel.
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            Dictionary containing channel details
        """
        try:
            request = self.youtube.channels().list(
                part="snippet,contentDetails,statistics,brandingSettings",
                id=channel_id
            )
            response = request.execute()
            
            if not response.get('items'):
                return {}
                
            channel = response["items"][0]
            
            channel_details = {
                "title": channel["snippet"]["title"],
                "description": channel["snippet"]["description"],
                "customUrl": channel["snippet"].get("customUrl", ""),
                "viewCount": int(channel["statistics"].get("viewCount", 0)),
                "subscriberCount": int(channel["statistics"].get("subscriberCount", 0)),
                "videoCount": int(channel["statistics"].get("videoCount", 0)),
                "uploads": channel['contentDetails']['relatedPlaylists']['uploads'],
                "thumbnail": channel['snippet']['thumbnails'].get('medium', {}).get('url', ''),
                "country": channel["snippet"].get("country", ""),
                "publishedAt": channel["snippet"]["publishedAt"],
                "keywords": channel.get("brandingSettings", {}).get("channel", {}).get("keywords", "")
            }
            
            return channel_details
        except Exception as e:
            print(f"Error getting channel info: {str(e)}")
            return {}
    
    def get_playlists(self, channel_id: str) -> List[Dict[str, Any]]:
        """
        Get all playlists from a channel.
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            List of playlist dictionaries
        """
        try:
            playlists = []
            next_page_token = None
            
            while True:
                request = self.youtube.playlists().list(
                    part="snippet,contentDetails",
                    channelId=channel_id,
                    maxResults=50,
                    pageToken=next_page_token
                )
                response = request.execute()
                
                for playlist in response.get('items', []):
                    playlist_data = {
                        'id': playlist['id'],
                        'title': playlist['snippet']['title'],
                        'description': playlist['snippet']['description'],
                        'publishedAt': playlist['snippet']['publishedAt'],
                        'thumbnail': playlist['snippet']['thumbnails'].get('medium', {}).get('url', ''),
                        'video_count': playlist['contentDetails']['itemCount']
                    }
                    playlists.append(playlist_data)
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            return playlists
        except Exception as e:
            print(f"Error getting playlists: {str(e)}")
            return []
    
    def get_playlist_videos(self, playlist_id: str) -> List[Dict[str, Any]]:
        """
        Get all videos from a playlist.
        
        Args:
            playlist_id: YouTube playlist ID
            
        Returns:
            List of video dictionaries
        """
        try:
            videos = []
            next_page_token = None
            
            while True:
                request = self.youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                )
                response = request.execute()
                
                for item in response.get('items', []):
                    video_data = {
                        'id': item['contentDetails']['videoId'],
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'publishedAt': item['snippet']['publishedAt'],
                        'thumbnail': item['snippet']['thumbnails'].get('medium', {}).get('url', ''),
                        'position': item['snippet']['position'],
                        'playlist_id': playlist_id
                    }
                    videos.append(video_data)
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            return videos
        except Exception as e:
            print(f"Error getting playlist videos: {str(e)}")
            return []
    
    def get_videos_details(self, video_ids: List[str]) -> pd.DataFrame:
        """
        Get detailed information about a list of videos.
        
        Args:
            video_ids: List of YouTube video IDs
            
        Returns:
            DataFrame with video details
        """
        if not video_ids:
            return pd.DataFrame()
            
        try:
            all_video_data = []
            
            # Process in batches of 50 (API limit)
            for i in range(0, len(video_ids), 50):
                batch_ids = video_ids[i:i+50]
                
                request = self.youtube.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=",".join(batch_ids)
                )
                response = request.execute()
                
                for video in response.get('items', []):
                    # Extract basic info
                    video_data = {
                        'id': video['id'],
                        'title': video['snippet']['title'],
                        'description': video['snippet']['description'],
                        'publishedAt': video['snippet']['publishedAt'],
                        'channelId': video['snippet']['channelId'],
                        'channelTitle': video['snippet']['channelTitle'],
                        'tags': video['snippet'].get('tags', []),
                        'categoryId': video['snippet'].get('categoryId', ''),
                        'duration': video['contentDetails']['duration'],
                        'viewCount': int(video['statistics'].get('viewCount', 0)),
                        'likeCount': int(video['statistics'].get('likeCount', 0)),
                        'commentCount': int(video['statistics'].get('commentCount', 0)),
                        'thumbnail': video['snippet']['thumbnails'].get('high', {}).get('url', '')
                    }
                    all_video_data.append(video_data)
            
            # Convert to DataFrame
            if not all_video_data:
                return pd.DataFrame()
                
            df = pd.DataFrame(all_video_data)
            
            # Add computed fields
            if not df.empty:
                # Convert duration to minutes
                df['duration_minutes'] = df['duration'].apply(self._parse_duration)
                
                # Calculate engagement metrics
                df['likeRatio'] = df['likeCount'] / df['viewCount'].replace(0, 1)
                df['commentRatio'] = df['commentCount'] / df['viewCount'].replace(0, 1)
                df['engagementScore'] = (df['likeRatio'] * 0.7) + (df['commentRatio'] * 0.3)
                
                # Convert dates
                df['publishedAt'] = pd.to_datetime(df['publishedAt'])
            
            return df
            
        except Exception as e:
            print(f"Error getting video details: {str(e)}")
            return pd.DataFrame()
    
    def search_videos(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Search for videos matching a query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of video dictionaries
        """
        try:
            videos = []
            next_page_token = None
            results_count = 0
            
            while results_count < max_results:
                request = self.youtube.search().list(
                    part="snippet",
                    q=query,
                    type="video",
                    maxResults=min(50, max_results - results_count),
                    pageToken=next_page_token,
                    order="relevance"
                )
                response = request.execute()
                
                for item in response.get('items', []):
                    video_data = {
                        'id': item['id']['videoId'],
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'publishedAt': item['snippet']['publishedAt'],
                        'channelId': item['snippet']['channelId'],
                        'channelTitle': item['snippet']['channelTitle'],
                        'thumbnail': item['snippet']['thumbnails'].get('high', {}).get('url', '')
                    }
                    videos.append(video_data)
                    results_count += 1
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token or results_count >= max_results:
                    break
                    
            return videos
        except Exception as e:
            print(f"Error searching videos: {str(e)}")
            return []
    
    def _parse_duration(self, duration_str: str) -> float:
        """
        Parse ISO 8601 duration format to minutes.
        
        Args:
            duration_str: Duration in ISO 8601 format (e.g. PT1H30M15S)
            
        Returns:
            Duration in minutes
        """
        hours = re.search(r'(\d+)H', duration_str)
        minutes = re.search(r'(\d+)M', duration_str)
        seconds = re.search(r'(\d+)S', duration_str)
        
        total_minutes = 0
        if hours:
            total_minutes += int(hours.group(1)) * 60
        if minutes:
            total_minutes += int(minutes.group(1))
        if seconds:
            total_minutes += int(seconds.group(1)) / 60
            
        return total_minutes
