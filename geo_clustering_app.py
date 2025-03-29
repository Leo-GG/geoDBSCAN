"""
Streamlit App for Geographical Clustering Visualization

This app visualizes the clustered geographical data from the mongo_listings table
on an interactive map, with points colored by cluster.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from geo_clustering import DatabaseConnector, GeoClusterer, GeoUtils

# Set page configuration
st.set_page_config(
    page_title="Geo Clustering Visualization",
    page_icon="🌎",
    layout="wide"
)

# App title and description
st.title("Geographical Clustering Visualization")
st.markdown("""
This app visualizes geographical clusters from the mongo_listings database.
Adjust the clustering parameters to see how they affect the resulting clusters.
""")

# Sidebar for parameters
st.sidebar.header("Clustering Parameters")

# Database connection parameters
st.sidebar.subheader("Database Connection")
with st.sidebar.expander("Database Settings", expanded=False):
    host = st.text_input(
        "Host URL", 
        value="http://reservation-history.c2wevclshsed.us-east-1.rds.amazonaws.com"
    )
    db_name = st.text_input("Database Name", value="reservation_history")
    username = st.text_input("Username", value="admin-user-2")
    password = st.text_input("Password", value="k#8zP@vW7q$Y2fN5xG!e", type="password")

# Clustering parameters
max_cluster_diameter = st.sidebar.slider(
    "Max Cluster Diameter (km)",
    min_value=0.1,
    max_value=50.0,
    value=10.0,
    step=0.1,
    help="Maximum distance between any two points in a cluster"
)

target_cluster_diameter = st.sidebar.slider(
    "Target Cluster Diameter (km)",
    min_value=0.0,
    max_value=25.0,
    value=5.0,
    step=0.1,
    help="Desired average size for clusters (not enforced, but guides optimization)"
)

# Function to load and cluster data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_and_cluster_data(host, db_name, username, password, max_diameter, target_diameter):
    """Load data from database and perform clustering"""
    try:
        # Initialize database connector
        db_connector = DatabaseConnector(host, db_name, username, password)
        
        # Fetch data
        df = db_connector.fetch_data('mongo_listings')
        
        # Initialize clusterer
        clusterer = GeoClusterer()
        
        # Perform clustering
        df_clustered = clusterer.cluster_data(
            df, 
            max_cluster_diameter_km=max_diameter, 
            target_cluster_diameter_km=target_diameter
        )
        
        # Get cluster statistics
        cluster_stats = get_cluster_statistics(df_clustered)
        
        return df, df_clustered, cluster_stats
    except Exception as e:
        st.error(f"Error loading or clustering data: {str(e)}")
        return None, None, None

def get_cluster_statistics(df):
    """Calculate statistics for each cluster"""
    if 'cluster' not in df.columns:
        return None
    
    # Skip noise points (cluster_id = -1)
    df_clusters = df[df['cluster'] != -1]
    
    # Group by cluster and calculate statistics
    stats = []
    for cluster_id, group in df_clusters.groupby('cluster'):
        # Calculate cluster center
        center_lat = group['latitude'].mean()
        center_lon = group['longitude'].mean()
        
        # Calculate maximum distance between any two points in the cluster
        max_distance = 0
        if len(group) > 1:
            for i, row1 in group.iterrows():
                for j, row2 in group.iterrows():
                    if i < j:  # Avoid calculating the same pair twice
                        dist = GeoUtils.haversine_distance(
                            row1['latitude'], row1['longitude'],
                            row2['latitude'], row2['longitude']
                        )
                        max_distance = max(max_distance, dist)
        
        stats.append({
            'cluster_id': int(cluster_id),
            'count': len(group),
            'center_lat': center_lat,
            'center_lon': center_lon,
            'max_distance_km': max_distance,
            'avg_latitude': group['latitude'].mean(),
            'avg_longitude': group['longitude'].mean(),
            'min_latitude': group['latitude'].min(),
            'max_latitude': group['latitude'].max(),
            'min_longitude': group['longitude'].min(),
            'max_longitude': group['longitude'].max()
        })
    
    return pd.DataFrame(stats)

# Button to fetch data
if st.sidebar.button("Fetch Data"):
    with st.spinner("Fetching data..."):
        # Load data
        df = DatabaseConnector(host, db_name, username, password).fetch_data('mongo_listings')
        
        if df is not None:
            # Store the data in session state for later use
            st.session_state['df'] = df
            st.success("Data fetched successfully!")

# Button to trigger clustering
if st.sidebar.button("Run Clustering"):
    with st.spinner("Performing clustering..."):
        # Load data from session state
        df = st.session_state.get('df')
        
        if df is not None:
            # Initialize clusterer
            clusterer = GeoClusterer()
            
            # Perform clustering
            df_clustered = clusterer.cluster_data(
                df, 
                max_cluster_diameter_km=max_cluster_diameter, 
                target_cluster_diameter_km=target_cluster_diameter
            )
            
            # Get cluster statistics
            cluster_stats = get_cluster_statistics(df_clustered)
            
            if df_clustered is not None:
                # Store the data in session state for later use
                st.session_state['df_clustered'] = df_clustered
                st.session_state['cluster_stats'] = cluster_stats
                st.success(f"Clustering complete! Found {len(cluster_stats)} clusters.")
        else:
            # Load data and cluster
            df, df_clustered, cluster_stats = load_and_cluster_data(
                host, db_name, username, password, 
                max_cluster_diameter, target_cluster_diameter
            )
            
            if df_clustered is not None:
                # Store the data in session state for later use
                st.session_state['df'] = df
                st.session_state['df_clustered'] = df_clustered
                st.session_state['cluster_stats'] = cluster_stats
                st.success(f"Clustering complete! Found {len(cluster_stats)} clusters.")

# Display results if data is available
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Map View", "Raw Data", "Cluster Statistics"])
    
    with tab1:
        st.subheader("Map Visualization")
        
        # Prepare data for map
        df_map = df.copy()
        
        # Map view options
        map_type = st.radio(
            "Map Type", 
            options=["Scatter Plot"], 
            horizontal=True
        )
        
        # Check if clustered data is available
        if 'df_clustered' in st.session_state:
            show_clusters = st.checkbox("Show Clusters", value=True)
            if show_clusters:
                df_map = st.session_state['df_clustered'].copy()
                # Skip noise points for visualization
                df_map = df_map[df_map['cluster'] != -1]
                # Convert cluster to string for better coloring
                df_map['cluster_str'] = df_map['cluster'].astype(str)
                
                # Create scatter plot with clusters
                fig = px.scatter_mapbox(
                    df_map, 
                    lat="latitude", 
                    lon="longitude", 
                    color="cluster_str",
                    hover_name="cluster_str",
                    hover_data=["latitude", "longitude"],
                    zoom=3,
                    height=600,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
            else:
                # Create scatter plot without clusters
                fig = px.scatter_mapbox(
                    df_map, 
                    lat="latitude", 
                    lon="longitude", 
                    hover_name="latitude",
                    hover_data=["latitude", "longitude"],
                    zoom=3,
                    height=600,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
        else:
            # Create scatter plot without clusters
            fig = px.scatter_mapbox(
                df_map, 
                lat="latitude", 
                lon="longitude", 
                hover_name="latitude",
                hover_data=["latitude", "longitude"],
                zoom=3,
                height=600,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
        
        fig.update_layout(
            mapbox_style="carto-positron",
            margin={"r":0, "t":0, "l":0, "b":0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Raw Data")
        
        # Display the data
        st.dataframe(df, use_container_width=True)
    
    if 'df_clustered' in st.session_state and 'cluster_stats' in st.session_state:
        df_clustered = st.session_state['df_clustered']
        cluster_stats = st.session_state['cluster_stats']
        
        with tab3:
            st.subheader("Cluster Statistics")
            
            # Display cluster statistics
            st.dataframe(cluster_stats, use_container_width=True)
            
            # Visualize cluster sizes
            st.subheader("Cluster Size Distribution")
            fig = px.bar(
                cluster_stats, 
                x='cluster_id', 
                y='count',
                labels={'cluster_id': 'Cluster ID', 'count': 'Number of Points'},
                color='count',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualize cluster diameters
            st.subheader("Cluster Diameter Distribution")
            fig = px.bar(
                cluster_stats, 
                x='cluster_id', 
                y='max_distance_km',
                labels={'cluster_id': 'Cluster ID', 'max_distance_km': 'Maximum Distance (km)'},
                color='max_distance_km',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    # Display instructions if no data is loaded yet
    st.info("👈 Adjust the parameters in the sidebar and click 'Fetch Data' to start.")

# Add explanation of the clustering method
with st.sidebar.expander("About the Clustering Method", expanded=False):
    st.markdown("""
    ### DBSCAN with Haversine Distance
    
    This app uses DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 
    with a custom Haversine distance metric for geographical clustering.
    
    **Why DBSCAN?**
    - It doesn't require specifying the number of clusters in advance
    - It can find arbitrarily shaped clusters
    - It can identify outliers as noise
    - It's well-suited for spatial data
    
    **Parameters:**
    - **Max Cluster Diameter**: Maximum distance between any two points in a cluster
    - **Target Cluster Diameter**: Guides the optimization of cluster sizes
    
    The Haversine formula calculates the great-circle distance between two points 
    on a sphere (Earth), accounting for the Earth's curvature.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Geo Clustering App")
