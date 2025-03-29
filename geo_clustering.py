"""
Geographical Clustering Module for MongoDB Listings

This module provides functionality to cluster geographical points from a database
based on proximity using DBSCAN algorithm with Haversine distance.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy import create_engine
import math
from typing import List, Dict, Union, Optional, Tuple, Any
from urllib.parse import quote_plus
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeoUtils:
    """Utility class for geographical calculations."""

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the Haversine distance between two points on the earth.
        
        Args:
            lat1: Latitude of first point in degrees
            lon1: Longitude of first point in degrees
            lat2: Latitude of second point in degrees
            lon2: Longitude of second point in degrees
            
        Returns:
            Distance between the points in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r


class HaversineDistance:
    """Custom metric for DBSCAN that calculates Haversine distance between points."""
    
    def __call__(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Haversine distance between two points."""
        return GeoUtils.haversine_distance(p1[0], p1[1], p2[0], p2[1])


class DatabaseConnector:
    """Class to handle database connection and data retrieval."""
    
    def __init__(
        self,
        host: str = "http://reservation-history.c2wevclshsed.us-east-1.rds.amazonaws.com",
        db_name: str = "reservation_history",
        username: str = "admin-user-2",
        password: str = "k#8zP@vW7q$Y2fN5xG!e"
    ):
        """
        Initialize the database connector.
        
        Args:
            host: Database host URL
            db_name: Database name
            username: Database username
            password: Database password
        """
        self.host = host
        self.db_name = db_name
        self.username = username
        self.password = password
        self.engine = None
    
    def get_connection_string(self) -> str:
        """
        Create and return a properly formatted connection string.
        
        Returns:
            Database connection string
        """
        # Extract the hostname from the URL without protocol
        if self.host.startswith("http://"):
            hostname = self.host.replace("http://", "")
        elif self.host.startswith("https://"):
            hostname = self.host.replace("https://", "")
        else:
            hostname = self.host
        
        # Create database connection with properly escaped password
        password_encoded = quote_plus(self.password)
        return f"mysql+pymysql://{self.username}:{password_encoded}@{hostname}/{self.db_name}"
    
    def connect(self) -> None:
        """Establish a connection to the database."""
        try:
            connection_string = self.get_connection_string()
            self.engine = create_engine(connection_string)
            # Test the connection
            with self.engine.connect() as conn:
                logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def fetch_data(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch data from the specified table.
        
        Args:
            table_name: Name of the table to fetch data from
            columns: List of columns to fetch, None for all columns
            
        Returns:
            DataFrame containing the fetched data
        """
        if self.engine is None:
            self.connect()
        
        try:
            if columns:
                column_str = ", ".join(columns)
                query = f"SELECT {column_str} FROM {table_name}"
            else:
                query = f"SELECT * FROM {table_name}"
            
            logger.info(f"Executing query: {query}")
            df = pd.read_sql(query, self.engine)
            logger.info(f"Fetched {len(df)} rows from {table_name}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data from {table_name}: {str(e)}")
            raise


class GeoClusterer:
    """Class to perform geographical clustering on data points."""
    
    def __init__(self):
        """Initialize the GeoClusterer."""
        pass
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare data for clustering.
        
        Args:
            df: DataFrame containing the data to validate
            
        Returns:
            Validated and prepared DataFrame
            
        Raises:
            ValueError: If required columns are missing
        """
        # Check if latitude and longitude columns exist
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            raise ValueError("The data must contain 'latitude' and 'longitude' columns")
        
        # Drop rows with missing coordinates and create an explicit copy
        df_clean = df.dropna(subset=['latitude', 'longitude']).copy()
        
        # If there's no data, raise an error
        if df_clean.empty:
            raise ValueError("No valid coordinates found in the data")
        
        return df_clean
    
    def cluster_data(
        self, 
        df: pd.DataFrame, 
        max_cluster_diameter_km: float = 1.0,
        target_cluster_diameter_km: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Cluster data points based on geographical proximity.
        
        Args:
            df: DataFrame containing the data to cluster
            max_cluster_diameter_km: Maximum distance between points in a cluster
            target_cluster_diameter_km: Desired average size for clusters (optional)
            
        Returns:
            DataFrame with cluster labels added
        """
        # Validate data
        df_clean = self.validate_data(df)
        
        # Extract coordinates for clustering
        coordinates = df_clean[['latitude', 'longitude']].values
        
        # Calculate epsilon (max distance between points in a cluster)
        epsilon = max_cluster_diameter_km / 2
        
        # If epsilon is 0, we need a small value to group only identical points
        if epsilon == 0:
            epsilon = 1e-10
        
        # Adjust min_samples based on target_cluster_diameter if provided
        min_samples = 2  # Default: at least 2 points to form a cluster
        if target_cluster_diameter_km is not None and target_cluster_diameter_km > 0:
            # Heuristic: adjust min_samples based on the ratio of max to target diameter
            ratio = max(1, max_cluster_diameter_km / target_cluster_diameter_km)
            min_samples = max(2, int(2 * ratio))
        
        # Apply DBSCAN clustering with Haversine distance
        logger.info(f"Running DBSCAN with eps={epsilon}, min_samples={min_samples}")
        db = DBSCAN(
            eps=epsilon,
            min_samples=min_samples,
            metric=HaversineDistance()
        ).fit(coordinates)
        
        # Get cluster labels
        labels = db.labels_
        
        # Add cluster labels to the dataframe
        df_clean.loc[:, 'cluster'] = labels
        
        # Log clustering results
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        logger.info(f"Found {n_clusters} clusters and {n_noise} noise points")
        
        return df_clean
    
    @staticmethod
    def format_clusters(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Format clustered data into a structured output.
        
        Args:
            df: DataFrame with cluster labels
            
        Returns:
            List of clusters, each containing the rows that belong to that group
        """
        clusters = []
        
        # Group by cluster
        for cluster_id, group in df.groupby('cluster'):
            # Skip noise points (cluster_id = -1)
            if cluster_id != -1:
                clusters.append({
                    'cluster_id': int(cluster_id),
                    'count': len(group),
                    'listings': group.to_dict('records')
                })
        
        return clusters


def cluster_mongo_listings(
    host: str = "http://reservation-history.c2wevclshsed.us-east-1.rds.amazonaws.com",
    db_name: str = "reservation_history",
    username: str = "admin-user-2",
    password: str = "k#8zP@vW7q$Y2fN5xG!e",
    max_cluster_diameter_km: float = 1.0,
    target_cluster_diameter_km: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Group rows from mongo_listings table into clusters based on proximity.
    
    Args:
        host: Database host URL
        db_name: Database name
        username: Database username
        password: Database password
        max_cluster_diameter_km: Maximum distance (in kilometers) between the two 
                                farthest points in any given cluster
        target_cluster_diameter_km: Desired average size for clusters (optional)
        
    Returns:
        A list of clusters, each containing the rows that belong to that group
    """
    try:
        # Initialize database connector
        db_connector = DatabaseConnector(host, db_name, username, password)
        
        # Fetch data from mongo_listings table
        df = db_connector.fetch_data('mongo_listings')
        
        # Initialize clusterer
        clusterer = GeoClusterer()
        
        # Perform clustering
        df_clustered = clusterer.cluster_data(
            df, 
            max_cluster_diameter_km, 
            target_cluster_diameter_km
        )
        
        # Format and return clusters
        return clusterer.format_clusters(df_clustered)
    
    except Exception as e:
        logger.error(f"Error in cluster_mongo_listings: {str(e)}")
        raise


def explain_clustering_method() -> str:
    """
    Provides an explanation of the clustering method used.
    
    Returns:
        A string explaining the clustering method
    """
    explanation = """
    Clustering Method Explanation:
    
    This function uses DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 
    with a custom Haversine distance metric for geographical clustering.
    
    Why DBSCAN?
    - It doesn't require specifying the number of clusters in advance
    - It can find arbitrarily shaped clusters
    - It can identify outliers as noise
    - It's well-suited for spatial data
    
    The Haversine formula is used to calculate the great-circle distance between two points 
    on a sphere (Earth), accounting for the Earth's curvature. This provides accurate 
    distance measurements between geographical coordinates.
    
    Parameters:
    - epsilon (eps): Set to half of max_cluster_diameter_km, as the diameter is the maximum
      distance between any two points in a cluster
    - min_samples: Minimum number of points required to form a dense region (cluster)
    
    When max_cluster_diameter_km is set to 0, only points with identical or nearly identical 
    coordinates are grouped together. With larger values, the clustering becomes more inclusive,
    potentially grouping points across wider geographical areas.
    """
    return explanation


if __name__ == "__main__":
    try:
        # Example usage
        clusters = cluster_mongo_listings(max_cluster_diameter_km=10.0)
        print(f"Found {len(clusters)} clusters")
        
        # Print explanation
        print(explain_clustering_method())
    except Exception as e:
        logger.error(f"Error running script: {str(e)}")
        print(f"Error: {str(e)}")
