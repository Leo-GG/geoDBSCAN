# Geographical Clustering Visualization

A Streamlit application for visualizing geographical clusters from listings data.

**Live Demo**: [[https://geodbscan.streamlit.app/](https://geodbscan-dmicrqvck3cvpu2g7nkgzz.streamlit.app/)]([https://geodbscan.streamlit.app/](https://geodbscan-dmicrqvck3cvpu2g7nkgzz.streamlit.app/))

## Overview

This application provides an interactive visualization tool for geographical clustering analysis. It connects to a database containing geographical data points, applies DBSCAN clustering, and visualizes the results on an interactive map.

## Features

- **Interactive Clustering**: Adjust clustering parameters in real-time and see how they affect the resulting clusters
- **Map Visualization**: View clustered data points on an interactive map with color-coded clusters
- **Cluster Statistics**: Analyze detailed statistics for each cluster
- **Raw Data View**: Examine the underlying data in tabular format
- **Secure Database Connection**: Connect to your database using credentials from a secure .env file
- **Custom Database Credentials**: Temporarily use custom database credentials without modifying the .env file

## Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd Meshu
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your database credentials:
   ```
   DB_HOST=your_database_host
   DB_NAME=your_database_name
   DB_USERNAME=your_username
   DB_PASSWORD=your_password
   ```

### Running the Application

Start the Streamlit app:
```
streamlit run geo_clustering_app.py
```

## Usage

1. **Adjust Clustering Parameters**:
   - Set the maximum and target cluster diameters using the sliders in the sidebar

2. **Database Connection**:
   - The app uses database credentials from your `.env` file by default
   - To use custom credentials temporarily, click "Change DB Credentials" in the Database Options section
   - To revert to the default credentials from the `.env` file, click "Reset to Default"

3. **Data Operations**:
   - Click "Fetch Data" to retrieve data from the database
   - Click "Run Clustering" to perform clustering on the fetched data

4. **View Results**:
   - Map View: Interactive map showing color-coded clusters
   - Raw Data: Tabular view of the underlying data
   - Cluster Statistics: Detailed metrics for each cluster

## Project Structure

- `geo_clustering_app.py`: Main Streamlit application
- `geo_clustering.py`: Core clustering functionality and database operations
- `.env`: Environment variables for database credentials (not tracked in git)
- `requirements.txt`: Python dependencies

## Security Notes

- Database credentials are stored in the `.env` file, which should never be committed to version control
- The application never displays the credentials from the `.env` file in the UI
- Custom credentials are stored only in the session state and are not persisted between sessions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
