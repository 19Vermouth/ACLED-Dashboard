import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import plotly.express as px
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ACLED API credentials
API_KEY = os.getenv("ACLED_ACCESS_KEY")
EMAIL = os.getenv("ACLED_EMAIL_ADDRESS")

# Validate credentials
if not API_KEY or not EMAIL:
    st.error("Missing ACLED credentials. Please set ACLED_ACCESS_KEY and ACLED_EMAIL_ADDRESS in the .env file.")
    st.markdown("""
    **Steps to fix**:
    1. Create a `.env` file in the project root with:
       ```
       ACLED_ACCESS_KEY=your_api_key
       ACLED_EMAIL_ADDRESS=your_email@example.com
       ```
    2. Get your credentials from the [ACLED Access Portal](https://developer.acleddata.com/).
    3. Ensure you’ve accepted the Terms of Use and verified your email.
    4. Test with:
       ```bash
       curl "https://api.acleddata.com/acled/read?key=your_api_key&email=your_email@example.com&country=Yemen&year=2023&limit=10"
       ```
    """)
    st.stop()

# File paths
OUTPUT_DIR = "output"
TIME_SERIES_PATH = os.path.join(OUTPUT_DIR, "conflict_events_over_time.png")
HEATMAP_PATH = os.path.join(OUTPUT_DIR, "conflict_heatmap.html")
FATALITIES_MAP_PATH = os.path.join(OUTPUT_DIR, "fatalities_map.html")

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ACLED API endpoint
BASE_URL = "https://api.acleddata.com/acled/read"

# Countries (from your provided list)
COUNTRIES = [
    "Yemen", "Syria", "Ukraine", "Nigeria", "India",
    "Afghanistan", "Iraq", "Somalia", "Pakistan", "Myanmar",
    "Israel", "Saudi Arabia", "France", "United Kingdom", "Germany"
]

# Years available (up to 2025)
YEARS = list(range(2010, 2026))


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_acled_data(country, year, limit=500):
    """Fetch ACLED data for a given country and year."""
    params = {
        "key": API_KEY,
        "email": EMAIL,
        "country": country,
        "year": year,
        "fields": "event_id_cnty|event_date|event_type|latitude|longitude|fatalities",
        "limit": limit
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()  # Raises for 4xx/5xx HTTP errors

        data = response.json()
        if data.get('status') == 200 and 'data' in data:
            df = pd.DataFrame(data['data'])
            if df.empty:
                st.warning(
                    f"No events found for {country} ({year}). Try increasing the data limit or selecting a different country/year.")
                return df
            st.success(f"Fetched {len(df)} events for {country} ({year})")
            return df
        else:
            error_msg = data.get('message', 'Unknown error')
            st.error(f"API Error: {error_msg}")
            if data.get('status') == 403:
                st.error(
                    f"Invalid email or access key for {EMAIL}. Please verify your credentials in the ACLED Access Portal (https://developer.acleddata.com/).")
                st.markdown("""
                **Troubleshooting Steps**:
                1. Check your `.env` file for correct ACLED_ACCESS_KEY and ACLED_EMAIL_ADDRESS.
                2. Verify your email address via the confirmation link from ACLED.
                3. Accept the Terms of Use in the portal.
                4. Generate a new key (revoke old key and click 'Add New Key').
                5. Contact [access@acleddata.com](mailto:access@acleddata.com) for support.
                6. Test with:
                   ```bash
                   curl "https://api.acleddata.com/acled/read?key=your_api_key&email=your_email@example.com&country=Yemen&year=2023&limit=10"
                   ```
                """)
            return pd.DataFrame()

    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again later or reduce the data limit.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return pd.DataFrame()


def compute_statistics(df):
    """Compute summary statistics for the dashboard."""
    if df.empty:
        return {
            "total_events": 0,
            "event_types": {},
            "total_fatalities": 0
        }
    total_events = len(df)
    event_types = df['event_type'].value_counts().to_dict()
    total_fatalities = df['fatalities'].astype(float).sum()
    return {
        "total_events": total_events,
        "event_types": event_types,
        "total_fatalities": total_fatalities
    }


def create_visualizations(df, country, year):
    """Create visualizations: time-series area chart, conflict heatmap, fatalities map."""
    if df.empty:
        st.warning("No data to visualize. Please check your credentials or try a different country/year.")
        return None, None, None

    # Prepare data
    df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce')

    # 1. Conflict Events Over Time by Event Type (Plotly area chart)
    df['month'] = df['event_date'].dt.to_period('M').astype(str)
    event_counts = df.groupby(['month', 'event_type']).size().reset_index(name='count')
    event_counts['month'] = pd.Categorical(event_counts['month'], ordered=True)
    fig_time_series = px.area(
        event_counts,
        x='month',
        y='count',
        color='event_type',
        title=f'Conflict Event Types Over Time in {country} ({year})',
        labels={'month': 'Month', 'count': 'Number of Events', 'event_type': 'Event Type'}
    )
    fig_time_series.update_layout(
        xaxis_tickangle=45,
        legend_title_text='Event Type',
        hovermode='x unified',
        height=600
    )

    # Save static version as PNG
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.lineplot(data=event_counts, x='month', y='count', hue='event_type', marker='o')
    plt.title(f'Conflict Event Types Over Time in {country} ({year})', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Events', fontsize=12)
    plt.legend(title='Event Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(TIME_SERIES_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Conflict Heatmap
    valid_coords = df[['latitude', 'longitude']].dropna()
    if not valid_coords.empty:
        m_heatmap = folium.Map(location=[valid_coords['latitude'].mean(), valid_coords['longitude'].mean()],
                               zoom_start=6)
        heat_data = [[row['latitude'], row['longitude']] for _, row in valid_coords.iterrows()]
        HeatMap(heat_data).add_to(m_heatmap)
        m_heatmap.save(HEATMAP_PATH)
    else:
        st.warning("No valid coordinates for Conflict Heatmap.")
        m_heatmap = None

    # 3. Fatalities Event Map
    valid_fatalities = df[['latitude', 'longitude', 'fatalities', 'event_type']].dropna()
    if not valid_fatalities.empty:
        m_fatalities = folium.Map(location=[valid_fatalities['latitude'].mean(), valid_fatalities['longitude'].mean()],
                                  zoom_start=6)
        for _, row in valid_fatalities.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=max(5, min(row['fatalities'] * 0.5, 20)),
                popup=f"Event: {row['event_type']}, Fatalities: {row['fatalities']}",
                fill=True,
                fill_opacity=0.7,
                color='red'
            ).add_to(m_fatalities)
        m_fatalities.save(FATALITIES_MAP_PATH)
    else:
        st.warning("No valid coordinates or fatalities for Fatalities Map.")
        m_fatalities = None

    return fig_time_series, m_heatmap, m_fatalities


def main():
    """Main function to run the Streamlit dashboard."""
    st.set_page_config(page_title="ACLED Conflict Dashboard", layout="wide")
    st.title("🌍 ACLED Conflict Data Dashboard")
    st.markdown("""
    This dashboard visualizes conflict events using data from the Armed Conflict Location & Event Data (ACLED) project.
    Select a country and year to view statistics, a time-series chart, a conflict heatmap, and a fatalities map.
    Data source: Armed Conflict Location & Event Data Project (ACLED); www.acleddata.com.
    """)

    # Sidebar for user inputs
    with st.sidebar:
        st.header("📊 Filters")
        country = st.selectbox("🌍 Select Country", COUNTRIES)
        year = st.selectbox("📅 Select Year", YEARS, index=YEARS.index(2023))
        limit = st.slider("Data Limit", 100, 2000, 500, 100)
        fetch_button = st.button("🔄 Fetch Data", type="primary")

        # Instructions for API errors
        with st.expander("⚠️ Troubleshooting API Errors"):
            st.markdown(f"""
            If you see errors like 'Incorrect email or access key' for {EMAIL}:
            1. Verify your email and API key in the [ACLED Access Portal](https://developer.acleddata.com/).
            2. Check your email for a verification link from ACLED.
            3. Accept the Terms of Use in the portal.
            4. Update the `.env` file with:
               ```
               ACLED_ACCESS_KEY=your_api_key
               ACLED_EMAIL_ADDRESS=your_email@example.com
               ```
            5. Contact [access@acleddata.com](mailto:access@acleddata.com) for support.
            6. Test with:
               ```bash
               curl "https://api.acleddata.com/acled/read?key=your_api_key&email=your_email@example.com&country=Yemen&year=2023&limit=10"
               ```
            """)

    # Initialize session state for data
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()

    # Fetch data when button is clicked
    if fetch_button:
        with st.spinner("🔄 Fetching data from ACLED API..."):
            st.session_state.df = fetch_acled_data(country, year, limit)

    # Display statistics
    if not st.session_state.df.empty:
        stats = compute_statistics(st.session_state.df)
        st.subheader("📊 Conflict Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events", stats["total_events"])
        col2.metric("Total Fatalities", int(stats["total_fatalities"]))
        col3.metric("Unique Event Types", len(stats["event_types"]))

        st.write("Events by Type:")
        st.dataframe(pd.DataFrame.from_dict(stats["event_types"], orient='index', columns=['Count']))

    # Create and display visualizations
    if not st.session_state.df.empty:
        st.subheader("📈 Visualizations")
        fig_time_series, m_heatmap, m_fatalities = create_visualizations(st.session_state.df, country, year)

        # Display time-series plot
        st.subheader("Conflict Event Types Over Time")
        if fig_time_series:
            st.plotly_chart(fig_time_series, use_container_width=True)
            try:
                with open(TIME_SERIES_PATH, "rb") as file:
                    st.download_button(
                        label="📥 Download PNG",
                        data=file,
                        file_name="conflict_events_over_time.png",
                        mime="image/png"
                    )
            except FileNotFoundError:
                st.error("Time-series chart file not found. Please try again.")
        else:
            st.error("Failed to generate Time-Series Chart.")

        # Display heatmap
        st.subheader("Conflict Heatmap")
        if m_heatmap:
            st.components.v1.html(m_heatmap._repr_html_(), height=500)
            try:
                with open(HEATMAP_PATH, "rb") as file:
                    st.download_button(
                        label="📥 Download HTML",
                        data=file,
                        file_name="conflict_heatmap.html",
                        mime="text/html"
                    )
            except FileNotFoundError:
                st.error("Heatmap file not found. Please try again.")
        else:
            st.warning("No valid data for Conflict Heatmap.")

        # Display fatalities map
        st.subheader("Fatalities Event Map")
        if m_fatalities:
            st.components.v1.html(m_fatalities._repr_html_(), height=500)
            try:
                with open(FATALITIES_MAP_PATH, "rb") as file:
                    st.download_button(
                        label="📥 Download HTML",
                        data=file,
                        file_name="fatalities_map.html",
                        mime="text/html"
                    )
            except FileNotFoundError:
                st.error("Fatalities map file not found. Please try again.")
        else:
            st.warning("No valid data for Fatalities Map.")
    else:
        st.info("👈 Select a country, year, and click 'Fetch Data' to view results.")

        # Sample data info
        st.subheader("ℹ️ About This Dashboard")
        st.markdown("""
        This dashboard provides analysis of conflict events using ACLED data.
        - **Time-Series Chart**: Tracks event types over months.
        - **Conflict Heatmap**: Shows event density by location.
        - **Fatalities Map**: Displays events with fatalities by location.
        Register for a free API key at [ACLED Access Portal](https://developer.acleddata.com/).
        Data source: Armed Conflict Location & Event Data Project (ACLED); www.acleddata.com.
        """)


if __name__ == "__main__":
    main()