# LLM-DataViz: Spotify Music Analytics
The visualisation is available [here](https://rishabhmakes.github.io/llm-dataviz/)

## Project Overview

LLM-DataViz is a comprehensive data visualization and analysis project that explores patterns and insights from a dataset of 114,000+ Spotify tracks. The project combines Python-based data analysis with interactive web visualizations to tell compelling data stories about music trends, audio features, and listening patterns.

## Key Features

- **Comprehensive Analytics Dashboard**: Statistical analysis of 114,000+ Spotify tracks
- **12 Data Stories**: Narrative-driven visualizations exploring specific music insights
- **Interactive Web Interface**: D3.js-powered visualizations for exploring the data
- **Machine Learning Insights**: Includes PCA, clustering, and Random Forest analysis

## Dataset

The project uses a comprehensive Spotify dataset (`dataset.csv`) containing 114,000+ tracks with 21 columns including:
- Track metadata: artists, album_name, track_name, track_genre, popularity
- Audio features: danceability, energy, loudness, valence, tempo, acousticness, etc.
- Technical data: duration_ms, time_signature, key, mode, explicit

## Project Structure

```
├── dataset.csv                       # Main dataset (114,000+ Spotify tracks)
├── spotify-analysis.py               # Comprehensive analysis dashboard script
├── spotify-12-stories.py             # Narrative-driven analysis script
├── spotify_analysis_award_winning.png # Main analysis dashboard output
├── spotify_12_stories_complete.png   # Combined story visualizations
├── index.html                        # Interactive web visualization
├── output/                           # Individual story visualization outputs
│   ├── story1_dance_science.png
│   ├── story2_happiness_index.png
│   └── ...
└── sampling/                         # Data sampling for web visualizations
    ├── sample_data.py                # Creates sample dataset
    └── sample_data.json              # Sample data for web visualization
```

## Installation & Requirements

The project requires the following Python libraries:
```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
```

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## Usage

### Running Analysis Scripts

Generate the comprehensive analysis dashboard:
```bash
python spotify-analysis.py
```

Generate the 12 data stories:
```bash
python spotify-12-stories.py
```

Create data sample for web visualization:
```bash
python sampling/sample_data.py
```

### Viewing the Web Visualization

Open `index.html` in a web browser to view the interactive data visualizations.

## Data Stories

The project includes 12 narrative-driven data stories:

1. **The Science of What Makes You Dance**: Analysis of danceability features
2. **The Happiness Index**: Exploring valence and emotional content in music
3. **The Loudness Wars**: Historical trends in music loudness
4. **Artist Analysis**: Comparing audio signatures of popular artists
5. **The Three-Minute Rule**: Analysis of track length patterns
6. **Explicit Content Trends**: Patterns in explicit content across genres
7. **The Instrumental Underground**: Analysis of instrumental music trends
8. **Major vs Minor: The Emotional DNA**: Musical mode analysis
9. **Algorithm Favorites**: Characteristics of algorithm-friendly tracks
10. **Live vs Studio**: Comparing live and studio recordings
11. **The Acoustic Revival**: Trends in acoustic music
12. **Speech-Music Hybrid**: Analysis of spoken word in music

## License

[Add appropriate license information]

## Contributors

[Add contributor information]

## Acknowledgments

- Data sourced from Spotify API
- Visualization inspiration from award-winning data journalism
