# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data visualization project analyzing Spotify music data (114,000+ tracks) using Python. The project creates comprehensive visualizations and statistical analyses of music trends, audio features, and patterns.

## Dataset Structure

- **Primary dataset**: `dataset.csv` - Contains 114,000+ tracks with 21 columns including:
  - Track metadata: artists, album_name, track_name, track_genre, popularity
  - Audio features: danceability, energy, loudness, valence, tempo, acousticness, etc.
  - Technical data: duration_ms, time_signature, key, mode, explicit

## Key Scripts

### Main Analysis Scripts
- `spotify-analysis.py` - Comprehensive 18-panel analysis dashboard with statistical tests, PCA, clustering, and multiple visualizations
- `spotify-12-stories.py` - Narrative-driven analysis creating 12 separate story visualizations

### Data Processing
- `sampling/sample_data.py` - Creates stratified sample of 500 tracks across genres and popularity levels
- Outputs `sampling/sample_data.json` for web visualizations

## Dependencies

The project uses standard data science libraries:
- pandas, numpy - Data manipulation
- matplotlib, seaborn - Visualization
- scipy - Statistical analysis
- sklearn - Machine learning (PCA, clustering, Random Forest)

## Running Analysis

Execute main scripts directly:
```bash
python spotify-analysis.py          # Generates comprehensive analysis
python spotify-12-stories.py        # Creates story-based visualizations
python sampling/sample_data.py      # Creates data sample
```

## Output Files

- `spotify_analysis_award_winning.png` - Main comprehensive dashboard
- `spotify_12_stories_complete.png` - Story-based analysis
- `output/` directory - Individual story visualizations (story1_dance_science.png, etc.)

## Code Architecture

- **Visualization Style**: Uses seaborn-v0_8-darkgrid with custom color palettes
- **Data Processing**: Cleans column names, handles missing data, creates derived features
- **Analysis Pattern**: Each visualization combines descriptive statistics with inferential testing
- **Statistical Methods**: Hypothesis testing, correlation analysis, PCA, clustering, Random Forest feature importance