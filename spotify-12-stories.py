import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set up the aesthetic style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define beautiful color palettes
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#F8B500']
gradient_colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140', '#30cfd0']

# Read the dataset
df = pd.read_csv('dataset.csv')
df.columns = df.columns.str.strip()

# Story 1: The Science of What Makes You Dance
print("Story 1: The Science of What Makes You Dance")
# Create figure for Story 1
fig1 = plt.figure(figsize=(18, 6))
fig1.suptitle('The Science of What Makes You Dance', fontsize=20, fontweight='bold', y=1.05)

# Subplot 1a: Correlation heatmap
ax1a = plt.subplot(1, 3, 1)
dance_features = ['danceability', 'energy', 'tempo', 'valence', 'loudness', 'speechiness']
dance_corr = df[dance_features].corr()
sns.heatmap(dance_corr, annot=True, cmap='RdYlBu_r', center=0, square=True, 
            linewidths=1, cbar_kws={"shrink": .8}, ax=ax1a)
ax1a.set_title('Dance Factor Correlations', fontsize=14, fontweight='bold')

# Subplot 1b: Sweet spot analysis
ax1b = plt.subplot(1, 3, 2)
high_dance = df[df['danceability'] > 0.8]
scatter = ax1b.scatter(high_dance['energy'], high_dance['valence'], 
                      c=high_dance['tempo'], cmap='viridis', 
                      s=high_dance['popularity']*2, alpha=0.6)
ax1b.set_xlabel('Energy', fontweight='bold')
ax1b.set_ylabel('Valence (Happiness)', fontweight='bold')
ax1b.set_title('The Dance Sweet Spot (Danceability > 0.8)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax1b, label='Tempo (BPM)')

# Subplot 1c: Perfect dance tracks
ax1c = plt.subplot(1, 3, 3)
perfect_dance = df[(df['danceability'] > 0.8) & (df['energy'] > 0.7) & 
                   (df['valence'] > 0.7) & (df['tempo'].between(120, 130))]
top_dance_genres = perfect_dance['track_genre'].value_counts().head(10)
bars = ax1c.bar(range(len(top_dance_genres)), top_dance_genres.values, 
                color=gradient_colors[:len(top_dance_genres)])
ax1c.set_xticks(range(len(top_dance_genres)))
ax1c.set_xticklabels(top_dance_genres.index, rotation=45, ha='right')
ax1c.set_ylabel('Count', fontweight='bold')
ax1c.set_title(f'Genres with Perfect Dance Formula\n({len(perfect_dance)} tracks)', 
               fontsize=14, fontweight='bold')

# Adjust layout and save Story 1
plt.tight_layout()
plt.savefig('output/story1_dance_science.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Story 2: The Happiness Index of Music Genres
print("Story 2: The Happiness Index of Music Genres")
# Create figure for Story 2
fig2 = plt.figure(figsize=(18, 6))
fig2.suptitle('The Happiness Index of Music Genres', fontsize=20, fontweight='bold', y=1.05)

# Subplot 2a: Happiness by genre
ax2a = plt.subplot(1, 3, 1)
genre_happiness = df.groupby('track_genre')['valence'].agg(['mean', 'std', 'count'])
genre_happiness = genre_happiness[genre_happiness['count'] >= 100].sort_values('mean', ascending=False)
top_15_happy = genre_happiness.head(15)

positions = range(len(top_15_happy))
bars = ax2a.barh(positions, top_15_happy['mean'], 
                 xerr=top_15_happy['std']/np.sqrt(top_15_happy['count']),
                 color=plt.cm.RdYlGn(top_15_happy['mean']), alpha=0.8)
ax2a.set_yticks(positions)
ax2a.set_yticklabels(top_15_happy.index)
ax2a.set_xlabel('Average Valence (Happiness Score)', fontweight='bold')
ax2a.set_title('The Happiness Index: Top 15 Genres', fontsize=14, fontweight='bold')
ax2a.set_xlim(0, 1)

# Subplot 2b: Happiness distribution
ax2b = plt.subplot(1, 3, 2)
top_5_genres = df['track_genre'].value_counts().head(5).index
for i, genre in enumerate(top_5_genres):
    genre_data = df[df['track_genre'] == genre]['valence']
    ax2b.hist(genre_data, bins=30, alpha=0.6, label=genre, color=colors[i], density=True)
ax2b.set_xlabel('Valence (Happiness)', fontweight='bold')
ax2b.set_ylabel('Density', fontweight='bold')
ax2b.set_title('Happiness Distribution: Top 5 Genres', fontsize=14, fontweight='bold')
ax2b.legend()

# Subplot 2c: Emotional map
ax2c = plt.subplot(1, 3, 3)
sample_emotion = df.sample(5000, random_state=42)
scatter = ax2c.scatter(sample_emotion['valence'], sample_emotion['energy'], 
                      c=sample_emotion['danceability'], cmap='plasma', alpha=0.5, s=20)
ax2c.set_xlabel('Valence (Happiness)', fontweight='bold')
ax2c.set_ylabel('Energy', fontweight='bold')
ax2c.set_title('Emotional Map of Music', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax2c, label='Danceability')
# Add quadrant labels
ax2c.text(0.25, 0.75, 'Turbulent\n(Sad & Energetic)', ha='center', va='center', 
          fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax2c.text(0.75, 0.75, 'Joyful\n(Happy & Energetic)', ha='center', va='center', 
          fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax2c.text(0.25, 0.25, 'Melancholic\n(Sad & Calm)', ha='center', va='center', 
          fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax2c.text(0.75, 0.25, 'Peaceful\n(Happy & Calm)', ha='center', va='center', 
          fontsize=10, bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

# Adjust layout and save Story 2
plt.tight_layout()
plt.savefig('output/story2_happiness_index.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Story 3: The Loudness Wars
print("Story 3: The Loudness Wars")
# Create figure for Story 3
fig3 = plt.figure(figsize=(18, 6))
fig3.suptitle('The Loudness Wars', fontsize=20, fontweight='bold', y=1.05)

# Subplot 3a: Loudness vs Popularity
ax3a = plt.subplot(1, 3, 1)
loudness_bins = pd.qcut(df['loudness'], q=10)
loudness_popularity = df.groupby(loudness_bins)['popularity'].mean()
ax3a.plot(range(len(loudness_popularity)), loudness_popularity.values, 
          'o-', linewidth=2, markersize=8, color='#FF6B6B')
ax3a.set_xlabel('Loudness Decile (Quiet → Loud)', fontweight='bold')
ax3a.set_ylabel('Average Popularity', fontweight='bold')
ax3a.set_title('The Loudness Wars: Louder = More Popular?', fontsize=14, fontweight='bold')
ax3a.grid(True, alpha=0.3)

# Subplot 3b: Loudness by genre
ax3b = plt.subplot(1, 3, 2)
genre_loudness = df.groupby('track_genre')['loudness'].mean().sort_values(ascending=False).head(15)
bars = ax3b.bar(range(len(genre_loudness)), genre_loudness.values, 
                color=plt.cm.Reds(np.linspace(0.4, 0.9, len(genre_loudness))))
ax3b.set_xticks(range(len(genre_loudness)))
ax3b.set_xticklabels(genre_loudness.index, rotation=45, ha='right')
ax3b.set_ylabel('Average Loudness (dB)', fontweight='bold')
ax3b.set_title('Loudest Genres: The Arms Race', fontsize=14, fontweight='bold')
ax3b.axhline(y=df['loudness'].mean(), color='black', linestyle='--', alpha=0.5, label='Overall Average')
ax3b.legend()

# Subplot 3c: Loudness evolution scatter
ax3c = plt.subplot(1, 3, 3)
sample_loud = df.sample(10000, random_state=42)
scatter = ax3c.scatter(sample_loud['loudness'], sample_loud['energy'], 
                      c=sample_loud['popularity'], cmap='hot', alpha=0.5, s=10)
ax3c.set_xlabel('Loudness (dB)', fontweight='bold')
ax3c.set_ylabel('Energy', fontweight='bold')
ax3c.set_title('The Loudness-Energy-Popularity Triangle', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax3c, label='Popularity')

# Adjust layout and save Story 3
plt.tight_layout()
plt.savefig('output/story3_loudness_wars.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Story 4: The Beatles vs. Everyone
print("Story 4: The Beatles vs. Everyone")
# Create figure for Story 4
fig4 = plt.figure(figsize=(18, 6))
fig4.suptitle('The Beatles vs. Everyone', fontsize=20, fontweight='bold', y=1.05)

# First, let's identify if The Beatles are in the dataset
beatles_tracks = df[df['artists'].str.contains('Beatles', case=False, na=False)]

ax4a = plt.subplot(1, 3, 1)
if len(beatles_tracks) > 0:
    # Compare Beatles to average
    features_compare = ['danceability', 'energy', 'speechiness', 'acousticness', 
                       'instrumentalness', 'liveness', 'valence']
    beatles_avg = beatles_tracks[features_compare].mean()
    overall_avg = df[features_compare].mean()
    
    x = np.arange(len(features_compare))
    width = 0.35
    
    bars1 = ax4a.bar(x - width/2, beatles_avg, width, label='The Beatles', color='#4ECDC4', alpha=0.8)
    bars2 = ax4a.bar(x + width/2, overall_avg, width, label='All Artists', color='#FF6B6B', alpha=0.8)
    
    ax4a.set_xticks(x)
    ax4a.set_xticklabels(features_compare, rotation=45, ha='right')
    ax4a.set_ylabel('Average Value', fontweight='bold')
    ax4a.set_title(f'The Beatles Sound Signature ({len(beatles_tracks)} tracks)', fontsize=14, fontweight='bold')
    ax4a.legend()
else:
    # Alternative: Top artist analysis
    top_artist = df['artists'].value_counts().index[0]
    top_artist_tracks = df[df['artists'] == top_artist]
    
    features_compare = ['danceability', 'energy', 'speechiness', 'acousticness', 
                       'instrumentalness', 'liveness', 'valence']
    artist_avg = top_artist_tracks[features_compare].mean()
    overall_avg = df[features_compare].mean()
    
    x = np.arange(len(features_compare))
    width = 0.35
    
    bars1 = ax4a.bar(x - width/2, artist_avg, width, label=f'{top_artist[:20]}...', color='#4ECDC4', alpha=0.8)
    bars2 = ax4a.bar(x + width/2, overall_avg, width, label='All Artists', color='#FF6B6B', alpha=0.8)
    
    ax4a.set_xticks(x)
    ax4a.set_xticklabels(features_compare, rotation=45, ha='right')
    ax4a.set_ylabel('Average Value', fontweight='bold')
    ax4a.set_title(f'Top Artist Sound Signature ({len(top_artist_tracks)} tracks)', fontsize=14, fontweight='bold')
    ax4a.legend()

# Subplot 4b: Radar chart comparison
ax4b = plt.subplot(1, 3, 2, projection='polar')
if len(beatles_tracks) > 0:
    artist_data = beatles_tracks[features_compare].mean().values
    artist_name = "The Beatles"
else:
    artist_data = top_artist_tracks[features_compare].mean().values
    artist_name = top_artist[:20] + "..."

angles = np.linspace(0, 2 * np.pi, len(features_compare), endpoint=False).tolist()
angles += angles[:1]

artist_values = artist_data.tolist()
artist_values += artist_values[:1]
overall_values = overall_avg.values.tolist()
overall_values += overall_values[:1]

ax4b.plot(angles, artist_values, 'o-', linewidth=2, label=artist_name, color='#4ECDC4')
ax4b.fill(angles, artist_values, alpha=0.25, color='#4ECDC4')
ax4b.plot(angles, overall_values, 'o-', linewidth=2, label='Dataset Average', color='#FF6B6B')
ax4b.fill(angles, overall_values, alpha=0.25, color='#FF6B6B')

ax4b.set_theta_offset(np.pi / 2)
ax4b.set_theta_direction(-1)
ax4b.set_xticks(angles[:-1])
ax4b.set_xticklabels(features_compare)
ax4b.set_ylim(0, 1)
ax4b.set_title('Audio DNA Comparison', fontsize=14, fontweight='bold', pad=20)
ax4b.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Subplot 4c: Uniqueness score
ax4c = plt.subplot(1, 3, 3)
# Calculate uniqueness scores for top artists
top_20_artists = df['artists'].value_counts().head(20).index
uniqueness_scores = []

for artist in top_20_artists:
    artist_tracks = df[df['artists'] == artist]
    if len(artist_tracks) >= 5:  # Only consider artists with 5+ tracks
        artist_features = artist_tracks[features_compare].mean()
        overall_features = df[features_compare].mean()
        uniqueness = np.sqrt(np.sum((artist_features - overall_features)**2))
        uniqueness_scores.append((artist[:25], uniqueness))

uniqueness_df = pd.DataFrame(uniqueness_scores, columns=['Artist', 'Uniqueness'])
uniqueness_df = uniqueness_df.sort_values('Uniqueness', ascending=False).head(15)

bars = ax4c.barh(range(len(uniqueness_df)), uniqueness_df['Uniqueness'], 
                 color=gradient_colors[:len(uniqueness_df)])
ax4c.set_yticks(range(len(uniqueness_df)))
ax4c.set_yticklabels(uniqueness_df['Artist'])
ax4c.set_xlabel('Uniqueness Score', fontweight='bold')
ax4c.set_title('Most Unique Artists (vs. Dataset Average)', fontsize=14, fontweight='bold')

# Adjust layout and save Story 4
plt.tight_layout()
plt.savefig('output/story4_artist_analysis.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Story 5: The 3-Minute Rule
print("Story 5: The 3-Minute Rule")
# Create figure for Story 5
fig5 = plt.figure(figsize=(18, 6))
fig5.suptitle('The 3-Minute Rule', fontsize=20, fontweight='bold', y=1.05)

# Subplot 5a: Duration vs Popularity
ax5a = plt.subplot(1, 3, 1)
df['duration_min'] = df['duration_ms'] / 60000
duration_bins = pd.cut(df['duration_min'], bins=[0, 2, 2.5, 3, 3.5, 4, 5, 100])
duration_popularity = df.groupby(duration_bins)['popularity'].agg(['mean', 'count'])
duration_popularity = duration_popularity[duration_popularity['count'] >= 100]

bars = ax5a.bar(range(len(duration_popularity)), duration_popularity['mean'], 
                color='#96CEB4', alpha=0.8)
ax5a.set_xticks(range(len(duration_popularity)))
ax5a.set_xticklabels(['<2min', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-5', '>5min'])
ax5a.set_ylabel('Average Popularity', fontweight='bold')
ax5a.set_title('The 3-Minute Rule: Optimal Song Length', fontsize=14, fontweight='bold')

# Add count labels
for i, (bar, count) in enumerate(zip(bars, duration_popularity['count'])):
    ax5a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'n={count}', ha='center', va='bottom', fontsize=9)

# Subplot 5b: Duration distribution
ax5b = plt.subplot(1, 3, 2)
ax5b.hist(df[df['duration_min'] < 10]['duration_min'], bins=50, 
          color='#45B7D1', alpha=0.7, edgecolor='black')
ax5b.axvline(x=3, color='red', linestyle='--', linewidth=2, label='3-minute mark')
ax5b.axvline(x=df['duration_min'].median(), color='green', linestyle='--', 
             linewidth=2, label=f'Median: {df["duration_min"].median():.2f} min')
ax5b.set_xlabel('Duration (minutes)', fontweight='bold')
ax5b.set_ylabel('Count', fontweight='bold')
ax5b.set_title('Song Duration Distribution', fontsize=14, fontweight='bold')
ax5b.legend()

# Subplot 5c: Genre duration preferences
ax5c = plt.subplot(1, 3, 3)
genre_duration = df.groupby('track_genre')['duration_min'].mean().sort_values(ascending=False).head(10)
bars = ax5c.barh(range(len(genre_duration)), genre_duration.values, 
                 color=gradient_colors[:len(genre_duration)])
ax5c.set_yticks(range(len(genre_duration)))
ax5c.set_yticklabels(genre_duration.index)
ax5c.set_xlabel('Average Duration (minutes)', fontweight='bold')
ax5c.set_title('Longest Songs by Genre', fontsize=14, fontweight='bold')
ax5c.axvline(x=3, color='red', linestyle='--', alpha=0.5, label='3-minute mark')
ax5c.legend()

# Adjust layout and save Story 5
plt.tight_layout()
plt.savefig('output/story5_three_minute_rule.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Story 6: Explicit Content - Does Controversy Pay?
print("Story 6: Explicit Content - Does Controversy Pay?")
# Create figure for Story 6
fig6 = plt.figure(figsize=(18, 6))
fig6.suptitle('Explicit Content: Does Controversy Pay?', fontsize=20, fontweight='bold', y=1.05)

# Subplot 6a: Explicit vs Clean popularity
ax6a = plt.subplot(1, 3, 1)
explicit_data = df.groupby('explicit')['popularity'].agg(['mean', 'std', 'count'])
x = ['Clean', 'Explicit']
means = [explicit_data.loc[False, 'mean'], explicit_data.loc[True, 'mean']]
stds = [explicit_data.loc[False, 'std'], explicit_data.loc[True, 'std']]
counts = [explicit_data.loc[False, 'count'], explicit_data.loc[True, 'count']]

bars = ax6a.bar(x, means, yerr=stds/np.sqrt(counts), color=['#4ECDC4', '#FF6B6B'], 
                alpha=0.8, capsize=10)
ax6a.set_ylabel('Average Popularity', fontweight='bold')
ax6a.set_title('Explicit vs Clean: Popularity Comparison', fontsize=14, fontweight='bold')

# Add statistical test
t_stat, p_value = stats.ttest_ind(df[df['explicit'] == True]['popularity'].dropna(),
                                  df[df['explicit'] == False]['popularity'].dropna())
ax6a.text(0.5, max(means) + 2, f'p-value: {p_value:.4f}\n{"Significant!" if p_value < 0.05 else "Not Significant"}',
         ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Add count labels
for bar, count in zip(bars, counts):
    ax6a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'n={count:,}', ha='center', va='bottom', fontsize=10)

# Subplot 6b: Explicit content by genre
ax6b = plt.subplot(1, 3, 2)
genre_explicit = df.groupby('track_genre')['explicit'].mean().sort_values(ascending=False).head(15)
genre_explicit_pct = genre_explicit * 100

bars = ax6b.barh(range(len(genre_explicit_pct)), genre_explicit_pct.values, 
                 color=plt.cm.Reds(genre_explicit_pct.values/genre_explicit_pct.max()))
ax6b.set_yticks(range(len(genre_explicit_pct)))
ax6b.set_yticklabels(genre_explicit_pct.index)
ax6b.set_xlabel('Percentage of Explicit Tracks (%)', fontweight='bold')
ax6b.set_title('Most Explicit Genres', fontsize=14, fontweight='bold')

# Subplot 6c: Audio features comparison
ax6c = plt.subplot(1, 3, 3)
explicit_features = df[df['explicit'] == True][['energy', 'danceability', 'valence', 'speechiness']].mean()
clean_features = df[df['explicit'] == False][['energy', 'danceability', 'valence', 'speechiness']].mean()

x = np.arange(len(explicit_features))
width = 0.35

bars1 = ax6c.bar(x - width/2, explicit_features, width, label='Explicit', color='#FF6B6B', alpha=0.8)
bars2 = ax6c.bar(x + width/2, clean_features, width, label='Clean', color='#4ECDC4', alpha=0.8)

ax6c.set_xticks(x)
ax6c.set_xticklabels(['Energy', 'Danceability', 'Valence', 'Speechiness'])
ax6c.set_ylabel('Average Value', fontweight='bold')
ax6c.set_title('Audio Profile: Explicit vs Clean', fontsize=14, fontweight='bold')
ax6c.legend()

# Adjust layout and save Story 6
plt.tight_layout()
plt.savefig('output/story6_explicit_content.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Story 7: The Instrumental Underground
print("Story 7: The Instrumental Underground")
# Create figure for Story 7
fig7 = plt.figure(figsize=(18, 6))
fig7.suptitle('The Instrumental Underground', fontsize=20, fontweight='bold', y=1.05)

# Subplot 7a: Instrumentalness distribution
ax7a = plt.subplot(1, 3, 1)
instrumental_threshold = 0.5
df['is_instrumental'] = df['instrumentalness'] > instrumental_threshold

ax7a.hist(df['instrumentalness'], bins=50, color='#98D8C8', alpha=0.7, edgecolor='black')
ax7a.axvline(x=instrumental_threshold, color='red', linestyle='--', linewidth=2, 
             label=f'Instrumental threshold ({instrumental_threshold})')
ax7a.set_xlabel('Instrumentalness Score', fontweight='bold')
ax7a.set_ylabel('Count', fontweight='bold')
ax7a.set_title('The Instrumental Spectrum', fontsize=14, fontweight='bold')
ax7a.legend()
ax7a.set_yscale('log')  # Log scale to see the distribution better

# Subplot 7b: Instrumental genres
ax7b = plt.subplot(1, 3, 2)
genre_instrumental = df.groupby('track_genre')['is_instrumental'].mean().sort_values(ascending=False).head(15)
genre_instrumental_pct = genre_instrumental * 100

bars = ax7b.bar(range(len(genre_instrumental_pct)), genre_instrumental_pct.values, 
                color=gradient_colors[:len(genre_instrumental_pct)])
ax7b.set_xticks(range(len(genre_instrumental_pct)))
ax7b.set_xticklabels(genre_instrumental_pct.index, rotation=45, ha='right')
ax7b.set_ylabel('% Instrumental Tracks', fontweight='bold')
ax7b.set_title('Most Instrumental Genres', fontsize=14, fontweight='bold')

# Subplot 7c: Instrumental track characteristics
ax7c = plt.subplot(1, 3, 3)
instrumental_tracks = df[df['is_instrumental']]
vocal_tracks = df[~df['is_instrumental']]

features_inst = ['energy', 'valence', 'danceability', 'acousticness']
inst_avg = instrumental_tracks[features_inst].mean()
vocal_avg = vocal_tracks[features_inst].mean()

x = np.arange(len(features_inst))
width = 0.35

bars1 = ax7c.bar(x - width/2, inst_avg, width, label='Instrumental', color='#98D8C8', alpha=0.8)
bars2 = ax7c.bar(x + width/2, vocal_avg, width, label='Vocal', color='#BB8FCE', alpha=0.8)

ax7c.set_xticks(x)
ax7c.set_xticklabels(features_inst)
ax7c.set_ylabel('Average Value', fontweight='bold')
ax7c.set_title('Instrumental vs Vocal Track Profiles', fontsize=14, fontweight='bold')
ax7c.legend()

# Adjust layout and save Story 7
plt.tight_layout()
plt.savefig('output/story7_instrumental_underground.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Story 8: Major vs Minor - The Emotional DNA
print("Story 8: Major vs Minor - The Emotional DNA")
# Create figure for Story 8
fig8 = plt.figure(figsize=(18, 6))
fig8.suptitle('Major vs Minor: The Emotional DNA', fontsize=20, fontweight='bold', y=1.05)

# Subplot 8a: Mode distribution
ax8a = plt.subplot(1, 3, 1)
mode_counts = df['mode'].value_counts()
colors_mode = ['#4ECDC4', '#FF6B6B']
wedges, texts, autotexts = ax8a.pie(mode_counts.values, 
                                    labels=['Minor', 'Major'],
                                    autopct='%1.1f%%',
                                    colors=colors_mode,
                                    explode=(0.05, 0))
ax8a.set_title('Major vs Minor Distribution', fontsize=14, fontweight='bold')

# Subplot 8b: Emotional comparison
ax8b = plt.subplot(1, 3, 2)
major_tracks = df[df['mode'] == 1]
minor_tracks = df[df['mode'] == 0]

emotional_features = ['valence', 'energy', 'danceability', 'acousticness']
major_emotion = major_tracks[emotional_features].mean()
minor_emotion = minor_tracks[emotional_features].mean()

x = np.arange(len(emotional_features))
width = 0.35

bars1 = ax8b.bar(x - width/2, major_emotion, width, label='Major', color='#4ECDC4', alpha=0.8)
bars2 = ax8b.bar(x + width/2, minor_emotion, width, label='Minor', color='#FF6B6B', alpha=0.8)

ax8b.set_xticks(x)
ax8b.set_xticklabels(emotional_features)
ax8b.set_ylabel('Average Value', fontweight='bold')
ax8b.set_title('Emotional DNA: Major vs Minor', fontsize=14, fontweight='bold')
ax8b.legend()

# Add significance tests
for i, feature in enumerate(emotional_features):
    t_stat, p_val = stats.ttest_ind(major_tracks[feature].dropna(), 
                                    minor_tracks[feature].dropna())
    if p_val < 0.05:
        ax8b.text(i, max(major_emotion[i], minor_emotion[i]) + 0.02, 
                 '*', ha='center', fontsize=16, fontweight='bold')

# Subplot 8c: Genre mode preferences
ax8c = plt.subplot(1, 3, 3)
genre_mode = df.groupby('track_genre')['mode'].mean().sort_values(ascending=False).head(15)
genre_major_pct = genre_mode * 100

bars = ax8c.barh(range(len(genre_major_pct)), genre_major_pct.values, 
                 color=plt.cm.RdYlBu(genre_major_pct.values/100))
ax8c.set_yticks(range(len(genre_major_pct)))
ax8c.set_yticklabels(genre_major_pct.index)
ax8c.set_xlabel('% Major Key Tracks', fontweight='bold')
ax8c.set_title('Genre Preferences for Major Keys', fontsize=14, fontweight='bold')
ax8c.axvline(x=50, color='black', linestyle='--', alpha=0.5)

# Adjust layout and save Story 8
plt.tight_layout()
plt.savefig('output/story8_major_minor_emotional_dna.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Story 9: The Algorithm's Favorites
print("Story 9: The Algorithm's Favorites - Decoding Popularity")
# Create figure for Story 9
fig9 = plt.figure(figsize=(18, 6))
fig9.suptitle("The Algorithm's Favorites: Decoding Popularity", fontsize=20, fontweight='bold', y=1.05)

# Subplot 9a: Feature importance
ax9a = plt.subplot(1, 3, 1)

# Prepare data for ML
ml_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
               'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 
               'time_signature', 'explicit', 'mode']
df_ml = df.dropna(subset=ml_features + ['popularity'])
df_ml['explicit'] = df_ml['explicit'].astype(int)

# Sample for performance
sample_ml = df_ml.sample(min(10000, len(df_ml)), random_state=42)
X = sample_ml[ml_features]
y = sample_ml['popularity']

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Plot feature importance
importance_df = pd.DataFrame({
    'feature': ml_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True)

bars = ax9a.barh(range(len(importance_df)), importance_df['importance'],
                 color='#DDA0DD', alpha=0.8)
ax9a.set_yticks(range(len(importance_df)))
ax9a.set_yticklabels(importance_df['feature'])
ax9a.set_xlabel('Feature Importance', fontweight='bold')
ax9a.set_title('The Algorithm\'s Favorites: What Drives Popularity?', fontsize=14, fontweight='bold')

# Subplot 9b: Popularity prediction accuracy
ax9b = plt.subplot(1, 3, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_full = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_full.fit(X_train, y_train)
y_pred = rf_full.predict(X_test)

ax9b.scatter(y_test, y_pred, alpha=0.5, s=20, color='#45B7D1')
ax9b.plot([0, 100], [0, 100], 'r--', linewidth=2)
ax9b.set_xlabel('Actual Popularity', fontweight='bold')
ax9b.set_ylabel('Predicted Popularity', fontweight='bold')
ax9b.set_title(f'Prediction Accuracy (R² = {rf_full.score(X_test, y_test):.3f})', 
               fontsize=14, fontweight='bold')

# Subplot 9c: Success recipe
ax9c = plt.subplot(1, 3, 3)
# Find optimal values for top features
top_features = importance_df.nlargest(6, 'importance')['feature'].values
optimal_values = {}

for feature in top_features:
    # Find average value for highly popular tracks
    popular_tracks = sample_ml[sample_ml['popularity'] > 70]
    optimal_values[feature] = popular_tracks[feature].mean()

y_pos = range(len(optimal_values))
values = list(optimal_values.values())
features = list(optimal_values.keys())

bars = ax9c.barh(y_pos, values, color=gradient_colors[:len(values)])
ax9c.set_yticks(y_pos)
ax9c.set_yticklabels(features)
ax9c.set_xlabel('Optimal Value', fontweight='bold')
ax9c.set_title('The Success Recipe: Optimal Feature Values', fontsize=14, fontweight='bold')

# Add value labels
for i, (bar, value) in enumerate(zip(bars, values)):
    ax9c.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{value:.3f}', va='center', fontsize=9)

# Adjust layout and save Story 9
plt.tight_layout()
plt.savefig('output/story9_algorithm_favorites.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Story 10: Live vs Studio - The Authenticity Spectrum
print("Story 10: Live vs Studio - The Authenticity Spectrum")
# Create figure for Story 10
fig10 = plt.figure(figsize=(18, 6))
fig10.suptitle('Live vs Studio: The Authenticity Spectrum', fontsize=20, fontweight='bold', y=1.05)

# Subplot 10a: Liveness distribution
ax10a = plt.subplot(1, 3, 1)
ax10a.hist(df['liveness'], bins=50, color='#F7DC6F', alpha=0.7, edgecolor='black')
ax10a.axvline(x=df['liveness'].mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {df["liveness"].mean():.3f}')
ax10a.set_xlabel('Liveness Score', fontweight='bold')
ax10a.set_ylabel('Count', fontweight='bold')
ax10a.set_title('The Authenticity Spectrum: Live vs Studio', fontsize=14, fontweight='bold')
ax10a.legend()

# Subplot 10b: Liveness by genre
ax10b = plt.subplot(1, 3, 2)
genre_liveness = df.groupby('track_genre')['liveness'].mean().sort_values(ascending=False).head(15)

bars = ax10b.bar(range(len(genre_liveness)), genre_liveness.values,
                 color=plt.cm.YlOrRd(genre_liveness.values))
ax10b.set_xticks(range(len(genre_liveness)))
ax10b.set_xticklabels(genre_liveness.index, rotation=45, ha='right')
ax10b.set_ylabel('Average Liveness', fontweight='bold')
ax10b.set_title('Most "Live" Genres', fontsize=14, fontweight='bold')

# Subplot 10c: Liveness vs Popularity
ax10c = plt.subplot(1, 3, 3)
liveness_bins = pd.qcut(df['liveness'], q=10, duplicates='drop')
liveness_popularity = df.groupby(liveness_bins)['popularity'].mean()

ax10c.plot(range(len(liveness_popularity)), liveness_popularity.values,
          'o-', linewidth=2, markersize=8, color='#F8B500')
ax10c.set_xlabel('Liveness Decile (Studio → Live)', fontweight='bold')
ax10c.set_ylabel('Average Popularity', fontweight='bold')
ax10c.set_title('Does Authenticity Pay? Liveness vs Popularity', fontsize=14, fontweight='bold')
ax10c.grid(True, alpha=0.3)

# Adjust layout and save Story 10
plt.tight_layout()
plt.savefig('output/story10_live_vs_studio.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Story 11: The Acoustic Revival
print("Story 11: The Acoustic Revival")
# Create figure for Story 11
fig11 = plt.figure(figsize=(18, 6))
fig11.suptitle('The Acoustic Revival', fontsize=20, fontweight='bold', y=1.05)

# Subplot 11a: Acousticness distribution
ax11a = plt.subplot(1, 3, 1)
ax11a.hist(df['acousticness'], bins=50, color='#96CEB4', alpha=0.7, edgecolor='black')
acoustic_threshold = 0.5
ax11a.axvline(x=acoustic_threshold, color='red', linestyle='--', linewidth=2,
              label=f'Acoustic threshold ({acoustic_threshold})')
ax11a.set_xlabel('Acousticness Score', fontweight='bold')
ax11a.set_ylabel('Count', fontweight='bold')
ax11a.set_title('The Acoustic Spectrum', fontsize=14, fontweight='bold')
ax11a.legend()

# Subplot 11b: Acoustic genres
ax11b = plt.subplot(1, 3, 2)
df['is_acoustic'] = df['acousticness'] > acoustic_threshold
genre_acoustic = df.groupby('track_genre')['is_acoustic'].mean().sort_values(ascending=False).head(15)
genre_acoustic_pct = genre_acoustic * 100

bars = ax11b.barh(range(len(genre_acoustic_pct)), genre_acoustic_pct.values,
                  color=plt.cm.Greens(genre_acoustic_pct.values/100))
ax11b.set_yticks(range(len(genre_acoustic_pct)))
ax11b.set_yticklabels(genre_acoustic_pct.index)
ax11b.set_xlabel('% Acoustic Tracks', fontweight='bold')
ax11b.set_title('Most Acoustic Genres', fontsize=14, fontweight='bold')

# Subplot 11c: Acoustic vs Electronic comparison
ax11c = plt.subplot(1, 3, 3)
acoustic_tracks = df[df['is_acoustic']]
electronic_tracks = df[~df['is_acoustic']]

comparison_features = ['energy', 'loudness', 'tempo', 'valence', 'popularity']
acoustic_profile = acoustic_tracks[comparison_features].mean()
electronic_profile = electronic_tracks[comparison_features].mean()

# Normalize for comparison
acoustic_norm = acoustic_profile / acoustic_profile.abs().max()
electronic_norm = electronic_profile / electronic_profile.abs().max()

x = np.arange(len(comparison_features))
width = 0.35

bars1 = ax11c.bar(x - width/2, acoustic_norm, width, label='Acoustic', color='#96CEB4', alpha=0.8)
bars2 = ax11c.bar(x + width/2, electronic_norm, width, label='Electronic', color='#BB8FCE', alpha=0.8)

ax11c.set_xticks(x)
ax11c.set_xticklabels(comparison_features, rotation=45, ha='right')
ax11c.set_ylabel('Normalized Value', fontweight='bold')
ax11c.set_title('Acoustic vs Electronic: Profile Comparison', fontsize=14, fontweight='bold')
ax11c.legend()

# Adjust layout and save Story 11
plt.tight_layout()
plt.savefig('output/story11_acoustic_revival.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Story 12: The Speech-Music Hybrid
print("Story 12: The Speech-Music Hybrid")
# Create figure for Story 12
fig12 = plt.figure(figsize=(18, 6))
fig12.suptitle('The Speech-Music Hybrid', fontsize=20, fontweight='bold', y=1.05)

# Subplot 12a: Speechiness spectrum
ax12a = plt.subplot(1, 3, 1)
speechiness_categories = pd.cut(df['speechiness'], 
                               bins=[0, 0.33, 0.66, 1.0],
                               labels=['Music', 'Music+Speech', 'Mostly Speech'])
speech_counts = speechiness_categories.value_counts()

colors_speech = ['#4ECDC4', '#FECA57', '#FF6B6B']
wedges, texts, autotexts = ax12a.pie(speech_counts.values, 
                                     labels=speech_counts.index,
                                     autopct='%1.1f%%',
                                     colors=colors_speech,
                                     explode=(0.05, 0.05, 0.1))
ax12a.set_title('The Speech-Music Spectrum', fontsize=14, fontweight='bold')

# Subplot 12b: Speechiness by genre
ax12b = plt.subplot(1, 3, 2)
genre_speechiness = df.groupby('track_genre')['speechiness'].mean().sort_values(ascending=False).head(15)

bars = ax12b.bar(range(len(genre_speechiness)), genre_speechiness.values,
                 color=gradient_colors[:len(genre_speechiness)])
ax12b.set_xticks(range(len(genre_speechiness)))
ax12b.set_xticklabels(genre_speechiness.index, rotation=45, ha='right')
ax12b.set_ylabel('Average Speechiness', fontweight='bold')
ax12b.set_title('Most Speech-Heavy Genres', fontsize=14, fontweight='bold')

# Subplot 12c: Speech vs Music characteristics
ax12c = plt.subplot(1, 3, 3)
high_speech = df[df['speechiness'] > 0.33]
low_speech = df[df['speechiness'] <= 0.33]

speech_features = ['energy', 'danceability', 'valence', 'acousticness']
high_speech_profile = high_speech[speech_features].mean()
low_speech_profile = low_speech[speech_features].mean()

x = np.arange(len(speech_features))
width = 0.35

bars1 = ax12c.bar(x - width/2, high_speech_profile, width, 
                  label='High Speech', color='#FF6B6B', alpha=0.8)
bars2 = ax12c.bar(x + width/2, low_speech_profile, width, 
                  label='Low Speech', color='#4ECDC4', alpha=0.8)

ax12c.set_xticks(x)
ax12c.set_xticklabels(speech_features)
ax12c.set_ylabel('Average Value', fontweight='bold')
ax12c.set_title('Speech-Heavy vs Pure Music Profiles', fontsize=14, fontweight='bold')
ax12c.legend()

# Adjust layout and save Story 12
plt.tight_layout()
plt.savefig('output/story12_speech_music_hybrid.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Generate comprehensive summary report
print("\n" + "="*80)
print("12 CAPTIVATING SPOTIFY DATA STORIES - COMPLETE ANALYSIS SUMMARY")
print("="*80)

print("\n📊 STORY 1: THE SCIENCE OF WHAT MAKES YOU DANCE")
print("-" * 60)
print(f"Perfect dance formula found in {len(perfect_dance)} tracks:")
print(f"  • Danceability > 0.8")
print(f"  • Energy > 0.7")
print(f"  • Valence > 0.7")
print(f"  • Tempo: 120-130 BPM")
print(f"Top dance genre: {perfect_dance['track_genre'].value_counts().index[0]}")

print("\n😊 STORY 2: THE HAPPINESS INDEX OF MUSIC GENRES")
print("-" * 60)
print(f"Happiest genre: {genre_happiness.index[0]} (valence: {genre_happiness.iloc[0]['mean']:.3f})")
print(f"Saddest genre: {genre_happiness.index[-1]} (valence: {genre_happiness.iloc[-1]['mean']:.3f})")
print(f"Happiness range: {genre_happiness['mean'].min():.3f} to {genre_happiness['mean'].max():.3f}")

print("\n🔊 STORY 3: THE LOUDNESS WARS")
print("-" * 60)
print(f"Loudness-Popularity correlation: {df[['loudness', 'popularity']].corr().iloc[0,1]:.3f}")
print(f"Loudest genre: {genre_loudness.index[0]} ({genre_loudness.iloc[0]:.1f} dB)")
print(f"Quietest genre: {df.groupby('track_genre')['loudness'].mean().sort_values().index[0]}")

print("\n🎸 STORY 4: LEGENDARY PATTERN ANALYSIS")
print("-" * 60)
if len(beatles_tracks) > 0:
    print(f"The Beatles tracks analyzed: {len(beatles_tracks)}")
    print("Unique characteristics vs. average:")
    for feature in features_compare:
        diff = beatles_avg[feature] - overall_avg[feature]
        print(f"  • {feature}: {'+' if diff > 0 else ''}{diff:.3f}")
else:
    print(f"Top artist ({top_artist}) tracks analyzed: {len(top_artist_tracks)}")
    print("Most unique artists identified based on audio DNA distance")

print("\n⏱️ STORY 5: THE 3-MINUTE RULE")
print("-" * 60)
print(f"Average song duration: {df['duration_min'].mean():.2f} minutes")
print(f"Median song duration: {df['duration_min'].median():.2f} minutes")
# Find the label with highest popularity directly
max_popularity_idx = duration_popularity['mean'].values.argmax()
optimal_duration_bin = ax5a.get_xticklabels()[max_popularity_idx].get_text()
print(f"Optimal duration for popularity: {optimal_duration_bin}")
print(f"Songs under 3 minutes: {(df['duration_min'] < 3).sum() / len(df) * 100:.1f}%")

print("\n🔞 STORY 6: EXPLICIT CONTENT - DOES CONTROVERSY PAY?")
print("-" * 60)
print(f"Explicit tracks: {df['explicit'].sum():,} ({df['explicit'].mean()*100:.1f}%)")
print(f"Popularity difference: {means[1] - means[0]:.2f} points")
print(f"Statistical significance: p-value = {p_value:.4f} ({'Significant' if p_value < 0.05 else 'Not significant'})")
print(f"Most explicit genre: {genre_explicit.index[0]} ({genre_explicit.iloc[0]*100:.1f}%)")

print("\n🎹 STORY 7: THE INSTRUMENTAL UNDERGROUND")
print("-" * 60)
print(f"Instrumental tracks (>0.5): {df['is_instrumental'].sum():,} ({df['is_instrumental'].mean()*100:.1f}%)")
print(f"Most instrumental genre: {genre_instrumental.index[0]} ({genre_instrumental.iloc[0]*100:.1f}%)")
print("Instrumental vs Vocal differences:")
for feature in features_inst:
    diff = inst_avg[feature] - vocal_avg[feature]
    print(f"  • {feature}: {'+' if diff > 0 else ''}{diff:.3f}")

print("\n🎼 STORY 8: MAJOR VS MINOR - THE EMOTIONAL DNA")
print("-" * 60)
print(f"Major key tracks: {(df['mode'] == 1).sum():,} ({(df['mode'] == 1).mean()*100:.1f}%)")
print(f"Minor key tracks: {(df['mode'] == 0).sum():,} ({(df['mode'] == 0).mean()*100:.1f}%)")
print("Emotional differences (Major - Minor):")
for feature in emotional_features:
    diff = major_emotion[feature] - minor_emotion[feature]
    t_stat, p_val = stats.ttest_ind(major_tracks[feature].dropna(), 
                                    minor_tracks[feature].dropna())
    sig = "*" if p_val < 0.05 else ""
    print(f"  • {feature}: {'+' if diff > 0 else ''}{diff:.3f} {sig}")

print("\n🤖 STORY 9: THE ALGORITHM'S FAVORITES")
print("-" * 60)
print(f"Model R² score: {rf_full.score(X_test, y_test):.3f}")
print("Top 5 features for popularity:")
for i, (feature, importance) in enumerate(importance_df.nlargest(5, 'importance').values):
    print(f"  {i+1}. {feature}: {importance:.3f}")

print("\n🎤 STORY 10: LIVE VS STUDIO - THE AUTHENTICITY SPECTRUM")
print("-" * 60)
print(f"Average liveness: {df['liveness'].mean():.3f}")
print(f"Most 'live' genre: {genre_liveness.index[0]} ({genre_liveness.iloc[0]:.3f})")
print(f"Liveness-Popularity correlation: {df[['liveness', 'popularity']].corr().iloc[0,1]:.3f}")

print("\n🎸 STORY 11: THE ACOUSTIC REVIVAL")
print("-" * 60)
print(f"Acoustic tracks (>0.5): {df['is_acoustic'].sum():,} ({df['is_acoustic'].mean()*100:.1f}%)")
print(f"Most acoustic genre: {genre_acoustic.index[0]} ({genre_acoustic.iloc[0]*100:.1f}%)")
print("Acoustic vs Electronic energy difference:", 
      f"{acoustic_tracks['energy'].mean() - electronic_tracks['energy'].mean():.3f}")

print("\n🎙️ STORY 12: THE SPEECH-MUSIC HYBRID")
print("-" * 60)
print(f"Pure music (<0.33): {(df['speechiness'] <= 0.33).sum():,} ({(df['speechiness'] <= 0.33).mean()*100:.1f}%)")
print(f"Music+Speech (0.33-0.66): {((df['speechiness'] > 0.33) & (df['speechiness'] <= 0.66)).sum():,}")
print(f"Mostly Speech (>0.66): {(df['speechiness'] > 0.66).sum():,}")
print(f"Most speech-heavy genre: {genre_speechiness.index[0]} ({genre_speechiness.iloc[0]:.3f})")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - 12 STORIES, 36 VISUALIZATIONS, SAVED AS INDIVIDUAL PNG FILES!")
print("See the 'output' folder for the visualization files.")
print("="*80)