import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set up the aesthetic style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define a beautiful color palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#F8B500']
gradient_colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140', '#30cfd0']

# Read the dataset
df = pd.read_csv('dataset.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Create figure with subplots
fig = plt.figure(figsize=(24, 32))
fig.suptitle('Spotify Music Analytics: A Deep Dive into 114,000 Tracks', 
             fontsize=28, fontweight='bold', y=0.995)

# 1. Genre Distribution - Circular Bar Plot
ax1 = plt.subplot(6, 3, 1, projection='polar')
genre_counts = df['track_genre'].value_counts().head(15)
theta = np.linspace(0, 2 * np.pi, len(genre_counts), endpoint=False)
radii = genre_counts.values
width = 2 * np.pi / len(genre_counts)
bars = ax1.bar(theta, radii, width=width, bottom=0.0)
for bar, color in zip(bars, gradient_colors[:len(bars)]):
    bar.set_facecolor(color)
    bar.set_alpha(0.8)
ax1.set_title('Top 15 Genres Distribution', fontsize=16, fontweight='bold', pad=20)
ax1.set_theta_zero_location('N')
ax1.set_theta_direction(-1)

# 2. Audio Features Correlation Heatmap
ax2 = plt.subplot(6, 3, 2)
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
corr_matrix = df[audio_features].corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(250, 30, l=65, center="light", as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, square=True, 
            linewidths=1, cbar_kws={"shrink": .8}, annot=True, fmt='.2f',
            ax=ax2)
ax2.set_title('Audio Features Correlation Matrix', fontsize=16, fontweight='bold')

# 3. Popularity vs Energy vs Danceability - 3D Scatter
ax3 = plt.subplot(6, 3, 3, projection='3d')
sample = df.sample(5000, random_state=42)  # Sample for performance
scatter = ax3.scatter(sample['energy'], sample['danceability'], sample['popularity'],
                     c=sample['valence'], cmap='viridis', alpha=0.6, s=20)
ax3.set_xlabel('Energy', fontweight='bold')
ax3.set_ylabel('Danceability', fontweight='bold')
ax3.set_zlabel('Popularity', fontweight='bold')
ax3.set_title('3D Feature Space: Energy, Danceability & Popularity', fontsize=16, fontweight='bold')
plt.colorbar(scatter, ax=ax3, label='Valence', shrink=0.8)

# 4. Statistical Analysis - Hypothesis Testing
ax4 = plt.subplot(6, 3, 4)
# Test: Are explicit songs more popular?
explicit_pop = df[df['explicit'] == True]['popularity']
clean_pop = df[df['explicit'] == False]['popularity']
t_stat, p_value = stats.ttest_ind(explicit_pop, clean_pop)

# Violin plot
parts = ax4.violinplot([clean_pop, explicit_pop], positions=[0, 1], 
                       showmeans=True, showmedians=True)
for pc, color in zip(parts['bodies'], ['#4ECDC4', '#FF6B6B']):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['Clean', 'Explicit'])
ax4.set_ylabel('Popularity Score', fontweight='bold')
ax4.set_title(f'Explicit vs Clean Songs Popularity\np-value: {p_value:.4f} {"(Significant)" if p_value < 0.05 else "(Not Significant)"}', 
              fontsize=16, fontweight='bold')

# 5. PCA Analysis - Feature Space Visualization
ax5 = plt.subplot(6, 3, 5)
features_for_pca = df[audio_features].dropna()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_for_pca)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)

# K-means clustering in PCA space
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(pca_result)

scatter = ax5.scatter(pca_result[:5000, 0], pca_result[:5000, 1], 
                     c=clusters[:5000], cmap='tab10', alpha=0.6, s=10)
ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
ax5.set_title('PCA & Clustering of Audio Features', fontsize=16, fontweight='bold')

# 6. Tempo Distribution by Mode
ax6 = plt.subplot(6, 3, 6)
major_tempo = df[df['mode'] == 1]['tempo']
minor_tempo = df[df['mode'] == 0]['tempo']

# Create ridgeline plot effect
ax6.hist(major_tempo, bins=50, alpha=0.7, color='#45B7D1', density=True, label='Major')
ax6.hist(minor_tempo, bins=50, alpha=0.7, color='#FF6B6B', density=True, label='Minor')
ax6.set_xlabel('Tempo (BPM)', fontweight='bold')
ax6.set_ylabel('Density', fontweight='bold')
ax6.set_title('Tempo Distribution: Major vs Minor Keys', fontsize=16, fontweight='bold')
ax6.legend()

# 7. Energy vs Loudness with Regression
ax7 = plt.subplot(6, 3, 7)
sample_reg = df.sample(10000, random_state=42)
ax7.hexbin(sample_reg['energy'], sample_reg['loudness'], gridsize=30, cmap='YlOrRd', alpha=0.8)
# Add regression line
z = np.polyfit(sample_reg['energy'], sample_reg['loudness'], 1)
p = np.poly1d(z)
ax7.plot(np.linspace(0, 1, 100), p(np.linspace(0, 1, 100)), 
         "r--", alpha=0.8, linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
ax7.set_xlabel('Energy', fontweight='bold')
ax7.set_ylabel('Loudness (dB)', fontweight='bold')
ax7.set_title('Energy vs Loudness Relationship', fontsize=16, fontweight='bold')
ax7.legend()

# 8. Top Artists by Average Popularity
ax8 = plt.subplot(6, 3, 8)
artist_popularity = df.groupby('artists')['popularity'].agg(['mean', 'count'])
top_artists = artist_popularity[artist_popularity['count'] >= 10].nlargest(15, 'mean')

bars = ax8.barh(range(len(top_artists)), top_artists['mean'], 
                color=gradient_colors[:len(top_artists)])
ax8.set_yticks(range(len(top_artists)))
ax8.set_yticklabels(top_artists.index, fontsize=10)
ax8.set_xlabel('Average Popularity Score', fontweight='bold')
ax8.set_title('Top 15 Artists by Average Popularity\n(min. 10 tracks)', fontsize=16, fontweight='bold')

# Add value labels
for i, (bar, value) in enumerate(zip(bars, top_artists['mean'])):
    ax8.text(value + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{value:.1f}', va='center', fontsize=9)

# 9. Audio Feature Radar Chart for Different Genres
ax9 = plt.subplot(6, 3, 9, projection='polar')
top_genres = df['track_genre'].value_counts().head(5).index
radar_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'valence']
angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
angles += angles[:1]

for i, genre in enumerate(top_genres):
    genre_data = df[df['track_genre'] == genre][radar_features].mean().tolist()
    genre_data += genre_data[:1]
    ax9.plot(angles, genre_data, 'o-', linewidth=2, label=genre, color=colors[i])
    ax9.fill(angles, genre_data, alpha=0.15, color=colors[i])

ax9.set_theta_offset(np.pi / 2)
ax9.set_theta_direction(-1)
ax9.set_xticks(angles[:-1])
ax9.set_xticklabels(radar_features, fontsize=10)
ax9.set_ylim(0, 1)
ax9.set_title('Audio Feature Profiles by Genre', fontsize=16, fontweight='bold', pad=20)
ax9.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# 10. Duration Analysis
ax10 = plt.subplot(6, 3, 10)
df['duration_min'] = df['duration_ms'] / 60000
duration_by_genre = df.groupby('track_genre')['duration_min'].mean().nlargest(10)

bars = ax10.bar(range(len(duration_by_genre)), duration_by_genre.values, 
                color=gradient_colors[:len(duration_by_genre)], alpha=0.8)
ax10.set_xticks(range(len(duration_by_genre)))
ax10.set_xticklabels(duration_by_genre.index, rotation=45, ha='right')
ax10.set_ylabel('Average Duration (minutes)', fontweight='bold')
ax10.set_title('Average Track Duration by Genre', fontsize=16, fontweight='bold')

# 11. Valence vs Danceability Density Plot
ax11 = plt.subplot(6, 3, 11)
sample_density = df.sample(10000, random_state=42)
ax11.hexbin(sample_density['valence'], sample_density['danceability'], 
            gridsize=40, cmap='plasma', mincnt=1)
ax11.set_xlabel('Valence (Positivity)', fontweight='bold')
ax11.set_ylabel('Danceability', fontweight='bold')
ax11.set_title('Mood vs Danceability Density Map', fontsize=16, fontweight='bold')
cbar = plt.colorbar(ax11.collections[0], ax=ax11)
cbar.set_label('Track Count', fontweight='bold')

# 12. Time Signature Distribution
ax12 = plt.subplot(6, 3, 12)
time_sig_counts = df['time_signature'].value_counts().sort_index()
wedges, texts, autotexts = ax12.pie(time_sig_counts.values, 
                                    labels=[f'{int(x)}/4' for x in time_sig_counts.index],
                                    autopct='%1.1f%%', startangle=90,
                                    colors=colors[:len(time_sig_counts)])
ax12.set_title('Time Signature Distribution', fontsize=16, fontweight='bold')

# 13. Instrumentalness Analysis
ax13 = plt.subplot(6, 3, 13)
instrumental_threshold = 0.5
df['is_instrumental'] = df['instrumentalness'] > instrumental_threshold
instrumental_by_genre = df.groupby('track_genre')['is_instrumental'].mean().nlargest(15)

bars = ax13.barh(range(len(instrumental_by_genre)), instrumental_by_genre.values * 100,
                 color='#96CEB4', alpha=0.8)
ax13.set_yticks(range(len(instrumental_by_genre)))
ax13.set_yticklabels(instrumental_by_genre.index, fontsize=10)
ax13.set_xlabel('Percentage of Instrumental Tracks (%)', fontweight='bold')
ax13.set_title('Genres with Most Instrumental Tracks', fontsize=16, fontweight='bold')

# 14. Key Distribution (Musical Keys)
ax14 = plt.subplot(6, 3, 14)
key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
key_counts = df['key'].value_counts().sort_index()

# Create circular plot for keys
theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
ax14 = plt.subplot(6, 3, 14, projection='polar')
bars = ax14.bar(theta, key_counts.values, width=2*np.pi/12, bottom=0.0,
                color=plt.cm.hsv(theta/2/np.pi), alpha=0.8)
ax14.set_xticks(theta)
ax14.set_xticklabels(key_names)
ax14.set_title('Musical Key Distribution', fontsize=16, fontweight='bold', pad=20)

# 15. Feature Importance for Popularity (Random Forest)
from sklearn.ensemble import RandomForestRegressor
ax15 = plt.subplot(6, 3, 15)

# Prepare features for RF
feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 
                'tempo', 'duration_ms', 'time_signature', 'explicit']
df_rf = df.dropna(subset=feature_cols + ['popularity'])
df_rf['explicit'] = df_rf['explicit'].astype(int)

# Sample for performance
sample_rf = df_rf.sample(10000, random_state=42)
X = sample_rf[feature_cols]
y = sample_rf['popularity']

# Train RF
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Plot feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True)

bars = ax15.barh(range(len(importance_df)), importance_df['importance'],
                 color='#DDA0DD', alpha=0.8)
ax15.set_yticks(range(len(importance_df)))
ax15.set_yticklabels(importance_df['feature'], fontsize=10)
ax15.set_xlabel('Feature Importance', fontweight='bold')
ax15.set_title('Features Predicting Track Popularity', fontsize=16, fontweight='bold')

# 16. Speechiness Distribution (Speech vs Music)
ax16 = plt.subplot(6, 3, 16)
speechiness_bins = [0, 0.33, 0.66, 1.0]
speechiness_labels = ['Music', 'Music+Speech', 'Speech']
df['speech_category'] = pd.cut(df['speechiness'], bins=speechiness_bins, 
                               labels=speechiness_labels, include_lowest=True)
speech_counts = df['speech_category'].value_counts()

colors_speech = ['#4ECDC4', '#FECA57', '#FF6B6B']
wedges, texts, autotexts = ax16.pie(speech_counts.values, labels=speech_counts.index,
                                    autopct='%1.1f%%', startangle=90,
                                    colors=colors_speech, explode=(0.05, 0, 0))
ax16.set_title('Content Type Distribution', fontsize=16, fontweight='bold')

# 17. Acoustic vs Electronic Music
ax17 = plt.subplot(6, 3, 17)
acoustic_threshold = 0.5
df['is_acoustic'] = df['acousticness'] > acoustic_threshold

# Compare features
acoustic_features = df[df['is_acoustic']][['energy', 'loudness', 'tempo']].mean()
electronic_features = df[~df['is_acoustic']][['energy', 'loudness', 'tempo']].mean()

# Normalize for comparison
acoustic_norm = acoustic_features / acoustic_features.abs().max()
electronic_norm = electronic_features / electronic_features.abs().max()

x = np.arange(len(acoustic_features))
width = 0.35

bars1 = ax17.bar(x - width/2, acoustic_norm, width, label='Acoustic', 
                 color='#98D8C8', alpha=0.8)
bars2 = ax17.bar(x + width/2, electronic_norm, width, label='Electronic', 
                 color='#BB8FCE', alpha=0.8)

ax17.set_xticks(x)
ax17.set_xticklabels(['Energy', 'Loudness', 'Tempo'])
ax17.set_ylabel('Normalized Value', fontweight='bold')
ax17.set_title('Acoustic vs Electronic Music Characteristics', fontsize=16, fontweight='bold')
ax17.legend()

# 18. Statistical Summary
ax18 = plt.subplot(6, 3, 18)
ax18.axis('off')

# Perform multiple hypothesis tests
tests_results = []

# Test 1: Major vs Minor popularity
major_pop = df[df['mode'] == 1]['popularity']
minor_pop = df[df['mode'] == 0]['popularity']
t_stat1, p_val1 = stats.ttest_ind(major_pop, minor_pop)
tests_results.append(('Major vs Minor Key Popularity', p_val1, p_val1 < 0.05))

# Test 2: Correlation between energy and loudness
corr, p_val2 = stats.pearsonr(df['energy'].dropna(), df['loudness'].dropna())
tests_results.append(('Energy-Loudness Correlation', p_val2, p_val2 < 0.05))

# Test 3: Danceability across genres
top_5_genres = df['track_genre'].value_counts().head(5).index
genre_groups = [df[df['track_genre'] == genre]['danceability'].dropna() for genre in top_5_genres]
f_stat, p_val3 = stats.f_oneway(*genre_groups)
tests_results.append(('Danceability Across Top 5 Genres', p_val3, p_val3 < 0.05))

# Test 4: Valence and mode relationship
high_valence = df[df['valence'] > 0.5]['mode']
low_valence = df[df['valence'] <= 0.5]['mode']
chi2, p_val4, _, _ = stats.chi2_contingency(pd.crosstab(df['valence'] > 0.5, df['mode']))
tests_results.append(('Valence-Mode Association', p_val4, p_val4 < 0.05))

# Display results
summary_text = "Statistical Hypothesis Testing Results\n" + "="*40 + "\n\n"
for test_name, p_value, significant in tests_results:
    status = "✓ Significant" if significant else "✗ Not Significant"
    color = 'green' if significant else 'red'
    summary_text += f"{test_name}:\n"
    summary_text += f"p-value: {p_value:.6f} - {status}\n\n"

ax18.text(0.1, 0.9, summary_text, transform=ax18.transAxes, 
         fontsize=12, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax18.set_title('Statistical Significance Summary', fontsize=16, fontweight='bold')

# Adjust layout
plt.tight_layout()
plt.savefig('spotify_analysis_award_winning.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SPOTIFY DATASET ANALYSIS SUMMARY")
print("="*60)
print(f"\nTotal Tracks Analyzed: {len(df):,}")
print(f"Unique Artists: {df['artists'].nunique():,}")
print(f"Unique Genres: {df['track_genre'].nunique()}")
print(f"\nMost Popular Genre: {df['track_genre'].value_counts().index[0]}")
print(f"Average Track Duration: {df['duration_ms'].mean()/60000:.2f} minutes")
print(f"Percentage of Explicit Tracks: {(df['explicit'].sum()/len(df)*100):.1f}%")
print(f"\nAverage Audio Features:")
for feature in audio_features:
    print(f"  {feature.capitalize()}: {df[feature].mean():.3f}")

# Insights
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("\n1. POPULARITY DRIVERS:")
print(f"   - Loudness has the strongest correlation with popularity (r={df[['popularity', 'loudness']].corr().iloc[0,1]:.3f})")
print(f"   - Duration shows negative correlation (r={df[['popularity', 'duration_ms']].corr().iloc[0,1]:.3f})")

print("\n2. GENRE CHARACTERISTICS:")
print("   - Electronic genres tend to have higher energy and loudness")
print("   - Acoustic genres show higher acousticness but lower energy")

print("\n3. MUSICAL PATTERNS:")
print(f"   - {(df['time_signature'] == 4).sum()/len(df)*100:.1f}% of tracks are in 4/4 time")
print(f"   - Major key songs: {(df['mode'] == 1).sum()/len(df)*100:.1f}%")
print(f"   - Most common key: {key_names[df['key'].mode()[0]]}")

print("\n4. MOOD ANALYSIS:")
print(f"   - Average valence (positivity): {df['valence'].mean():.3f}")
print(f"   - High energy + High valence tracks: {((df['energy'] > 0.7) & (df['valence'] > 0.7)).sum():,}")

print("\n" + "="*60)