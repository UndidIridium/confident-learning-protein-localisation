import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure script runs from project root
os.chdir('/Users/aditya/Desktop/project_JL')

# 1. Plotting: Transitions in v41 Corrections (using standard matplotlib instead of seaborn)
df_corr = pd.read_csv('v41_98_corrections_detailed.csv')
transition_matrix = pd.crosstab(df_corr['Original_Label'], df_corr['Corrected_Label'])

plt.figure(figsize=(10, 8))
plt.imshow(transition_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar(label='Count')
plt.xticks(np.arange(len(transition_matrix.columns)), transition_matrix.columns, rotation=45)
plt.yticks(np.arange(len(transition_matrix.index)), transition_matrix.index)
plt.title('Transitions in Corrected Labels (v41_98)')
plt.savefig('figures/v41_corrections_heatmap.png')
plt.close()

# 2. Bar Chart: Performance of Cleaning Strategies
strategies = {
    'Baseline': 0.71007,
    'Fusion': 0.73504,
    'Label Drop': 0.72386,
    'Mito Enrich': 0.73706,
    'Surgical Rescue': 0.73800,
    'Auto Correct': 0.73900
}

plt.figure(figsize=(10, 6))
plt.bar(strategies.keys(), strategies.values(), color='salmon')
plt.ylim(0.70, 0.75)
plt.ylabel('F1 Score')
plt.title('Performance by Cleaning Strategy')
plt.tight_layout()
plt.savefig('figures/cleaning_strategy_performance.png')
plt.close()

print("Figures saved: figures/v41_corrections_heatmap.png and figures/cleaning_strategy_performance.png")
