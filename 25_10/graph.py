import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('simulation_results.csv')

# IEEE Standard Formatting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'

# Consistent color palette for the 3 strategies across all graphs
strategy_colors = {
    'Naive': '#d62728',     # Red
    'Cautious': '#1f77b4',  # Blue
    'Inertial': '#2ca02c'   # Green
}

# Ensure order is logical (Baseline -> Safe -> Fast)
order = ['Naive', 'Cautious', 'Inertial']

# ---------------------------------------------------------
# 1. Success Rates Bar Chart
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
success_rates = df.groupby('Strategy')['Success'].mean() * 100
success_rates = success_rates.reindex(order)

ax = sns.barplot(x=success_rates.index, y=success_rates.values, hue=success_rates.index, palette=strategy_colors, edgecolor='black', legend=False)
plt.title('Success Rates by Strategy (1,000,000 Runs)', fontweight='bold')
plt.ylabel('Success Rate (%)', fontweight='bold')
plt.xlabel('Strategy', fontweight='bold')
plt.ylim(0, 110)

# Add precise text labels on top of the bars
for i, v in enumerate(success_rates.values):
    ax.text(i, v + 1.5, f"{v:.2f}%", color='black', ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('success_rates.png')
plt.close()

# ---------------------------------------------------------
# 2. Step Count Distribution (Boxplot - ONLY Successful runs)
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

# Filter ONLY successful runs to see true navigation cost
df_success = df[df['Success'] == True]

ax = sns.boxplot(x='Strategy', y='Steps', hue='Strategy', data=df_success, order=order, palette=strategy_colors, fliersize=2, legend=False)
plt.title('Step Count Distribution (Successful Runs Only)', fontweight='bold')
plt.ylabel('Number of Steps to Target', fontweight='bold')
plt.xlabel('Strategy', fontweight='bold')

plt.tight_layout()
plt.savefig('steps_distribution.png')
plt.close()

print("Graphs successfully generated for IEEE publication (300 DPI).")