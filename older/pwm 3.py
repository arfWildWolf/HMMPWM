import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up professional plotting style
sns.set_theme(style="white")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'sans-serif'
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
fig.suptitle('Biochemical Feature Modeling: PWM (Local Context) vs. CUB (Global Bias)', 
             fontsize=18, fontweight='bold', y=0.98)

# =========================================================================
# LEFT PANEL: Position Weight Matrix (PWM) - Simulated Log-Odds Heatmap
# =========================================================================
# Simulating a small -3 to +4 flanking region around an ATG start codon
positions = ['-3', '-2', '-1', '+1 (A)', '+2 (T)', '+3 (G)', '+4']
nucleotides = ['A', 'C', 'G', 'T']

# Build matrix with explicit biological constraints (Kozak consensus signatures)
# e.g., strong G at +4, strong A at -3, and absolute ATG at +1 to +3
pwm_data = np.array([
    [ 1.2, -0.8, -0.5,  4.0, -4.0, -4.0, -0.2],  # A
    [-0.6,  0.4, -0.9, -4.0, -4.0, -4.0, -0.8],  # C
    [ 0.1, -0.2,  1.1, -4.0, -4.0,  4.0,  1.5],  # G
    [-0.7,  0.6,  0.3, -4.0,  4.0, -4.0, -0.5]   # T
])

sns.heatmap(pwm_data, annot=True, fmt=".1f", cmap="RdYlBu", center=0,
            xticklabels=positions, yticklabels=nucleotides, ax=ax1,
            cbar_kws={'label': 'Log-Odds Score'}, linewidths=1, linecolor='white')

ax1.set_title('1. Position Weight Matrix (PWM)\nTracks Local Sequence Conservation', fontweight='bold', pad=15)
ax1.set_xlabel('Nucleotide Position Relative to Start Codon', fontweight='bold')
ax1.set_ylabel('Base', fontweight='bold')

# Highlight the fixed translation initiation site frame
ax1.add_patch(plt.Rectangle((3, 0), 3, 4, fill=False, edgecolor='black', lw=3, linestyle='--'))
ax1.text(4.5, -0.2, "Fixed TIS Frame", ha='center', va='center', color='black', fontweight='bold', fontsize=11)


# =========================================================================
# RIGHT PANEL: Codon Usage Bias (CUB) - Simulated Synonymous Frequencies
# =========================================================================
# Example focusing on Leucine (6 synonymous codons) and Arginine (4 synonymous codons)
codons = ['CTG(Leu)', 'CTC(Leu)', 'TTG(Leu)', 'TTA(Leu)', 'AGG(Arg)', 'AGA(Arg)', 'CGG(Arg)', 'CGC(Arg)']
frequencies = [0.41, 0.20, 0.13, 0.07, 0.20, 0.21, 0.20, 0.18]

# Assign colors to group synonymous sets visually
colors_cub = ['#007acc', '#2980b9', '#3498db', '#5dade2', '#e74c3c', '#ec7063', '#f1948a', '#f5b7b1']

bars = ax2.bar(codons, frequencies, color=colors_cub, edgecolor='black', width=0.6, linewidth=1.2)

# Label values directly on top of bars
for bar in bars:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', fontsize=10)

ax2.set_title('2. Codon Usage Bias (CUB)\nTracks Background Coding Frequencies', fontweight='bold', pad=15)
ax2.set_xlabel('Synonymous Codons (Exon Signatures)', fontweight='bold')
ax2.set_ylabel('Relative Adaptiveness (w) / Frequency', fontweight='bold')
ax2.set_ylim(0, 0.5)
plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')

# Layout and rendering clean-up
sns.despine(ax=ax2, left=False, bottom=False)
plt.subplots_adjust(top=0.82, wspace=0.3)
plt.tight_layout()
plt.savefig('pwm_cub_models_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print("PWM and CUB diagnostic asset successfully generated.")