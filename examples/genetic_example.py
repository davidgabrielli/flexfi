"""
Genetic data analysis example.

This example demonstrates:
- Gene-based association testing
- Grouping SNPs by gene
- Identifying important genes from genetic variants
"""

import torch
import numpy as np
from flexfi import feature_importance, plot_fi_bar

print("="*60)
print("flexfi Genetic Data Analysis Example")
print("="*60)

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. Simulate Genetic Data
# ============================================================================

n_samples = 800
n_snps = 100  # 100 SNPs across 10 genes (10 SNPs per gene)

print(f"\nSimulating genetic data...")
print(f"  Samples: {n_samples}")
print(f"  SNPs: {n_snps}")
print(f"  Genes: {n_snps // 10}")

# Generate genotype data (0, 1, 2 for SNP genotypes)
X_genetic = torch.tensor(
    np.random.binomial(2, 0.3, size=(n_samples, n_snps)),
    dtype=torch.float32
)

# Create SNP names in format "GENE_rsID"
snp_names = []
for i in range(n_snps):
    gene_id = i // 10  # 10 SNPs per gene
    rs_id = f"rs{i}"
    snp_names.append(f"GENE{gene_id}_{rs_id}")

print(f"\nExample SNP names: {snp_names[:5]}")

# ============================================================================
# 2. Create Disease Outcome
# ============================================================================

# Make GENE0 (SNPs 0-9) causal for disease
# Sum of genotypes in GENE0 determines disease risk
causal_snps = X_genetic[:, 0:10]
genetic_risk_score = causal_snps.sum(dim=1)

# Convert to disease probability
prob_disease = torch.sigmoid(0.3 * (genetic_risk_score - 10))
y_disease = torch.bernoulli(prob_disease).long()

print("\nTrue causal gene: GENE0 (SNPs 0-9)")
print(f"Disease prevalence: {y_disease.float().mean():.1%}")

# ============================================================================
# 3. Create Gene Groups
# ============================================================================

# Group SNPs by gene
gene_groups = {}
for gene_id in range(10):
    start_idx = gene_id * 10
    end_idx = start_idx + 10
    gene_groups[f"GENE{gene_id}"] = list(range(start_idx, end_idx))

print("\n" + "="*60)
print("Gene Groups:")
print("="*60)
for gene, snps in list(gene_groups.items())[:3]:
    print(f"  {gene}: SNPs {snps[0]}-{snps[-1]}")
print("  ...")

# ============================================================================
# 4. Run Gene-based Feature Importance
# ============================================================================

print("\n" + "="*60)
print("Running Gene-based Feature Importance Analysis...")
print("="*60)

results = feature_importance(
    X=X_genetic,
    y=y_disease,
    outcome="binary",
    iterations=5,
    groups=gene_groups,
    kfolds=5,
    perm_count=50,
    epochs=150,  # More epochs for genetic data
    print_level=0
)

# ============================================================================
# 5. Display Results
# ============================================================================

print("\n" + "="*60)
print("Top 5 Most Important Genes:")
print("="*60)
print(results["fi_summary"].head())

print("\n" + "="*60)
print("Interpretation:")
print("="*60)
print("GENE0 should be the most important (most negative score)")
print("because it contains the causal variants.")
print("\nOther genes should have scores near zero,")
print("indicating they don't contribute to disease risk.")

# Identify significant genes (p < 0.05)
significant_genes = results["fi_summary"][
    results["fi_summary"]["P-Value"] < 0.05
]

print(f"\n{len(significant_genes)} genes with p-value < 0.05:")
if len(significant_genes) > 0:
    print(significant_genes["Feature/Group"].tolist())

# ============================================================================
# 6. Visualize
# ============================================================================

print("\nGenerating visualization...")
plot_fi_bar(results["fi_summary"], top_k=10,
           title="Gene-based Association Test Results")

print("\n" + "="*60)
print("Example Complete!")
print("="*60)
print("\nKey Takeaway:")
print("Gene-based testing groups SNPs within genes,")
print("providing more interpretable results than")
print("testing individual SNPs alone.")