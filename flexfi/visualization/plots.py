"""
Visualization functions for feature importance results.

This module provides plotting utilities to visualize feature importance scores
from permutation-based feature importance analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Optional


def plot_fi_bar(
    results_df: pd.DataFrame, 
    top_k: int = 10, 
    title: str = "Feature Importance (Bar Chart)"
) -> None:
    """
    Creates a horizontal bar chart showing feature importance scores with error bars.
    
    Features are sorted by importance score (most negative = most important) and 
    colored using a gradient to emphasize the ranking. Error bars represent the 
    standard deviation of importance scores across iterations.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing feature importance results with columns:
        - 'Feature/Group': Feature names or identifiers
        - 'Importance Score': Mean importance scores across iterations
        - 'Importance Score STD': Standard deviation of importance scores
    top_k : int, default=10
        Number of most important features to display. Features are ranked by
        importance score (most negative scores = most important).
    title : str, default="Feature Importance (Bar Chart)"
        Title for the plot.
        
    Returns:
    --------
    None
        Displays the plot using matplotlib.
        
    Notes:
    ------
    - Negative importance scores indicate higher feature importance
    - Features are automatically sorted by importance score
    - Color gradient helps visualize relative importance ranking
    - Plot size automatically adjusts based on number of features displayed
    
    """
    
    # Check top_k isn't larger than number of features
    n_covariates = results_df["Feature/Group"].nunique()
    top_k = min(top_k, n_covariates)

    # Select and sort top features by importance (most negative first)
    top_features = results_df.sort_values("Importance Score").head(top_k)
    
    # Create color mapping based on importance scores
    scores = top_features["Importance Score"] 
    vmin, vmax = scores.min(), scores.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["coolwarm_r"]  # Reversed colormap: blue=important, red=less important
    colors = [cmap(norm(s)) for s in scores]
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 0.5 * top_k + 2))
    plt.barh(
        top_features["Feature/Group"],
        top_features["Importance Score"],
        xerr=top_features["Importance Score STD"],
        color=colors,
        edgecolor='black',
        capsize=3  # Error bar cap size
    )
    
    # Add reference line at x=0
    plt.axvline(0, color='black', linewidth=0.8, alpha=0.7)
    
    # Customize plot appearance
    plt.xlabel("Importance Score (More Negative = More Important)")
    plt.ylabel("Feature/Group")
    plt.title(title, weight='bold', fontsize=14)
    plt.gca().invert_yaxis()  # Most important feature at top
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_fi_violin(
    results_df: pd.DataFrame, 
    top_k: int = 10, 
    title: str = "Feature Importance (Violin Plot)"
) -> None:
    """
    Creates a violin plot showing the distribution of importance scores across iterations.
    
    This plot is useful for understanding the variability and distribution shape
    of feature importance scores across multiple iterations/seeds. Each violin
    shows the full distribution of scores for that feature.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing per-iteration results with columns:
        - 'Feature/Group': Feature names or identifiers  
        - 'Importance Score': Individual importance scores from each iteration
        Must contain multiple rows per feature (from different iterations).
    top_k : int, default=10
        Number of most important features to display. Features are ranked by
        their mean importance score across iterations.
    title : str, default="Feature Importance (Violin Plot)"
        Title for the plot.
        
    Returns:
    --------
    None
        Displays the plot using matplotlib and seaborn.
        
    Notes:
    ------
    - Requires per-iteration results (not just summary statistics)
    - Violin width represents the distribution density at each score level
    - Box plots inside violins show quartiles and median
    - Colors indicate relative importance ranking
    - Features are sorted by mean importance score
    
    Raises:
    -------
    ValueError
        If results_df doesn't contain valid importance scores or selected features.
        
    """
    
    # Check top_k isn't larger than number of features
    n_covariates = results_df["Feature/Group"].nunique()
    top_k = min(top_k, n_covariates)

    # Clean data and validate
    results_df = results_df.dropna(subset=["Importance Score"])
    
    if len(results_df) == 0:
        raise ValueError("No valid importance scores found in results_df")
    
    # Identify top features based on mean importance score
    top_features = (
        results_df.groupby("Feature/Group")["Importance Score"]
        .mean()
        .sort_values()  # Most negative (important) first
        .head(top_k)
        .index.tolist()
    )
    
    # Filter data to top features only
    subset = results_df[results_df["Feature/Group"].isin(top_features)]
    
    if len(subset) == 0:
        raise ValueError("No data found for selected top features")
    
    # Create color mapping based on mean importance scores
    mean_scores = (
        subset.groupby("Feature/Group")["Importance Score"]
        .mean()
        .reindex(top_features)
    )
    vmin, vmax = mean_scores.min(), mean_scores.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["coolwarm_r"]
    colors = [cmap(norm(s)) for s in mean_scores]
    
    # Create violin plot
    fig, ax = plt.subplots(figsize=(10, 0.5 * top_k + 2))
    
    # Create violins with box plots inside
    sns.violinplot(
        data=subset, 
        x="Importance Score", 
        y="Feature/Group",
        scale="width",  # Same width for all violins
        ax=ax, 
        inner="box",    # Show box plot inside
        linewidth=1,
        order=top_features  # Maintain importance-based ordering
    )
    
    # Apply custom colors to violins
    for i, artist in enumerate(ax.collections[::2]):  # Every 2nd collection is a violin body
        if i < len(colors):
            artist.set_facecolor(colors[i])
            artist.set_edgecolor('black')
            artist.set_alpha(0.8)
    
    # Add reference line and customize appearance
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Importance Score (More Negative = More Important)")
    ax.set_ylabel("Feature/Group")
    ax.set_title(title, weight='bold', fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_score_vs_pval(
    results_df: pd.DataFrame, 
    top_k: int = 20, 
    title: str = "Importance vs. P-Value", 
    annotate: bool = True
) -> None:
    """
    Creates a scatter plot showing the relationship between importance scores and p-values.
    
    This plot helps identify features that are both practically important (large magnitude
    importance scores) and statistically significant (low p-values). Features in the 
    bottom-left quadrant (negative scores, low p-values) are most noteworthy.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing feature importance results with columns:
        - 'Feature/Group': Feature names or identifiers
        - 'Importance Score': Mean importance scores  
        - 'P-Value': Statistical significance values
    top_k : int, default=20
        Number of most important features to annotate with labels.
    title : str, default="Importance vs. P-Value"
        Title for the plot.
    annotate : bool, default=True
        Whether to add feature name labels to the most important points.
        
    Returns:
    --------
    None
        Displays the plot using matplotlib.
        
    Notes:
    ------
    - Features with negative importance scores and low p-values are most significant
    - Red dashed line at p=0.05 indicates conventional significance threshold
    - Text labels are automatically adjusted to avoid overlap when annotate=True
    - Requires the adjustText package for label positioning
    
    """
    from adjustText import adjust_text
    
    # Cap top_n at the number of covariates
    n_covariates = results_df["Feature/Group"].nunique()
    top_k = min(top_k, n_covariates)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        results_df["Importance Score"], 
        results_df["P-Value"],
        color='steelblue', 
        edgecolors='black', 
        alpha=0.7,
        s=60  # Point size
    )
    
    # Add reference lines
    ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(0.05, color='red', linestyle='--', alpha=0.7, linewidth=1, 
               label='p = 0.05 (significance threshold)')
    
    # Add feature labels for most important features
    if annotate:
        # Select top features for labeling
        top_features = results_df.sort_values("Importance Score").head(top_k)
        texts = [
            ax.text(row["Importance Score"], row["P-Value"], row["Feature/Group"],
                    fontsize=9, ha="center", va="center")
            for _, row in top_features.iterrows()
        ]

        # Adjust text positions to avoid overlap
        if texts:
            adjust_text(
                texts, 
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.7),
                ax=ax
            )
    
    # Customize plot appearance
    ax.set_xlabel("Importance Score (More Negative = More Important)", fontsize=12)
    ax.set_ylabel("P-Value (Lower = More Significant)", fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()