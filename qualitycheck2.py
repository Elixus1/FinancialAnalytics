# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 18,
    }
)
plt.rcParams["figure.figsize"] = (8, 5)

# %%
df = pd.read_excel("data/Assignment_Dataset_2.xlsx", sheet_name=0)
features = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "ACCR", "LEVI"]


# %%
def ratio(col, str1, str2):
    counts = df[col].value_counts()
    male = counts[str1]
    female = counts[str2]
    print(f"{str1}: {str2} = {male}:{female}")
    print(f"Ratio = {male/female:.2f}")


ratio("Manipulater", "Yes", "No")

# %%
# check for nan or missing values
print(df.isnull().sum())
# dataset is filled with some values =1. This is considered correct. Could also
# be a guessed value.

# %%
for i, feature in enumerate(features):
    plt.subplot(2, 4, i + 1)
    plt.boxplot(df[feature])

    # plt.xlabel(feature)
    plt.title(feature)

plt.tight_layout()
plt.savefig("pictures/boxplots2.pdf", bbox_inches="tight")

# %%
df[features].hist(figsize=(12, 8), bins=100)
plt.tight_layout()
plt.savefig("pictures/histograms2.pdf", bbox_inches="tight")

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 9))
axes = axes.flatten()

# First five KDE plots
for i, feature in enumerate(features):
    sns.kdeplot(
        data=df,
        x=feature,
        hue="Manipulater",
        common_norm=False,
        fill=True,
        alpha=0.3,
        ax=axes[i],
    )

    axes[i].set_xlabel(feature)
    # axes[i].set_yscale("log")
axes[0].set_ylabel("Density")
axes[5].set_ylabel("Density")
# alle anderen Y-Labels entfernen
for ax in axes:
    if ax not in [axes[0], axes[4]]:
        ax.set_ylabel("")

plt.tight_layout()
plt.savefig("pictures/kde_2.pdf", bbox_inches="tight")


# %%
fig, axes = plt.subplots(2, 4, figsize=(12, 8))
for ax, feature in zip(axes.flat, features):
    ax.hist(df[df["Manipulater"] == "Yes"][feature], bins=50, alpha=0.6, label="Yes")
    ax.hist(df[df["Manipulater"] == "No"][feature], bins=50, alpha=0.6, label="No")
    ax.set_title(feature)
    ax.set_ylim(0, 5)

axes[0, 0].legend()
plt.tight_layout()
plt.savefig("pictures/histogram2_zoom.pdf", bbox_inches="tight")
plt.show()

# %%
fig, axes = plt.subplots(4, 2, figsize=(7, 9))
for ax, feature in zip(axes.flat, features):
    ax.hist(df[df["Manipulater"] == "Yes"][feature], bins=50, alpha=0.9, label="Yes")
    ax.hist(df[df["Manipulater"] == "No"][feature], bins=50, alpha=0.4, label="No")
    ax.set_title(feature)
    axins = inset_axes(ax, width="40%", height="40%", loc="upper right")
    axins.hist(df[df["Manipulater"] == "Yes"][feature], bins=50, alpha=0.9)
    axins.hist(df[df["Manipulater"] == "No"][feature], bins=50, alpha=0.5)

    axins.set_ylim(0, 5)
    axins.set_xticks([])
    axins.set_yticks([])

axes[0, 0].legend()

plt.tight_layout()
plt.savefig("pictures/detailed_histogram_2.pdf", bbox_inches="tight")
plt.show()
