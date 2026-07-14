# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from latex_io import format_matrix

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
)
plt.rcParams["figure.figsize"] = (8, 5)

# %%
# import dataset
df = pd.read_excel("data/Assignment_Dataset_2.xlsx", sheet_name=0)


# %%
# do pca for df, but ignore gender and accomodation_class
features = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "ACCR", "LEVI"]
X = df[features].to_numpy()
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_std = (X - mean) / std  # standardize data
varicance = np.var(X_std)
cov = np.cov(X_std.T)  # covariance matrix
print("cov matrix: ")
print(format_matrix(cov, 2))

# %%
# Eigenvalues / Eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
print("Eigenvalues:")
print(eigenvalues)
print("Sum eigenvalues: ", sum(eigenvalues))

# Projection onto principal components
PCs = X_std @ eigenvectors
print("Shape of PCs:", PCs.shape)

explained_variance = eigenvalues / np.sum(eigenvalues)

print("Explained variance:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.3f}")
labels = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"]
# plt.figure(figsize=(8, 5))
for i in range(5):
    plt.hist(PCs[:, i], bins=20, alpha=0.5, density=True, label=labels[i])

plt.xlabel("Principal Component Value")
plt.ylabel("Density")
plt.legend()
plt.savefig("pictures/pca2.pdf", bbox_inches="tight")
plt.show()

# %%
# Scree
# plt.figure(figsize=(6, 4))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker="o")
plt.xticks(range(1, len(labels) + 1), labels)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)
plt.savefig("pictures/screeplot2.pdf", bbox_inches="tight")
plt.show()

# %%
loadings = pd.DataFrame(eigenvectors, index=features, columns=labels)
print(loadings)
plt.figure(figsize=(7, 4))
plt.imshow(loadings, cmap="coolwarm", aspect="auto")
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(features)), features)
plt.colorbar(label="Loading")
plt.xlabel("Principal Components")
plt.ylabel("Original Features")
plt.savefig("pictures/loading_pca2.pdf", bbox_inches="tight")
plt.show()


# %%
# plt.figure(figsize=(7, 7))
for i, feature in enumerate(features):
    plt.arrow(
        0,
        0,
        eigenvectors[i, 0],
        eigenvectors[i, 1],
        head_width=0.03,
        length_includes_head=True,
    )
    plt.text(eigenvectors[i, 0] * 1.1, eigenvectors[i, 1] * 1.1, feature, fontsize=11)

plt.axhline(0, color="gray", linewidth=0.5)
plt.axvline(0, color="gray", linewidth=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.axis("equal")
plt.grid(True)
plt.show()


# %%

colors = {"Yes": "tab:blue", "No": "tab:red"}

# plt.figure(figsize=(7, 6))

for decision in colors:
    mask = df["Manipulater"] == decision

    plt.scatter(PCs[mask, 0], PCs[mask, 1], label=decision, alpha=0.7)  # PC1  # PC2

plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}% Varianz)")
plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}% Varianz)")
plt.legend()
plt.grid(True)
plt.savefig("pictures/pc1vspc2_2.pdf", bbox_inches="tight")
plt.show()


# %%
