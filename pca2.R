# =====================================================
# PCA Analysis - Dataset 2
# =====================================================

library(readxl)

# -----------------------------------------------------
# Load data
# -----------------------------------------------------

df <- read_excel("data/Assignment_Dataset_2.xlsx", sheet = 1)

features <- c(
  "DSRI", "GMI", "AQI", "SGI",
  "DEPI", "SGAI", "ACCR", "LEVI"
)

X <- df[, features]

# -----------------------------------------------------
# Standardize data
# -----------------------------------------------------

X_std <- scale(X)

cov_matrix <- cov(X_std)

cat("Covariance matrix:\n")
print(round(cov_matrix, 2))

# -----------------------------------------------------
# PCA
# -----------------------------------------------------

pca <- prcomp(
  X,
  center = TRUE,
  scale. = TRUE
)

eigenvalues <- pca$sdev^2
eigenvectors <- pca$rotation
PCs <- pca$x

cat("\nEigenvalues:\n")
print(eigenvalues)

cat("\nSum of eigenvalues:\n")
print(sum(eigenvalues))

cat("\nShape of PCs:\n")
print(dim(PCs))

explained_variance <- eigenvalues / sum(eigenvalues)

cat("\nExplained variance:\n")

for(i in seq_along(explained_variance)){
  cat(sprintf("PC%d: %.3f\n",
              i,
              explained_variance[i]))
}

labels <- paste0("PC",1:8)

# -----------------------------------------------------
# Histograms of the first five principal components
# -----------------------------------------------------

hist(
  PCs[,1],
  probability = TRUE,
  breaks = 20,
  col = rgb(0,0,1,0.3),
  xlab = "Principal Component Value",
  main = ""
)

for(i in 2:5){
  
  hist(
    PCs[,i],
    probability = TRUE,
    breaks = 20,
    col = rgb(i/5,0,1-i/5,0.3),
    add = TRUE
  )
  
}

legend(
  "topright",
  legend = labels[1:5],
  fill = sapply(1:5,
                function(i) rgb(i/5,0,1-i/5,0.3))
)

# -----------------------------------------------------
# Scree plot
# -----------------------------------------------------

plot(
  explained_variance,
  type = "b",
  xaxt = "n",
  xlab = "Principal Component",
  ylab = "Explained Variance Ratio"
)

axis(
  1,
  at = 1:8,
  labels = labels
)

grid()

# -----------------------------------------------------
# Loadings
# -----------------------------------------------------

loadings <- eigenvectors

cat("\nLoadings:\n")
print(round(loadings,3))

image(
  t(loadings[nrow(loadings):1, ]),
  axes = FALSE,
  col = heat.colors(100)
)

axis(
  1,
  at = seq(0,1,length.out = 8),
  labels = labels
)

axis(
  2,
  at = seq(0,1,length.out = 8),
  labels = rev(features)
)

# -----------------------------------------------------
# Loading plot (PC1 vs PC2)
# -----------------------------------------------------

plot(
  c(-1,1),
  c(-1,1),
  type = "n",
  asp = 1,
  xlab = "PC1",
  ylab = "PC2"
)

abline(h = 0, col = "gray")
abline(v = 0, col = "gray")

for(i in 1:length(features)){
  
  arrows(
    0,
    0,
    eigenvectors[i,1],
    eigenvectors[i,2],
    length = 0.1
  )
  
  text(
    eigenvectors[i,1]*1.1,
    eigenvectors[i,2]*1.1,
    labels = features[i]
  )
  
}

grid()

# -----------------------------------------------------
# Scores: PC1 vs PC2
# -----------------------------------------------------

colors <- c(
  "Yes" = "blue",
  "No" = "red"
)

plot(
  PCs[,1],
  PCs[,2],
  col = colors[df$Manipulater],
  pch = 16,
  xlab = sprintf(
    "PC1 (%.1f%% Variance)",
    explained_variance[1]*100
  ),
  ylab = sprintf(
    "PC2 (%.1f%% Variance)",
    explained_variance[2]*100
  )
)

legend(
  "topright",
  legend = names(colors),
  col = colors,
  pch = 16
)

grid()