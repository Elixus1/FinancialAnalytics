# =====================================================
# Analysis of Dataset 1 - PCA
# =====================================================

library(readxl)
library(ggplot2)

# Optional plotting parameters
theme_set(theme_bw(base_size = 14))

# -----------------------------------------------------
# Import data
# -----------------------------------------------------

df <- read_excel("Dataset_Assignment_1.xlsx", sheet = 1)
temp_data <- read_excel("Dataset_Assignment_1.xlsx", sheet = 2)

# -----------------------------------------------------
# PCA
# Ignore Gender and Accommodation_Class
# -----------------------------------------------------

X <- df[, c("DTI", "FOIR", "LTV", "Age", "OldEmi")]

# Standardized PCA
pca <- prcomp(X,
              center = TRUE,
              scale. = TRUE)

# -----------------------------------------------------
# Covariance matrix of standardized data
# -----------------------------------------------------

X_std <- scale(X)

cov_matrix <- cov(X_std)

cat("Covariance matrix:\n")
print(round(cov_matrix, 2))

# -----------------------------------------------------
# Eigenvalues / Eigenvectors
# -----------------------------------------------------

eigenvalues <- pca$sdev^2
eigenvectors <- pca$rotation

cat("\nEigenvalues:\n")
print(eigenvalues)

cat("\nSum of eigenvalues:\n")
print(sum(eigenvalues))

# -----------------------------------------------------
# Principal Components
# -----------------------------------------------------

PCs <- pca$x

cat("\nShape of PCs:\n")
print(dim(PCs))

explained_variance <- eigenvalues / sum(eigenvalues)

cat("\nExplained variance:\n")

for(i in seq_along(explained_variance)){
  cat(sprintf("PC%d: %.3f\n",
              i,
              explained_variance[i]))
}

# -----------------------------------------------------
# Histograms of Principal Components
# -----------------------------------------------------

labels <- paste0("PC", 1:5)

hist(PCs[,1],
     probability = TRUE,
     breaks = 20,
     col = rgb(0,0,1,0.3),
     xlab = "Principal Component Value",
     main = "")

for(i in 2:5){
  
  hist(PCs[,i],
       probability = TRUE,
       breaks = 20,
       col = rgb(i/5,0,1-i/5,0.3),
       add = TRUE)
  
}

legend("topright",
       legend = labels,
       fill = sapply(1:5,
                     function(i) rgb(i/5,0,1-i/5,0.3)))

# -----------------------------------------------------
# Scree Plot
# -----------------------------------------------------

plot(explained_variance,
     type = "b",
     xaxt = "n",
     xlab = "Principal Component",
     ylab = "Explained Variance Ratio")

axis(1,
     at = 1:5,
     labels = labels)

grid()

# -----------------------------------------------------
# Loadings
# -----------------------------------------------------

features <- c("DTI", "FOIR", "LTV", "Age", "OldEmi")

loadings <- eigenvectors

cat("\nLoadings:\n")
print(round(loadings,3))

image(
  t(loadings[nrow(loadings):1,]),
  axes = FALSE,
  col = heat.colors(100)
)

axis(1,
     at = seq(0,1,length.out = 5),
     labels = labels)

axis(2,
     at = seq(0,1,length.out = 5),
     labels = rev(features))

# -----------------------------------------------------
# Loading Plot (PC1 vs PC2)
# -----------------------------------------------------

plot(c(-1,1),
     c(-1,1),
     type = "n",
     asp = 1,
     xlab = "PC1",
     ylab = "PC2")

abline(h = 0, col = "grey")
abline(v = 0, col = "grey")

for(i in 1:length(features)){
  
  arrows(0,
         0,
         eigenvectors[i,1],
         eigenvectors[i,2],
         length = 0.1)
  
  text(eigenvectors[i,1]*1.1,
       eigenvectors[i,2]*1.1,
       labels = features[i])
  
}

grid()

# -----------------------------------------------------
# Scores: PC1 vs PC2
# -----------------------------------------------------

colors <- c("Approve" = "blue",
            "Reject" = "red")

plot(PCs[,1],
     PCs[,2],
     col = colors[df$Decision],
     pch = 16,
     xlab = sprintf("PC1 (%.1f%% Variance)",
                    explained_variance[1]*100),
     ylab = sprintf("PC2 (%.1f%% Variance)",
                    explained_variance[2]*100))

legend("topright",
       legend = names(colors),
       col = colors,
       pch = 16)

grid()

