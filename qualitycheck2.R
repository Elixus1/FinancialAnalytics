# =====================================================
# Exploratory analysis of Dataset 2
# =====================================================

library(readxl)
library(ggplot2)

theme_set(theme_bw(base_size = 14))

# -----------------------------------------------------
# Load data
# -----------------------------------------------------

df <- read_excel("data/Assignment_Dataset_2.xlsx", sheet = 1)

features <- c(
  "DSRI", "GMI", "AQI", "SGI",
  "DEPI", "SGAI", "ACCR", "LEVI"
)

# -----------------------------------------------------
# Function to calculate class ratio
# -----------------------------------------------------

ratio <- function(column, class1, class2){
  
  counts <- table(df[[column]])
  
  n1 <- counts[class1]
  n2 <- counts[class2]
  
  cat(class1, ":", class2, "=", n1, ":", n2, "\n")
  cat("Ratio =", round(as.numeric(n1/n2),2), "\n")
  
}

ratio("Manipulater","Yes","No")

# -----------------------------------------------------
# Missing values
# -----------------------------------------------------

cat("Missing values:\n")
print(colSums(is.na(df)))

# -----------------------------------------------------
# Boxplots
# -----------------------------------------------------

par(mfrow = c(2,4))

for(feature in features){
  
  boxplot(df[[feature]],
          main = feature)
  
}

par(mfrow = c(1,1))

# -----------------------------------------------------
# Histograms
# -----------------------------------------------------

par(mfrow = c(2,4))

for(feature in features){
  
  hist(df[[feature]],
       breaks = 100,
       main = feature,
       xlab = feature,
       col = "lightgray")
  
}

par(mfrow = c(1,1))

# -----------------------------------------------------
# Density plots
# -----------------------------------------------------

par(mfrow = c(2,4))

for(feature in features){
  
  yes <- density(df[df$Manipulater == "Yes", feature][[1]])
  no  <- density(df[df$Manipulater == "No", feature][[1]])
  
  plot(yes,
       main = feature,
       xlab = feature,
       col = "blue",
       lwd = 2)
  
  lines(no,
        col = "red",
        lwd = 2)
  
  legend("topright",
         legend = c("Yes","No"),
         col = c("blue","red"),
         lwd = 2,
         cex = 0.8)
  
}

par(mfrow = c(1,1))

# -----------------------------------------------------
# Overlay histograms (zoom)
# -----------------------------------------------------

par(mfrow = c(2,4))

for(feature in features){
  
  hist(df[df$Manipulater=="Yes",feature][[1]],
       breaks = 50,
       col = rgb(0,0,1,0.5),
       xlab = feature,
       main = feature,
       ylim = c(0,5))
  
  hist(df[df$Manipulater=="No",feature][[1]],
       breaks = 50,
       col = rgb(1,0,0,0.5),
       add = TRUE)
  
}

legend("topright",
       legend = c("Yes","No"),
       fill = c(rgb(0,0,1,0.5),
                rgb(1,0,0,0.5)))

par(mfrow = c(1,1))

# -----------------------------------------------------
# Detailed histograms
# (R has no simple equivalent to matplotlib inset_axes)
# -----------------------------------------------------

par(mfrow = c(4,2))

for(feature in features){
  
  hist(df[df$Manipulater=="Yes",feature][[1]],
       breaks = 50,
       col = rgb(0,0,1,0.8),
       xlab = feature,
       main = feature)
  
  hist(df[df$Manipulater=="No",feature][[1]],
       breaks = 50,
       col = rgb(1,0,0,0.4),
       add = TRUE)
  
}

legend("topright",
       legend = c("Yes","No"),
       fill = c(rgb(0,0,1,0.8),
                rgb(1,0,0,0.4)))

par(mfrow = c(1,1))