# Note Elias Jedam: Python Code translated into R by basically Chatgpt
library(readxl)
library(dplyr)
library(ggplot2)
getwd()
#get data
df <- read_excel("data/Dataset_Assignment_1.xlsx", sheet = 1)

# categorical variables
df <- df %>%
  mutate(
    Gender = recode(Gender,
                    "Male" = 0,
                    "Female" = 1),
    Accommodation_Class = recode(Accommodation_Class,
                                 "Rented" = 0,
                                 "Non_Rented" = 1)
  )

# general check of data quality
colSums(is.na(df))

features <- c("DTI", "FOIR", "LTV", "Age")

#histogram oldemi
hist_info <- hist(df$OldEmi,
                  main = "",
                  xlab = "OldEmi")

text(hist_info$mids,
     hist_info$counts,
     labels = hist_info$counts,
     pos = 3)

# pdf("pictures/table1_OldEmi.pdf")
# hist(...)
# dev.off()

# gender distributions
counts <- table(df$Gender)

male <- counts["0"]
female <- counts["1"]

cat("Male : Female =", male, ":", female, "\n")
cat("Ratio =", round(as.numeric(male/female), 2), "\n")

# AC distribution
counts <- table(df$Accommodation_Class)

rented <- counts["0"]
non_rented <- counts["1"]

cat("Rented : Non Rented =", rented, ":", non_rented, "\n")
cat("Ratio =", round(as.numeric(rented/non_rented), 2), "\n")

#histogram of all features
features <- c(
  "DTI", "FOIR", "LTV",
  "Age", "OldEmi",
  "Gender", "Accommodation_Class"
)

par(mfrow = c(3,3))

for(feature in features){
  hist(df[[feature]],
       main = feature,
       xlab = feature,
       col = "lightgray")
}

par(mfrow = c(1,1))

#boxplots
par(mfrow = c(2,4))

for(feature in features){
  boxplot(df[[feature]],
          main = feature)
}

par(mfrow = c(1,1))


prop.table(table(df$Gender, df$Decision), margin = 1)
prop.table(table(df$Accommodation_Class, df$Decision), margin = 1)

# Countplots
ggplot(df, aes(x = factor(Gender), fill = Decision)) +
  geom_bar(position = "dodge") +
  labs(x = "Gender")

ggplot(df, aes(x = factor(Accommodation_Class), fill = Decision)) +
  geom_bar(position = "dodge") +
  labs(x = "Accommodation Class")

#KDE plots
features <- c("DTI","FOIR","LTV","Age","OldEmi")

par(mfrow = c(2,3))

for(feature in features){
  
  plot(
    density(df[df$Decision=="Approve", feature][[1]]),
    main = feature,
    xlab = feature,
    col = "blue",
    lwd = 2
  )
  
  lines(
    density(df[df$Decision=="Reject", feature][[1]]),
    col = "red",
    lwd = 2
  )
  
  legend("topright",
         legend = c("Approve","Reject"),
         col = c("blue","red"),
         lwd = 2)
}

# Zoom OldEmi
plot(
  density(df[df$Decision=="Approve","OldEmi"][[1]]),
  xlim = c(20000,40000),
  main = "OldEmi (Zoom)",
  xlab = "OldEmi",
  col = "blue",
  lwd = 2
)

lines(
  density(df[df$Decision=="Reject","OldEmi"][[1]]),
  col = "red",
  lwd = 2
)

legend("topright",
       legend = c("Approve","Reject"),
       col = c("blue","red"),
       lwd = 2)

par(mfrow = c(1,1))

#histogram seperated by decision
par(mfrow = c(2,3))

for(feature in features){
  
  hist(df[df$Decision=="Approve", feature][[1]],
       breaks = 20,
       col = rgb(0,0,1,0.4),
       main = feature,
       xlab = feature)
  
  hist(df[df$Decision=="Reject", feature][[1]],
       breaks = 20,
       col = rgb(1,0,0,0.4),
       add = TRUE)
  
  legend("topright",
         legend = c("Approve","Reject"),
         fill = c(rgb(0,0,1,0.4), rgb(1,0,0,0.4)))
}

par(mfrow = c(1,1))

#histogram oldemi
hist(df[df$Decision=="Approve","OldEmi"][[1]],
     breaks = 20,
     col = rgb(0,0,1,0.4),
     xlab = "OldEmi",
     main = "OldEmi")

hist(df[df$Decision=="Reject","OldEmi"][[1]],
     breaks = 20,
     col = rgb(1,0,0,0.4),
     add = TRUE)

legend("topright",
       legend = c("Approve","Reject"),
       fill = c(rgb(0,0,1,0.4), rgb(1,0,0,0.4)))

