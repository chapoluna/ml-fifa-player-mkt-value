###################################
# FIFA Players Makert Value
# Fernando Scalice Luna
# HarvardX Data Science Program
###################################

# This project will work with FIFA 19 data set available from the link below on Kaggle and aim to predict 
# soccer player market value based on player's attributes.
# Althought this data set contains stats from the video game series from EA, reliable and useful information  
# for the purpose of this course which is to train a machine learning algorithm.

# FIFA 19 complete player dataset
# https://www.kaggle.com/karangadiya/fifa19


# Required libraries

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("caret", repos = "http://cran.us.r-project.org")

# -----------------------------------------
# Data Wrangling
# -----------------------------------------

player_dat <- read.csv("data/data.csv")

glimpse(player_dat)

# Looking at the information available from this data set, we can see there are some of them which are useless
# for our model, such as logos and flags, so we can remove them

player_dat <- select(player_dat, -"Photo",-"Flag",-"Club.Logo")

p_dat <- player_dat

# All monetary values are stored as factors. We need them as numeric values
value_pattern <- "^(€)(\\d+\\.?\\d*)([MK]?)$"

scales <- setNames(c(1E6,1000), c("M","K"))  
p_dat[c("Value","Wage","Release.Clause")] <- lapply(p_dat[c("Value","Wage","Release.Clause")], function(v) {
  s <- str_match(as.character(v), value_pattern)
  value <- as.numeric(s[,3])
  value[is.na(value)] <- 0
  scale <- scales[s[,4]]
  scale[is.na(scale)] <- 1
  return(value*scale)
}) 

# Some player attributes (columns 26 to 51) are in the form of 99+9 and stored as factor. We can either transform 
# them to add up the second the number or just ignore them. To simplify, we'll just igonre them.

p_dat[,26:51] <- lapply(p_dat[,26:51], function(v) {
  gsub("(\\d{2})\\+(\\d*)", "\\1", as.character(v))
})

# Now we are almost ready. Let's see if we have NA's in our data set

qplot(rowSums(is.na(p_dat)), geom = "histogram") +
  scale_y_log10()

# As we can see, some players have almost all attributes as NA's which won't help us in our analysis. Let's
# get rid of them.

p_dat <- p_dat[-which(rowSums(is.na(p_dat)) > 1),]

# -----------------------------------------
# Create test and validation set
# Validation set will be 20% of FIFA data
# -----------------------------------------

set.seed(1)
test_index <- createDataPartition(y = p_dat$Value, times = 1, p = 0.2, list = FALSE)
train <- p_dat[-test_index,]
test <- p_dat[test_index,]

# ------------------------------
# Exploratory Data Analysis
# ------------------------------
