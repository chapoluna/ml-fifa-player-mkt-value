###################################
# FIFA Players Makert Value
# Fernando Scalice Luna
# HarvardX Data Science Program
###################################

# This project will work with FIFA 19 data set available from the link below on Kaggle and aim to predict 
# soccer player market value based on player's attributes.
# Althought this data set contains information from the video game series from EA, the data set contains 
# reliable and useful information to train a machine learning algorithm.

# FIFA 19 complete player dataset
# https://www.kaggle.com/karangadiya/fifa19


# ------------------------------
# Create test and validation set
# ------------------------------

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

player_dat <- read.csv("data/data.csv")

glimpse(player_dat)

# Looking at the information available from this data set, we can see there are some of them which are useless
# for our model, such as logos and flags

player_dat <- select(player_dat, -"Photo",-"Flag",-"Club.Logo")

as.numeric(as.character(player_dat$Value))[player_dat$Value])
player_dat[1,]

# Validation set will be 20% of FIFA data

set.seed(1)
test_index <- createDataPartition(y = player_dat$value, times = 1, p = 0.2, list = FALSE)
train <- player_dat[-test_index,]
test <- player_dat[test_index,]






# ------------------------------
# Exploratory Data Analysis
# ------------------------------

glimpse(fifa_dat)