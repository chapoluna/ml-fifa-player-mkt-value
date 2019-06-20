###################################
# FIFA Players Makert Value
# Fernando Scalice Luna
# HarvardX Data Science Program
###################################

# ------------------------------
# 1. Introduction
# ------------------------------

# This project will work with FIFA 19 data set available from the link below on Kaggle and aim to use 
# some machine learning algorithms to predict soccer player market value based on player's attributes.
# This study is for learning purposes only, as it contains stats from the video game series from EA, 
# and not real world information about the athletes. Nevertheless, the information available is reliable 
# and useful to achieve the goals of this project.

# Note: to run this project successfuly you'll first need to download the dataset from the link below 
# and place the unziped file under 'data' directory
# FIFA 19 complete player dataset
# https://www.kaggle.com/karangadiya/fifa19


# Required libraries

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

# -----------------------------------------
# Data Wrangling
# -----------------------------------------

player_dat <- read.csv("data/data.csv")

glimpse(player_dat)

# Looking at the information available from this data set, we can see there are some of them which are useless
# for our model, such as logos and flags, so we can remove them

player_dat <- select(player_dat, -"Photo",-"Flag",-"Club.Logo",-"Body.Type",-"Real.Face")

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

# Before we start building our models, we must normalize Overall, Age and Wage data for a correct kNN approach 
# in the future.
p_dat[c("Overall_s","Age_s", "Wage_s")] <- lapply(p_dat[c("Overall","Age", "Wage")], function(x) {
  (x - min(x)) / (max(x) - min(x))
})

# ------------------------------
# 2. Exploratory Data Analysis
# ------------------------------

# Distribution of Market Value
avg_value <- data.frame(data = "Average", value = mean(p_dat$Value))
med_value <- data.frame(data = "Median", value = median(p_dat$Value))
averages <- rbind.data.frame(avg_value, med_value)
averages <- averages %>% 
  mutate(labbel =  sprintf("%s = %s", data, format(value, big.mark = ",")))

p_dat %>%
  ggplot(aes(Value)) +
  geom_histogram() +
  geom_vline(aes(xintercept = value, color = data), data = averages, size = 1, show.legend = F) +
  geom_text(aes(x=value*.85, y=1000, color = data, label = labbel), data = averages, 
            size = 3, angle=90, show.legend = F) +
  scale_x_continuous(name="Market Value (in Euros)", labels = scales::comma, trans = "log10") +
  scale_y_continuous(name="Number of players", labels = scales::comma)

# R warns us that 252 players have values equals to 0, so they were dropped out from our graph.
# As we can see, on average players market value is €2,4M (with a median value of €675K) and a few players worthing as much as €100M
# like Neymar, Messi and De Bruyne

# Distribution of Overall Score
avg_value <- data.frame(data = "Average", value = mean(p_dat$Overall))
med_value <- data.frame(data = "Median", value = median(p_dat$Overall))
averages <- rbind.data.frame(avg_value, med_value)
averages <- averages %>% 
  mutate(labbel =  sprintf("%s = %s", data, format(value, big.mark = ",")))

p_dat %>%
  ggplot(aes(Overall)) +
  geom_histogram() +
  geom_vline(aes(xintercept = value, color = data), data = averages, size = 1, show.legend = F) +
  geom_text(aes(x=value, y=1000, color = data, label = labbel), data = averages, 
            size = 3, angle=90, show.legend = F, nudge_x = -0.01) +
  scale_x_continuous(name="Overall Score", labels = scales::comma, trans = "log10") +
  scale_y_continuous(name="Number of players", labels = scales::comma)

# Overall score seems more normally distributed as median and mean are very close.

# Now, let's see how Overall Score is related to Market Value
top_players_mkt <- p_dat %>% filter(Value >= 90E6)
top_players_scr <- p_dat %>% filter(Overall >= 90)

p_dat %>% ggplot(aes(Overall, Value)) +
  geom_point(alpha = .5, show.legend = FALSE) +
  geom_point(data = top_players_mkt, color = "red", size=3, show.legend = FALSE) +
  geom_text(aes(label = top_players_mkt$Name, hjust=1,vjust=0), data = top_players_mkt, nudge_x = -0.5) +
  geom_point(data = top_players_scr, color = "blue", size=2, alpha=.5, show.legend = FALSE) +
  scale_y_continuous(name="Market Value (in Euros)", labels = scales::comma)

# It comes as no suprise to see top players in market value having a overall score above 90. But we can also
# see top players in overall score having a market value as half as the top players (blue dots).

# Now, let's see how Age is related to Market Value
p_dat %>% ggplot(aes(Age, Value, group = Age)) +
  geom_boxplot() +
  geom_point(data = top_players_mkt, color = "red", size=3) +
  geom_point(data = top_players_scr, color = "blue", size=2, alpha=.5) +
  scale_y_continuous(name="Market Value (in Euros)", labels = scales::comma, trans = "log10")

# As players age, their values tend to be higher until they reach their peak performance around their 27-30s,
# then their values start to decrease. However, as we've seen above just few players worth huge
# amount of money. Age does play a role here, but it does not to see a key factor.

# Top European Clubs like Barcelona, Real Madrid, and Bayern tend to build high valuable squads.
p_dat %>% group_by(Club) %>%
  summarise(m_value = round(sum(Value)/1E6)) %>%
  arrange(desc(m_value)) %>% head(10) %>%
  ggplot(aes(x = as.factor(Club) %>%
               fct_reorder(m_value), m_value, label = m_value)) +
  geom_bar(stat = "identity") +
  geom_text(hjust = 2, size=4, color="white") + 
  scale_y_continuous(name="Value (in millions Euros)", labels = scales::comma) +
  xlab("Club") +
  ggtitle("Squad Market Value (in millions Euros)") +
  coord_flip()

# What about average market value for each top 10 Club?
p_dat %>% group_by(Club) %>%
  summarise(m_value = median(Value)) %>%
  arrange(desc(m_value)) %>% head(10) %>%
  ggplot(aes(x = as.factor(Club) %>%
               fct_reorder(m_value), m_value, label = format(m_value, big.mark = ","))) +
  geom_bar(stat = "identity") +
  geom_text(hjust = 1.5, size=3, color="white",labels = scales::comma) + 
  scale_y_continuous(name="Median Value (in Euros)", labels = scales::comma) +
  xlab("Club") +
  ggtitle("Squad Average Market Value (in Euros)") +
  coord_flip()

top_clubs <- p_dat %>% group_by(Club) %>%
  summarise(squad_score = mean(Overall)) %>%
  arrange(desc(squad_score)) %>% 
  top_n(10) %>% 
  pull(Club)

p_dat %>%
  filter(Club %in% top_clubs) %>%
  ggplot(aes(Club, Value)) +
  geom_boxplot() +
  geom_point(data = top_players_mkt, color = "red", size=3) +
  geom_point(data = top_players_scr, color = "blue", size=2, alpha=.5) +
  scale_y_continuous(name="Value (in Euros)", labels = scales::comma) +
  xlab("Club") +
  ggtitle("Players' Value and Top Clubs") +
  coord_flip()

p_dat %>%
  mutate(TopClub = Club %in% top_clubs) %>%
  ggplot(aes(x=TopClub, y=Value)) +
  geom_boxplot(outlier.shape = NA) +
  geom_point(data = top_players_mkt %>%
               mutate(TopClub = Club %in% top_clubs),
               color = "red", size=3) +
  geom_point(data = top_players_scr %>%
               mutate(TopClub = Club %in% top_clubs),
               color = "blue", size=2, alpha=.5) +
  scale_y_continuous(name="Value (in Euros)", labels = scales::comma) +
  xlab("Top 10 Club") +
  ggtitle("Players' Value and Top Clubs")

# We can check if TopClub variance and non-TopClub variance are relevent for our analysis by running an ANOVA test.
# And as we can see, from p-value < 0.05, Club is a relevant variable.
res.anova <- aov(Value ~ as.factor(TopClub), data = p_dat %>% mutate(TopClub = Club %in% top_clubs))
summary(res.anova)

# Another approach is grouping Clubs (as we have too many Clubs (+600))
x <- p_dat %>% 
  group_by(Club) %>%
  summarise(avg_score = mean(Overall))
row_names <- x$Club
x <- x[,-1] %>% as.matrix()
rownames(x) <- row_names
d <- dist(x)
h <- hclust(d)

groups <- cutree(h, k = 10)
split(names(groups), groups)

g <- data.frame(Club = names(groups), Club_Group = as.character(groups), row.names = NULL)
p_dat <- p_dat %>% 
  left_join(g, by = "Club")

# Attackers tend to be more valuable than other positions
p_dat %>%
  ggplot(aes(x = as.factor(Position) %>%
               fct_reorder(Value), Value)) +
  geom_boxplot() +
  geom_point(data = top_players_mkt, color = "red", size=3) +
  geom_point(data = top_players_scr, color = "blue", size=2, alpha=.5) +
  scale_y_continuous(name="Value (in Euros)", labels = scales::comma, trans = "log10") +
  xlab("Club") +
  ggtitle("Players' Value and Position") +
  coord_flip()

res.anova <- aov(Value ~ as.factor(Position), data = p_dat)
summary(res.anova)

# Another factor to consider is Players' current Wage and their Market Value.
p_dat %>% ggplot(aes(Wage, Value)) +
  geom_point(alpha = .5, show.legend = FALSE) +
  geom_smooth() +
  scale_y_continuous(name="Value (in Euros)", labels = scales::comma) +
  scale_x_continuous(name="Wage (in Euros)", labels = scales::comma)

# ------------------------------
# 3. Building the Model
# ------------------------------

# -----------------------------------------
# Create test and validation set
# Validation set will be 20% of FIFA data
# -----------------------------------------

set.seed(1)
test_index <- createDataPartition(y = p_dat$Value, times = 1, p = 0.2, list = FALSE)
train_set <- p_dat[-test_index,]
test_set <- p_dat[test_index,]

# The RMSE function to evaluate our models
RMSE <- function(true_ratings, predicted_ratings){ 
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# First Model - Simple Average
# Yu,i = mu + Eu,i

mu_hat <- mean(train_set$Value) 
mu_hat

naive_rmse <- RMSE(test_set$Value, mu_hat) 
naive_rmse

rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)

# Second Model - Local Weighted Regression
# Yu,i = f(Overall) + Eu,i
fit <- loess(Value ~ Overall, data = train_set, degree = 2, span = .1)

train_set %>% mutate(smooth = fit$fitted) %>% 
  ggplot(aes(Overall, Value)) +
  geom_point(size = 3, alpha = .5, color = "black") + 
  geom_line(aes(Overall, smooth), size = 1, color="red")

value_hat <- predict(fit, test_set$Overall, type = "prob")

l_ovr_rmse <- RMSE(test_set$Value, value_hat)
rmse_results <- bind_rows(rmse_results, tibble(method = "Loess with Overall Score", RMSE = l_ovr_rmse))
l_ovr_rmse

# Third Model - Regression with Overall Score and Age
# Yu,i = f(Overall,Age) + Eu,i
span <- seq(0.1, 0.9, .1)
loess_rmses <- sapply(span,function(x) {
  fit <- loess(Value ~ Overall + Age, data = train_set, degree = 2, span = x)
  RMSE(train_set$Value, fit$fitted)
})
plot(loess_rmses)

fit <- loess(Value ~ Overall + Age, data = train_set, degree = 2, span = span[which.min(loess_rmses)])
value_hat <- predict(fit, test_set[c("Overall","Age")], type = "prob")

loess2_rmse <- RMSE(test_set$Value, value_hat)
rmse_results <- bind_rows(rmse_results, tibble(method = "Loess with Overall Score and Age", RMSE = loess2_rmse))
loess2_rmse

# Fourth Model - Regression Tree with Overall Score, Age and Club Group

# Let's try to find the best cp for our model. 
# Warning: This may take some time to run.
train_rpart <- train(Value ~ Overall + Age + Club_Group,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, len = 10)), data = train_set)
ggplot(train_rpart)

# According to the simulation, the best cp for our model is 0, but this will lead to overfitting our model
y_hat_rpart <- predict(train_rpart, test_set[c("Overall","Age","Club_Group")])
model_rpart_rmse <- RMSE(test_set$Value, y_hat_rpart)
rmse_results <- bind_rows(rmse_results, tibble(method = "Regression Tree (Overall,Age,Club)", RMSE = model_rpart_rmse))
model_rpart_rmse

# 5th Model - Regression Tree with Overall Score, Age, Club Group and Wage
train_rpart_2 <- train(Value ~ Overall + Age + Club_Group + Wage,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, len = 3)), data = train_set)
ggplot(train_rpart_2)

y_hat_rpart <- predict(train_rpart_2, test_set[c("Overall","Age","Club_Group","Wage")])
model_rpart2_rmse <- RMSE(test_set$Value, y_hat_rpart)
rmse_results <- bind_rows(rmse_results, tibble(method = "Regression Tree (Overall,Age,Club,Wage)", RMSE = model_rpart2_rmse))
model_rpart2_rmse

# 6th - Random Forest OVerall + Age
nodesize <- seq(1, 51, 10)
rmses_rf <- sapply(nodesize, function(ns){
  train(Value ~ Overall + Age, 
        method = "rf", 
        data = train_set, 
        nodesize = ns)$results$RMSE 
})
qplot(nodesize, rmses_rf)

train_rf <- randomForest(Value ~ Overall + Age, 
                         data = train_set, 
                         nodesize = nodesize[which.min(rmses_rf)])
y_hat_rf <- predict(train_rf, test_set[c("Overall","Age")])
model_rf_rmse <- RMSE(test_set$Value, y_hat_rf)
rmse_results <- bind_rows(rmse_results, tibble(method = "Random Forest (Overall,Age)", RMSE = model_rf_rmse))
model_rf_rmse

# 7th - Random Forest Overall + Age + Wage + Club_Group
# Warning: this code may take several hours to run
nodesize <- seq(1, 51, 10)
rmses_rf <- sapply(nodesize, function(ns){
  train(Value ~ Overall + Age + Club_Group + Wage, 
        method = "rf", 
        data = train_set, 
        tuneGrid = data.frame(mtry = train_rf$bestTune$mtry),
        nodesize = ns)$results$RMSE 
})
qplot(nodesize, rmses_rf)

train_rf <- randomForest(Value ~ Overall + Age + Club_Group + Wage, 
                         data = train_set,
                         nodesize = nodesize[which.min(rmses_rf)])
y_hat_rf <- predict(train_rf, test_set[c("Overall","Age","Club_Group","Wage")])
model_rf_rmse <- RMSE(test_set$Value, y_hat_rf)
rmse_results <- bind_rows(rmse_results, tibble(method = "Random Forest (Overall,Age,ClubGroup,Wage)", 
                                               RMSE = model_rf_rmse))
model_rf_rmse

# 8th Model - Nearest Neighbours with Overall Score, Age and Club

control <- trainControl(method = "cv", number = 10, p = .9)
train_knn <- train(Value ~ ., method = "knn", 
                   data = train_set[c("Value","Overall_s","Age_s","Club_Group","Wage_s")],
                   tuneGrid = data.frame(k = seq(1,15,2)),
                   trControl = control)
ggplot(train_knn, highlight = TRUE)

y_hat_knn <- predict(train_knn, test_set[c("Overall_s","Age_s","Club_Group", "Wage_s")], "raw")
model_knn_rmse <- RMSE(test_set$Value, y_hat_knn)
rmse_results <- bind_rows(rmse_results, 
                          tibble(method = "Knn", RMSE = model_knn_rmse))

train_loess <- train(Value ~ ., method = "gamLoess", 
                     data = train_set[c("Value","Overall_s","Age_s","Wage_s","Club_Group")],
                     tuneGrid = data.frame(span = seq(0.15, 0.65, len = 10), degree = 1),
                     trControl = control)
ggplot(train_loess, highlight = TRUE)

y_hat_loess <- predict(train_loess, test_set[c("Overall_s","Age_s","Wage_s", "Club_Group")], "raw")
model_loess_rmse <- RMSE(test_set$Value, y_hat_loess)
rmse_results <- bind_rows(rmse_results, 
                          tibble(method = "gamLoess", RMSE = model_loess_rmse))

fit_knnreg <- knnreg(Value ~ Overall + Age + Wage + Club_Group, data = train_set, k = 3)

y_hat_kreg <- predict(fit_knnreg, test_set[c("Overall","Age","Wage","Club_Group")], "raw")
model_kreg_rmse <- RMSE(test_set$Value, y_hat_kreg)
rmse_results <- bind_rows(rmse_results, 
                          tibble(method = "Knnreg", RMSE = model_kreg_rmse))


# Final Results
rmse_results
