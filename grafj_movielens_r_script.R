#' ---
#' title: 'MovieLens Project: Prediction of Movie Ratings'
#' author: "Jonas Graf"
#' date: "`r format(Sys.Date(),'%e - %B - %Y')`"
#' output: 
#'     pdf_document: 
#'       highlight: kate
#'       toc: yes
#'       number_sections: yes
#'       toc_depth: 1
#'       latex_engine: lualatex
#' mainfont: Source Sans Pro
#' sansfont: Candara
#' ---
#'   
#' ***
#'   
## ----setup, include=FALSE---------------------------------------------------
# knitr::opts_chunk$set(echo = TRUE, tidy.opts=list(width.cutoff=60), tidy=TRUE)

#' \newpage
#' # Data Setup
## ---- include=TRUE, echo=TRUE, message=FALSE, warning=FALSE-----------------

# Libraries: install
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(forcats)) install.packages("forcats", repos = "http://cran.us.r-project.org")
if(!require(formatR)) install.packages("formatR", repos = "http://cran.us.r-project.org")

# Libraries: load
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(ggplot2)
library(kableExtra)
library(tidyr)
library(stringr)
library(forcats)
library(formatR)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Setting options to allow download
options(timeout = max(10000, getOption("timeout")))

# Downloading movielens file
dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# Using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#' 
#'   
#' ***
#'   
#' 
#' # Introduction
#' 
#' The overarching aim of this project is to predict movie ratings (rating scores 0-5) using a subset of the MovieLens dataset. 
#' 
#' This subset contains ~10,000,000 movie ratings. For this project, these ratings will be divided into 9,000,000 (90%) for training and 1,000,000 (10%) for validation purposes. The training dataset comprises ratings from ~70,000 users evaluating ~11,000 different movies. Of note, the rated movies are grouped in genres, e.g. Adventure, Action, Drama, Horror, and Thriller.
#' 
#' First, the data subset will be described. Second, a rating prediction model will be fitted on the training dataset. Third, the model deemed optimal will be applied to the validation dataset. The fitted model will be evaluated according to the Root Mean Squared Error (RMSE), which should be below 0.86490.
#' 
#' \[\boxed{\mbox{RMSE} = \sqrt{\frac{1}{N} \sum_{i}^{} \left( \hat{y}_{i} - y_{i} \right)^2 }}\]
#' 
## ---- include=FALSE, echo=FALSE---------------------------------------------
# The following RMSE function will be applied:
RMSE <- function(true_ratings = NULL, predicted_ratings = NULL) {
    sqrt(mean((true_ratings - predicted_ratings)^2))
}

#' 
#' 
#' 
#' A regularized model including 'Movie', 'User', 'Genre', 'Date of Rating' as well as 'Year of Release' fulfills the overarching aim of the project by reaching an RMSE of 0.8620.
#' 
#' 
#' 
#' ***
#' \newpage
#' # Methods & Analysis
#' 
#' ## Descriptive data exploration
#' 
#' First, an exploration of the raw training data is warranted.  
#' 
#' ### First six data entries: overview  
#' 
#' The head function in R provides us a first descriptive overview of the variables.  
#' 
## ---- echo=FALSE, include=TRUE----------------------------------------------
# Training dataset: first six entries
kable(head(edx), format = "latex") %>% kable_styling(latex_options = c("scale_down", "HOLD_position"))

#'   
#' Movie title and year of release appear to be combined within the 'title' variable. Further, the time point of the rating has been captured as a time stamp.  
#'   
#' 
#' ### Users and movies (counts) included in the training data subset: table.
## ---- echo=FALSE, include=TRUE----------------------------------------------
# Training dataset: n of users and movies
edx %>% summarize(users = n_distinct(userId), movies = n_distinct(movieId)) %>% kable(format = "simple", align = 'c')

#'   
#' ### Classes of variables:table
## ---- echo=FALSE, include=TRUE----------------------------------------------
# Training dataset: classes of variables
data.frame(variable = names(edx), class = c(class(edx$movieId)[1], class(edx$rating)[1], class(edx$timestamp)[1], class(edx$title)[1], class(edx$genres)[1], class(edx$userId)[1])) %>% kable(format = "simple", align = 'c') 

#' 
#'   
#' 
#' ### Formatting tasks
#' The descriptive data exploration revealed three formatting tasks:  
#' 1. Formatting 'timestamp' into readable data format.  
#' 2. Splitting the condensed 'genres' categorization.  
#' 3. Separating year of release from title.  
#'   
#' Of note, 'timestamp' represents seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.  
#' *Source:* *https://files.grouplens.org/datasets/movielens/ml-10m-README.html*  
#' 
#' \newpage
#' ## Formatting training & validation datasets
#' 
#' Next, the formatting task need to be tackled.  
#' 
#' ### Formatting 'timestamp' into readable data format: result.  
## ---- echo=TRUE, include=TRUE-----------------------------------------------
# 1.) Format 'timestamp' into readable data format
edx$date <- as.POSIXct(edx$timestamp, origin="1970-01-01")
validation$date <- as.POSIXct(validation$timestamp, origin="1970-01-01")
## training
edx$year_rating <- format(edx$date,"%Y")
edx$month_rating <- format(edx$date,"%m")
edx$day_rating <- format(edx$date,"%d")
edx$date_rating <- paste(edx$year_rating, "-", edx$month_rating,"-", edx$day_rating)

## validation
validation$year_rating <- format(validation$date,"%Y")
validation$month_rating <- format(validation$date,"%m")
validation$day_rating <- format(validation$date,"%d")
validation$date_rating <- paste(validation$year_rating, "-", validation$month_rating, "-", validation$day_rating)

## Result after task 1
kable(head(edx), format = "latex") %>% 
  kable_styling(latex_options = c("scale_down", "HOLD_position")) 

#'   
#' ### Splitting the 'genres' variable: result.  
#'   
#' 
## ---- echo=TRUE, include=TRUE-----------------------------------------------
# 2.) Split the condensed 'genres' categorization.
edx <- edx %>% separate_rows(genres, sep = "\\|")
validation <- validation %>% separate_rows(genres, sep = "\\|")
## Result after tasks 1 & 2
kable(head(edx), format = "latex") %>% 
  kable_styling(latex_options = c("scale_down", "HOLD_position")) 

#'   
#' \newpage
#' 
#' 
#' ### Separating year of release from title variable.  
## ---- echo=TRUE, include=TRUE-----------------------------------------------
edx <- edx %>%
# Removing whitespace
mutate(title = str_trim(title)) %>%
# Extracting year of release
extract(title, c("titleTemp", "year_release"), remove = FALSE, regex = "^(.*) \\(([0-9 \\-]*)\\)$") %>%
# Handling years with "-"
mutate(year_release = if_else(str_length(year_release) > 4, as.integer(str_split(year_release, "-", simplify = TRUE)[1]), as.integer(year_release))) %>%
# Redefining title variable
mutate(title = if_else(is.na(titleTemp), title, titleTemp)) %>%
select(-titleTemp)    

validation <- validation %>%
mutate(title = str_trim(title)) %>%
extract(title, c("titleTemp", "year_release"), remove = FALSE, regex = "^(.*) \\(([0-9 \\-]*)\\)$") %>%
mutate(year_release = if_else(str_length(year_release) > 4,  as.integer(str_split(year_release, "-", simplify = TRUE)[1]), as.integer(year_release))) %>%
mutate(title = if_else(is.na(titleTemp), title, titleTemp)) %>%
select(-titleTemp)

## Result after tasks 1, 2 & 3
kable(head(edx), format = "latex") %>% 
  kable_styling(latex_options = c("scale_down", "HOLD_position"))  

#' 
#' ### Converting & selecting columns of interest: result.  
#'   
## ---- echo=TRUE, include=FALSE----------------------------------------------
# Selecting & converting columns of interest.
## training
edx <- edx %>% 
select(userId, movieId, rating, title, genres, date_rating, year_rating, year_release)
edx$year_release <- as.factor(edx$year_release)
edx$year_rating <- as.factor(edx$year_rating)
edx$genres <-as.factor(edx$genres)
edx$userId <-as.factor(edx$userId)
edx$movieId <-as.factor(edx$movieId)
edx$date_rating <-as.factor(edx$date_rating)

## validation
validation <- validation %>% 
select(userId, movieId, rating, title, genres, date_rating, year_rating, year_release)
validation$year_release <- as.factor(validation$year_release)
validation$year_rating <- as.factor(validation$year_rating)
validation$genres <-as.factor(validation$genres)
validation$userId <-as.factor(validation$userId)
validation$movieId <-as.factor(validation$movieId)
validation$date_rating <-as.factor(validation$date_rating)

#' 
## ---- echo=FALSE, include=TRUE----------------------------------------------
# Result of the formatted dataset
kable(head(edx), format = "latex") %>% kable_styling(latex_options = c("scale_down", "HOLD_position"))  

#' \newpage
#' 
#' ## Visualization
#' 
#' 
#' Next, providing a visualization of the formatted training dataset is warranted.  
#' 
#' 
#' 
#' ### Ratings: count by rating score
#' 
#' 
#' 
#' 
## ----Plot - Ratings: counts by variable, echo=FALSE, include=TRUE, fig.width = 12, fig.height = 8----
ggplot(data.frame(edx$rating), aes(x=edx$rating)) +
  geom_bar()+
  theme_light()+
  labs(title = "Rating count by rating score")+
  labs(x="rating score", y="count")+
  xlim(c(-0.5,5.5))

#' 
#' 
#' 
#' Visualization of the rating count by rating score revealed that negative ratings < 3 are less common than ratings \ge 3 and above.  
#' 
#' 
#' \newpage
#' 
#' ### Ratings: count by genre  
## ---- echo=FALSE, include=TRUE, fig.width = 12, fig.height = 8--------------
ggplot(data.frame(edx$genres), aes(x=edx$genres)) +
  geom_bar()+
  theme_light()+
  labs(title = "Rating count by genre")+
  labs(x="genre", y="count")+
  theme(axis.text.x = element_text(angle = 90)) 

#' 
#' Visualization of the rating count by genre revealed that some genres may be more prone to ratings than others.
#' 
#' \newpage
#' 
#' 
#' ### Ratings: count by year of rating
## ----Plot - Rating count by year of rating, echo=FALSE, include=TRUE, fig.width = 12, fig.height = 8----
ggplot(data.frame(edx$year_rating), aes(x=edx$year_rating)) +
  geom_bar()+
  theme_light()+
  labs(title = "Rating count by year of rating")+
  labs(x="year", y="count")+
  theme(axis.text.x = element_text(angle = 90)) 

#' 
#' Visualization of the rating count by year of the rating revealed that some years are associated with more ratings than others.
#' 
#' \newpage
#' 
#' 
#' 
#' ### Ratings: count by year of movie release
## ----Plot - Rating count by year of release, echo=FALSE, include=TRUE, fig.width = 12, fig.height = 8----
ggplot(data.frame(edx$year_release), aes(x=edx$year_release)) +
  geom_bar()+
  theme_light()+
  labs(title = "Rating count by year of release")+
  labs(x="year", y="count")+
  theme(axis.text.x = element_text(angle = 90)) 

#' 
#' Visualization of the rating count by year of the movie release revealed that younger movies are associated with more ratings than older movies.
#'   
#' \newpage
#' 
#' ## Models: building & evaluation
#' 
#' First, a partition of the edx set for training/testing purposes needs to be created.  
## ----partition of the edx set for training/testing, echo=TRUE, warning=FALSE, include=TRUE----
# Creating data partition of edx for cross-validation
# Validation set will be 10% of edx data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_edx <- edx[-test_index,]
temp <- edx[test_index,]
# Make sure userId and movieId in validation set are also in edx set
test_edx <- temp %>% 
  semi_join(train_edx, by = "movieId") %>%
  semi_join(train_edx, by = "userId")
# Add the Rows removed from the test_edx back into train_edx
removed <- anti_join(temp, test_edx)
train_edx <- rbind(train_edx, removed)
rm(temp, test_index, removed)

#' 
#' 
#' ### Predicting the mean rating
#' 
#' 
#' 
#' Always predicting the mean rating is perhaps the simplest model. In the training dataset, the mean is ~3.5.  
#' 
#' 
#' 
## ---- echo=TRUE, include=TRUE-----------------------------------------------
mean(train_edx$rating)

#' 
## ---- echo=TRUE, include=TRUE-----------------------------------------------
# Mean rating of all movies
mu_hat <- mean(train_edx$rating)
# Predict the RMSE on the test_edx set
mean_model_result <- RMSE(test_edx$rating, mu_hat)
# Gathering RMSE results in a dataframe
results <- data.frame(model="_Mean_ **test**", RMSE=mean_model_result) 
results %>% kable(format = "simple", align = 'c')

#' 
#' An RMSE of >1 underscores a poor model performance.  
#' 
#' ### Considering the movie
#' 
#' The goal is to improve the 'predicting the mean' model by considering a possible movie effect.  
#' 
#' The following equation depicts this approach:
#' 
#' \[\boxed{Y_{u,i} = \hat{\mu} + b_i + \epsilon_{u,i}}\]
#' 
#' In short, $\hat{\mu}$ represents the mean rating and $\varepsilon_{i,u}$ the independent errors. The $b_i$ represents the magnitude of the movie effect $i$.
#' 
## ---- echo=TRUE, include=TRUE-----------------------------------------------
# Mean rating of all movies
mu_hat <- mean(train_edx$rating)
# Calculating the mean by user
movie_avgs <- train_edx %>%
   group_by(movieId) %>%
   summarize(b_i = mean(rating - mu_hat))
# Computing predicted ratings on test_edx dataset
movie_model <- test_edx %>%
   left_join(movie_avgs, by="movieId") %>%
   mutate(pred = mu_hat + b_i) %>%
   pull(pred)
mean_movie_result <- RMSE(movie_model, test_edx$rating)
# Expanding the results dataframe
results <- results %>% add_row(model="_Mean_+_Movie_ **test**", RMSE=mean_movie_result)
results %>% kable(format = "simple", align = 'c')

#' 
#' When considering the movie effect, the RMSE is 0.9662. Given that this value is still above the target RMSE, the performance of this model does not suffice.
#' 
#' ### Considering movie & user
#' 
#' The RMSE is still too high. Hence, further considerations are warranted. Here, a user effect is being accounted for.  
#' 
#' The following equation depicts this more complex approach:
#' 
#' \[\boxed{Y_{u,i} = \hat{\mu} + b_u + b_i + \epsilon_{u,i}}\]
#' 
#' In addition to the movie equation above, $b_u$ represents the magnitude of a potential user effect $u$.  
#' 
#' 
## ---- echo=TRUE, include=TRUE-----------------------------------------------
# Mean rating of all movies
mu_hat <- mean(train_edx$rating)
# Calculating mean by movie
movie_avgs <- train_edx %>%
   group_by(movieId) %>%
   summarize(b_i = mean(rating - mu_hat))
# Calculating mean by user
user_avgs <- train_edx %>%
   left_join(movie_avgs, by="movieId") %>%
   group_by(userId) %>%
   summarize(b_u = mean(rating - mu_hat - b_i))
# Computing predicted ratings on test_edx dataset
mean_user_movie_model <- test_edx %>%
   left_join(movie_avgs, by="movieId") %>%
   left_join(user_avgs, by="userId") %>%
   mutate(pred = mu_hat + b_u + b_i) %>%
   pull(pred)
mean_user_movie_model_result <- RMSE(mean_user_movie_model, test_edx$rating)
# Expanding results dataframe
results <- results %>% add_row(model="_Mean_+_Movie_+_User_ **test**", RMSE=mean_user_movie_model_result)
results %>% kable(format = "simple", align = 'c')

#' 
#' 
#' When considering both user and movie, the RMSE falls close to our target value with 0.8568.
#' 
#' ### Considering user, movie & genre
#' 
#' The following equation represents this model:
#' 
#' \[\boxed{Y_{u,i} = \hat{\mu} + b_i + b_u + b_{g} + \epsilon_{u,i}}\]
#' 
#' In addition to the previous model, the  $b_{u,g}$ represents the magnitude of a given genre effect $u,g$ on a given user.  
#' 
## ----user, movie & genre, echo=TRUE, include=TRUE---------------------------
# Mean rating of all movies
mu_hat <- mean(train_edx$rating)
# Calculating mean by movie
movie_avgs <- train_edx %>%
   group_by(movieId) %>%
   summarize(b_i = mean(rating - mu_hat))
# Calculating mean by user
user_avgs <- train_edx %>%
   left_join(movie_avgs, by="movieId") %>%
   group_by(userId) %>%
   summarize(b_u = mean(rating - mu_hat - b_i))
# Calculating mean by genre
genre_avgs <- train_edx %>%
   left_join(user_avgs, by="userId") %>%
   left_join(movie_avgs, by="movieId") %>%
   group_by(genres) %>%
   summarize(b_g = mean(rating - mu_hat - b_u - b_i))
# Computing predicted ratings on test_edx dataset
mean_user_movie_genre_model <- test_edx %>%
   left_join(user_avgs, by="userId") %>%
   left_join(movie_avgs, by="movieId") %>%
   left_join(genre_avgs, by="genres") %>%
   mutate(pred = mu_hat + b_u + b_i + b_g) %>%
   pull(pred)
mean_user_movie_genre_model_result <- RMSE(mean_user_movie_genre_model, test_edx$rating)
# Expanding results dataset
results <- results %>% add_row(model="_Mean_+_Movie_+_User_+_Genre_ **test**", RMSE=mean_user_movie_genre_model_result)
results %>% kable(format = "simple", align = 'c')


#' 
#' Genre decreases the RMSE. Perhaps including the date of rating into our model may lead to further improvement.  
#' 
#' 
#' 
#' 
#' ### Considering movie, user, genre & date of rating
#' 
#' The following equation represents this model:
#' 
#' \[\boxed{Y_{u,i} = \hat{\mu} + b_i + b_u + b_{g} + b_{dra} + \epsilon_{u,i}}\]
#' 
## ---- echo=TRUE, include=TRUE-----------------------------------------------
# Mean rating of all movies
mu_hat <- mean(train_edx$rating)
# Calculate the mean by movie
movie_avgs <- train_edx %>%
   group_by(movieId) %>%
   summarize(b_i = mean(rating - mu_hat))
# Calculate the mean by user
user_avgs <- train_edx %>%
   left_join(movie_avgs, by="movieId") %>%
   group_by(userId) %>%
   summarize(b_u = mean(rating - mu_hat - b_i))
# Calculating mean by genre
genre_avgs <- train_edx %>%
    left_join(user_avgs, by="userId") %>%
    left_join(movie_avgs, by="movieId") %>%
    group_by(genres) %>%
    summarize(b_g = mean(rating - mu_hat - b_u - b_i))
# Calculating mean by date of rating
date_ra_avgs <- train_edx %>%
    left_join(user_avgs, by="userId") %>%
    left_join(movie_avgs, by="movieId") %>%
    left_join(genre_avgs, by="genres") %>%
   group_by(date_rating) %>%
   summarize(b_d_ra = mean(rating - mu_hat - b_u - b_i - b_g))
# Compute the predicted ratings on test_edx dataset
mean_user_movie_date_ra_model <- test_edx %>%
    left_join(user_avgs, by="userId") %>%
    left_join(movie_avgs, by="movieId") %>%
    left_join(genre_avgs, by="genres") %>%
    left_join(date_ra_avgs, by="date_rating") %>%
    mutate(pred = mu_hat + b_u + b_i + b_d_ra + b_g) %>%
    pull(pred)
mean_user_movie_date_ra_model_result <- RMSE(mean_user_movie_date_ra_model, test_edx$rating)
# Expanding results dataframe
results <- results %>% add_row(model="_Mean_+_Movie_+_User_+_Genre_+_DateRa_ **test**", RMSE=mean_user_movie_date_ra_model_result)
results %>% kable(format = "simple", align = 'c')


#' 
#' Adding the date of the rating to the equation slightly improved (decreased) the RMSE.
#' 
#' 
#' 
#' ### Considering user, movie, genre, date of rating & year of release
#' 
#' The following equation represents this expanded model:
#' 
#' \[\boxed{Y_{u,i} = \hat{\mu} + b_i + b_u + b_{g} + b_{dra} + b_{yre} + \epsilon_{u,i}}\]
#' 
## ---- echo=TRUE, include=TRUE-----------------------------------------------
# Mean rating of all movies
mu_hat <- mean(train_edx$rating)
# Calculating mean by movie
movie_avgs <- train_edx %>%
   group_by(movieId) %>%
   summarize(b_i = mean(rating - mu_hat))
# Calculating mean by user
user_avgs <- train_edx %>%
   left_join(movie_avgs, by="movieId") %>%
   group_by(userId) %>%
   summarize(b_u = mean(rating - mu_hat - b_i))
# Calculating mean by genre
genre_avgs <- train_edx %>%
    left_join(user_avgs, by="userId") %>%
    left_join(movie_avgs, by="movieId") %>%
    group_by(genres) %>%
    summarize(b_g = mean(rating - mu_hat - b_u - b_i))
# Calculating mean by date of rating
date_ra_avgs <- train_edx %>%
    left_join(user_avgs, by="userId") %>%
    left_join(movie_avgs, by="movieId") %>%
    left_join(genre_avgs, by="genres") %>%
   group_by(date_rating) %>%
   summarize(b_d_ra = mean(rating - mu_hat - b_u - b_i - b_g))
# Calculating mean by year of release
year_re_avgs <- train_edx %>%
      left_join(user_avgs, by='userId') %>%
      left_join(movie_avgs, by='movieId') %>%
      left_join(genre_avgs, by='genres') %>%
      left_join(date_ra_avgs, by='date_rating') %>%
      group_by(year_release) %>%
      summarize(b_y_re = mean(rating-b_u-b_i-b_d_ra-b_g-mu_hat))
# Computing predicted ratings on test_edx dataset
mean_user_movie_date_ra_model <- test_edx %>%
      left_join(user_avgs, by="userId") %>%
      left_join(movie_avgs, by="movieId") %>%
      left_join(genre_avgs, by="genres") %>%
      left_join(date_ra_avgs, by="date_rating") %>%
      left_join(year_re_avgs, by="year_release") %>%
   mutate(pred = mu_hat + b_u + b_i + b_d_ra + b_g + b_y_re) %>%
   pull(pred)
mean_user_movie_date_ra_model_result <- RMSE(mean_user_movie_date_ra_model, test_edx$rating)
# Expanding results dataframe
results <- results %>% add_row(model="_Mean_+_Movie_+_User_+_Genre_+_DateRa_+_YearRe_ **test**", RMSE=mean_user_movie_date_ra_model_result)
results %>% kable(format = "simple", align = 'c') 


#' 
#' 
#' Given that the last model in the table above showed the best performance, it will be considered for further optimization.  
#' 
#' ### Regularization
#' 
#' The regularization approach allows us to adjust for variables (e.g., users and movies) with large estimates which were formed from small sample sizes by adding the term $\lambda$ (lambda). For example, in order to optimize $b_u$ and $b_i$, it is necessary to use this equation:
#' 
#' \[\boxed{\frac{1}{N} \sum_{u,i} (y_{u,i} - \mu - b_{i})^{2} + \lambda \sum_{i} b_{i}^2}\]   
#' 
#' which may be reduced to:   
#' 
#' \[\boxed{\hat{b_{i}} (\lambda) = \frac{1}{\lambda + n_{i}} \sum_{u=1}^{n_{i}} (Y_{u,i} - \hat{\mu})}\]   
#' 
#' 
#' 
#' ### Considering user, movie, date of rating, genre & year of release with regularization 
## ---- echo=TRUE, include=TRUE, fig.width = 12, fig.height = 8---------------
mu_hat <- mean(train_edx$rating) # Mean of all movies
lambdas <- seq(3.5, 5.5, 0.1) # Defining table of lambdas
# Cross-validation of lambdas
rmses <- sapply(lambdas, function(lambda) {
    b_i <- train_edx %>% # Including movies
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu_hat) / (n() + lambda))

    b_u <- train_edx %>% # Including users
      left_join(b_i, by='movieId') %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu_hat) / (n() + lambda))
    b_g <- train_edx %>% # Including genres
      left_join(b_u, by='userId') %>%
      left_join(b_i, by='movieId') %>%
      group_by(genres) %>%
      summarize(b_g = sum(rating-b_u-b_i-mu_hat) / (n()+lambda))
    b_d_ra <- train_edx %>% # Including date of rating
      left_join(b_u, by='userId') %>%
      left_join(b_i, by='movieId') %>%
      left_join(b_g, by='genres') %>%
      group_by(date_rating) %>%
      summarize(b_d_ra = sum(rating - b_u -b_i - b_g-mu_hat) / (n() + lambda))
    b_y_re <- train_edx %>% # Including year of release
      left_join(b_u, by='userId') %>%
      left_join(b_i, by='movieId') %>%
      left_join(b_g, by='genres') %>%
      left_join(b_d_ra, by='date_rating') %>%
      group_by(year_release) %>%
      summarize(b_y_re = sum(rating-b_u-b_i-b_d_ra-b_g-mu_hat) / (n()+lambda))
    predicted_ratings <- test_edx %>% # Predicting ratings  
      left_join(b_u, by='userId') %>%
      left_join(b_i, by='movieId') %>%
      left_join(b_d_ra, by="date_rating") %>%
      left_join(b_g, by="genres") %>%
      left_join(b_y_re, by="year_release") %>%
      mutate(pred = mu_hat + b_u + b_i + b_g + b_d_ra  + b_y_re) %>%
      pull(pred)
   return(RMSE(predicted_ratings, test_edx$rating))
})
# Plot: RMSEs versus lambdas
df <- data.frame(RMSE = rmses, lambdas = lambdas)
ggplot(df, aes(lambdas, rmses)) +
  geom_point()+
  theme_light()+
  labs(title = "RMSEs versus Lambdas - Regularized Model")+
  labs(x = "lambdas", y = "RMSEs")
# Identifying lambda value associated with lowest RMSE
lambda_min <- lambdas[which.min(rmses)]
# Predicting RMSE on the test_edx set
reg_model <- min(rmses)
# Expanding results dataframe
results <- results %>% add_row(model="_Mean_+_Movie_+_User_+_Genre_+_DateRa_+_YearRe_+_Reg_ **test**", RMSE=reg_model)
results %>% kable(format = "simple", align = 'c')

#' 
#' > The optimal lambda generated using the training dataset only is:  
## ---- echo=FALSE, include=TRUE----------------------------------------------
lambda_min

#' 
#' ### Final model applied to validation dataset
## ----Final model - validation, echo=TRUE, include=TRUE----------------------
# Mean of all movies
mu_hat <- mean(edx$rating)
# Applying lambda min
rmse <- sapply(lambda_min, function(lambda) {
# Including movies
    b_i <- edx %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu_hat) / (n() + lambda))
# Including users
    b_u <- edx %>%
      left_join(b_i, by='movieId') %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu_hat) / (n() + lambda))
# Including genres
    b_g <- train_edx %>%
      left_join(b_u, by='userId') %>%
      left_join(b_i, by='movieId') %>%
      group_by(genres) %>%
      summarize(b_g = sum(rating-b_u-b_i-mu_hat) / (n()+lambda))
# Including date of rating
    b_d_ra <- train_edx %>%
      left_join(b_u, by='userId') %>%
      left_join(b_i, by='movieId') %>%
      left_join(b_g, by='genres') %>%
      group_by(date_rating) %>%
      summarize(b_d_ra = sum(rating - b_u -b_i - b_g-mu_hat) / (n() + lambda))
# Including year of release
    b_y_re <- train_edx %>%
      left_join(b_u, by='userId') %>%
      left_join(b_i, by='movieId') %>%
      left_join(b_g, by='genres') %>%
      left_join(b_d_ra, by='date_rating') %>%
      group_by(year_release) %>%
      summarize(b_y_re = sum(rating-b_u-b_i-b_g-b_d_ra-mu_hat) / (n()+lambda))
# Predicting ratings
    predicted_ratings <- validation %>%
      left_join(b_u, by='userId') %>%
      left_join(b_i, by='movieId') %>%
      left_join(b_d_ra, by="date_rating") %>%
      left_join(b_g, by="genres") %>%
      left_join(b_y_re, by="year_release") %>%
      mutate(pred = mu_hat + b_u + b_i + b_g + b_d_ra  + b_y_re) %>%
      pull(pred)
    return(RMSE(predicted_ratings, validation$rating))
    predicted_ratings <- predicted_ratings
})
# Expanding results dataframe
results <- results %>% add_row(model="_Mean_+_Movie_+_User_+_Genre_+_DateRa_+_YearRe_+_Reg_ **validation**", RMSE=rmse)
results %>% kable(format = "simple", align = 'c')



#' 
#' 
#' The regularized User+Movie+Date_Rating+Genre+Year_release model offers a prediction with an RMSE below the desired cutoff.  
#'   
#' ***
#'   
#' 
#' # Results
#' 
#' ## Table summary
#' The table below summarizes the results of this project.  
#' 
## ---- echo=FALSE, include=TRUE----------------------------------------------
# Shows the results
results %>% kable(format = "simple", align = 'c')
#' 
#' 
### Generating the predicted ratings (i.e., 'predicted_ratings')
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat) / (n() + lambda_min))
  b_u <- edx %>%
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_hat) / (n() + lambda_min))
  b_g <- train_edx %>%
    left_join(b_u, by='userId') %>%
    left_join(b_i, by='movieId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating-b_u-b_i-mu_hat) / (n()+lambda_min))
  b_d_ra <- train_edx %>%
    left_join(b_u, by='userId') %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_g, by='genres') %>%
    group_by(date_rating) %>%
    summarize(b_d_ra = sum(rating - b_u -b_i - b_g-mu_hat) / (n() + lambda_min))
  b_y_re <- train_edx %>%
    left_join(b_u, by='userId') %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_g, by='genres') %>%
    left_join(b_d_ra, by='date_rating') %>%
    group_by(year_release) %>%
    summarize(b_y_re = sum(rating-b_u-b_i-b_g-b_d_ra-mu_hat) / (n()+lambda_min))
  predicted_ratings <- validation %>%
    left_join(b_u, by='userId') %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_d_ra, by="date_rating") %>%
    left_join(b_g, by="genres") %>%
    left_join(b_y_re, by="year_release") %>%
    mutate(pred = mu_hat + b_u + b_i + b_g + b_d_ra  + b_y_re) %>%
    pull(pred)
#' 
#' ## Final RMSE value
#' 
## ---- echo=TRUE, include=TRUE-----------------------------------------------
rmse


#'   
#' ***
#'   
#' # Conclusion
#' 
#' Including the variables 'movie ID' and 'user ID' is key when optimizing movie rating predictions. Other variables such as 'date of rating', and regularization did not affect the predictions as much. Excluding some of the variables from the final validation model may have led to a similar final result.  
#'   
#' ***  
#'   
#' # References
#' > https://rafalab.github.io/dsbook/  
#' > https://grouplens.org/datasets/movielens/10m/  
#' > http://files.grouplens.org/datasets/movielens/ml-10m.zip  
#' > https://files.grouplens.org/datasets/movielens/ml-10m-README.html  
#'   
#' ***
#'   
#' \newpage 
#'   
#' # Appendix
#'   
#' ## Transforming RMarkdown to RScript
## ---- echo=TRUE, include=TRUE-----------------------------------------------
# Example code
# knitr::purl("grafj_movielens_r_markdown.Rmd", documentation = 2)

#'   
#' ## R Version
## ---- echo=TRUE, include=TRUE-----------------------------------------------
version

#'   
#' ***
