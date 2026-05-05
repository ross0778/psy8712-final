# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(jsonlite)
library(httr)
library(qdap)
library(tm)
library(stm)
library(textstem)
library(parallel)
library(doParallel)
library(caret)
library(glmnet)
library(ranger)
library(xgboost)

# Data Import and Cleaning
# Embeddings

#used read_csv from tidyverse because it's more efficient than read.csv.
glassdoor_df <- read_csv("../data/glassdoor_reviews.csv")

df <- glassdoor_df %>% #this is to keep all the columns that have any review text as source material as required by the instructions, as well as the outcome variable overall_rating. Only headline, pros, and cons had text content, so I only kept these columns for source material.
  select(overall_rating, headline, pros, cons) %>% 
  #filter(!is.na(overall_rating)) #had this originally to get rid of any NAs in overall_rating but there were none, so it's unnecessary
  mutate(full_text_review = str_c( #used mutate to add a new column called full_text_review and str_c to combine the three text fields (headline, pros, cons) together
    replace_na(headline, ""), #used replace_na(column_name, "") for all of these to convert any NAs in the original columns into a blank string
    replace_na(pros, ""),
    replace_na(cons, ""),
    sep = " " #this adds a single space between the combined three texts in the single column
  ),
  id = row_number() #this makes a new column called id filled with the respective row number, included for later work
  )
#df$full_text_review[1] #check that the combined column looks good by extracting and displaying the first element - looks good
  
set.seed(42) #set the seed for reproducibility
sample_df <- df %>% #because the glassdoor dataset is so big (838,566 rows), we need to make a smaller sample from this dataset so it can run, while still taking as much as we can from the dataset
  group_by(overall_rating) %>% #this groups by overall rating so I can make sure we have a proportional representation of each rating in this sample dataset as to the original dataset
  slice_sample(prop = 10000 / nrow(df)) %>% #slice_sample draws a random sample from each group and prop = 10000/nrow(df) finds the proportion that's needed to get about 10,000 rows, which I believe should be able to run but still takes a good amount of data
  ungroup() #ungroups the data

nrow(sample_df) #this is a check to make sure the rows and proportions look correct and align with the original dataset, which they do
sample_df %>% 
  count(overall_rating)

get_embeddings <- function(text) { #using a function called get_embeddings so I don't have to unnecessarily repeat the same API call
  response <- POST( #sending the POST request
    url = "http://localhost:11434/api/embed", #defines the endpoint, using /api/embed which is a specialized endpoint that makes numerical embeddings for text
    content_type_json(), #tells the Ollama API that the data in the request body is in JSON format, which is required when making an Ollama API request
    body = list( #because the Ollama API expects only a single object per request, this is so that the request reiterates for each object in the list
      model = "nomic-embed-text", #using the nomic-embed-text model as specifies in instructions
      input = text #this represents the text that should be converted into an embedding
    ),
    encode = "json" #enforces JSON format
  )
  result <- httr::content(response, as = "parsed") #this returns an R list from the JSON response by default
  unlist(result$embeddings[[1]]) #using unlist() here takes the first element of the embeddings list and flattens it out into a single numeric vector
}
#length(get_embeddings(sample_df$full_text_review[1])) #used length() here to check that the code did actually return a single numeric vector. This test returned 768 so we know it worked, because we are using 768 dimensions since 768 is the length of the numerical vector produced by embedding models

nofail_embedding <- possibly(get_embeddings, otherwise = rep(NA_real_, 768)) #possibly() wraps get_embedding() so that if an API call fails on any row, it'll handle the error by returning NA instead of stopping the code
embeddings_list <- map(sample_df$full_text_review, nofail_embedding) #map() applies nofail_embedding to each row's full_text_review and returns the result as a new list called embeddings_list
embeddings_matrix <- do.call(rbind, embeddings_list) #used do.call instead of bind_rows() because bind_rows() returned an error that let me know the issue was most likely that embeddings_list is a list of numeric vectors and not a dataframe or tibble. Therefore, I used do.call(), which passes arguments as a single list object, to create a matrix
colnames(embeddings_matrix) <- paste0("emb_", seq_len(ncol(embeddings_matrix))) #this uses paste0 to name the columns in the matrix emb_1 through emb_768 for readability, seq_len() makes a sequence of 1 through the number of columns, which will be 768

embeddings_tbl <- as_tibble(embeddings_matrix) %>% #convert embeddings_matrix into a tibble so we have a tidy data structure
  bind_cols(id = sample_df$id) %>% #bind_cols() adds the identifier column
  filter(if_all(everything(), ~ !is.na(.))) #drop any rows where the embedding API failed and returned NAs, need to do here before joining so all downstream datasets are clean

# Topic Modeling
corpus <- VCorpus(VectorSource(sample_df$full_text_review)) #makes a volatile corpus from full_text_review from the sample dataset


corpus <- corpus %>%
  tm_map(content_transformer(replace_contraction)) %>% #this replaces the contractions
  tm_map(content_transformer(str_to_lower)) %>% #this converts the text to lowercase to standardize the tokens
  tm_map(removePunctuation) %>% #removes punctuation
  tm_map(removeNumbers) %>% #removes numbers
  tm_map(content_transformer(lemmatize_strings)) %>% #reduces words to their basic dictionary form
  tm_map(removeWords, #removes stopwords
         c(stopwords("en"),
           "company", "companies", "glassdoor", "job", "work", "employee", "employees")) %>% 
  tm_map(stripWhitespace) #removes whitespace

ids <- meta(corpus, "id") #this extracts the ids from the corpus, also helped fix the ids issue I was having
dtm <- DocumentTermMatrix(corpus) #converts the corpus into a dtm 
slim_dtm <- removeSparseTerms(dtm, 0.9997) #removeSparseTerms gets rid of sparsely occurring terms. I chose this sparsity in order to retain between a 2:1 and 3:1 N/k ratio in the slim dtm.
N <- nrow(sample_df) #this code down to N/k is how I checked the 2:1 and 3:1 N/k ratio. I get 2.50 when I run this so I am within the ratio
k <- ncol(slim_dtm)
N/k # 2.50

row_totals <- slam::row_sums(slim_dtm) #this finds the total sum of values for each row in the slim_dtm
keep_row <- row_totals > 0 #this makes a logical vector, is true for documents that have above 0 tokens
slim_dtm <- slim_dtm[keep_row, ] #this filters the slim_dtm to keep only non-empty documents
ids <- as.integer(rownames(slim_dtm)) #recovers original doc ids
dtm_tbl <- slim_dtm %>% 
  as.matrix() %>% 
  as_tibble() %>% #this code converts the slim_dtm to a tibble for later use
  mutate(id = as.integer(rownames(slim_dtm))) %>% #converts the rownames of slim_dtm to integers
  mutate(id = as.integer(id)) #makes sure the id column is an integer, helped to fix the ids problem I was having

dtm_lda <- readCorpus(slim_dtm, type = "slam") #readCorpus() converts the slim_dtm to the required format for topic modeling

num_cores <- detectCores() - 1 #detects and specifies the number of cores to use for parallelization
registerDoParallel(num_cores) #this starts the parallelization process which I'm using as recommended since this process is CPU intensive
kresult <- searchK( #searchK finds the optimal number of topics by fitting models across a range of K values to find the best number of topics
  dtm_lda$documents, #this represents the corpus as a list where each element corresponds to a document
  dtm_lda$vocab, #this is a character vector that contains all the unique words in the corpus. These two lines (the document and vocab lines) allow for the bag of words approach
  K = seq(2, 20, by = 2) #this sequence taken from slide 25 of week 12 lecture slides. Tells searchK to start at 2 topics, end at 20 topics, and increase by 2 for each model
)
stopImplicitCluster() #this shuts down the parallel processes
par(mar = c(2, 2, 1, 1)) #had to adjust margins to fix the error, the figure margins were too large
plot(kresult) #this plots the results of searchK. These plots will help us decide how many topics to use in the model. Based on these plots, the "elbow" appears to be consistently at 4, so I will use 4 topics in the model.

topic_model <- stm(dtm_lda$documents, dtm_lda$vocab, K = 4) #this fits the final model

#ran these lines from week 12 lecture slide 26 to explore the lda
labelTopics(topic_model, n = 10) #this gives the most representative words for each topic
findThoughts(topic_model, texts = sample_df$full_text_review[ids], n = 3) #gives the three most representative thoughts for each topic
plot(topic_model, type = "summary", n = 5) #this plots the topic models
topicCorr(topic_model) #this gives the correlations between the topic models
plot(topicCorr(topic_model)) #this just plots the correlations
theta <- topic_model$theta #this gives the probabilities

topics_tbl <- as_tibble(theta) %>% #as_tibble() changes the theta matrix to a tibble, keeping all of the topic probability columns instead of collapsing them, which we want since we'll be using the topics for machine learning
  rename_with(~ paste0("topic_", .)) %>% #rename_with() names the columns topics 1-4 just so they're easier to read using paste0
  mutate(id = ids) #this line attaches the original ids which we'll need for joining

#making the tokens only dataset
token_df <- dtm_tbl %>%
  inner_join(sample_df %>% 
               select(id, overall_rating), by = "id") %>% #chose to use a join here so that it matches on id rather than position like it would with indexing
  select(-id)

#making the topics only dataset
topic_df <- topics_tbl %>%
  inner_join(sample_df %>% #uses inner_join to join the datasets so it only keeps rows where id exists in both
               select(id, overall_rating), by = "id") %>% #selects id and overall_rating
  select(-id) #removes the id column

#making the embeddings only dataset
embedding_df <- embeddings_tbl %>% 
  inner_join(sample_df %>% #uses inner_join to join the datasets so it only keeps rows where id exists in both
               select(id, overall_rating), by = "id") %>%  #selects id and overall_rating
  select(-id) #removes the id column

#embeddings + topics
embedding_topic_df <- embeddings_tbl %>% 
  inner_join(topics_tbl, by = "id") %>% #uses inner_join to join both topics_tbl and sample_df
  inner_join(sample_df %>% 
               select(id, overall_rating), by = "id") %>% #selects id and overall_rating
  select(-id) #removes the id column

# Machine Learning Models

set.seed(42)
holdout_indices_token <- createDataPartition(token_df$overall_rating, p = 0.8, list = FALSE)
holdout_indices_embed <- createDataPartition(embedding_df$overall_rating, p = 0.8, list = FALSE)
holdout_indices_topic <- createDataPartition(topic_df$overall_rating, p = 0.8, list = FALSE)
holdout_indices_embed_topic <- createDataPartition(embedding_topic_df$overall_rating, p = 0.8, list = FALSE)

training_token <- token_df[holdout_indices_token, ]
holdout_token <- token_df[-holdout_indices_token, ]

training_embed <- embedding_df[holdout_indices_embed, ]
holdout_embed <- embedding_df[-holdout_indices_embed, ]

training_topic <- topic_df[holdout_indices_topic, ]
holdout_topic <- topic_df[-holdout_indices_topic, ]

training_embed_topic <- embedding_topic_df[holdout_indices_embed_topic, ]
holdout_embed_topic <- embedding_topic_df[-holdout_indices_embed_topic, ]

#saveRDS(token_df, "../out/data.RDS")

# Elastic Net Models
model1 <- train(
  overall_rating ~ .,
  training_token,
  na.action = na.pass,
  method = "glmnet",
  preProcess = c("nzv", "medianImpute", "center", "scale"),
  tuneGrid = expand.grid(
    alpha = c(0, 1),
    lambda = seq(0.0001, 0.1, length = 10)
  ),
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
)
model1

model2 <- train(
  overall_rating ~ .,
  training_embed,
  na.action = na.pass,
  method = "glmnet",
  preProcess = c("nzv", "medianImpute", "center", "scale"),
  tuneGrid = expand.grid(
    alpha = c(0, 1),
    lambda = seq(0.0001, 0.1, length = 10)
  ),
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
)
model2

model3 <- train(
  overall_rating ~ .,
  training_topic,
  na.action = na.pass,
  method = "glmnet",
  preProcess = c("nzv", "medianImpute","center", "scale"),
  tuneGrid = expand.grid(
    alpha = c(0, 1),
    lambda = seq(0.0001, 0.1, length = 10)
  ),
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
)
model3

model4 <- train(
  overall_rating ~ .,
  training_embed_topic,
  na.action = na.pass,
  method = "glmnet",
  preProcess = c("nzv", "medianImpute","center", "scale"),
  tuneGrid = expand.grid(
    alpha = c(0, 1),
    lambda = seq(0.0001, 0.1, length = 10)
  ),
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
)
model4

# Random Forest Models
model5 <- train(
  overall_rating ~ .,
  training_token,
  na.action = na.pass,
  method = "ranger",
  tuneLength = 3,
  preProcess = c("nzv", "medianImpute","center", "scale"),
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
)
model5

model6 <- train(
  overall_rating ~ .,
  training_embed,
  na.action = na.pass,
  method = "ranger",
  tuneLength = 3,
  preProcess = c("nzv", "medianImpute","center", "scale"),
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
)
model6

model7 <- train(
  overall_rating ~ .,
  training_topic,
  na.action = na.pass,
  method = "ranger",
  tuneLength = 3,
  preProcess = c("nzv", "medianImpute","center", "scale"),
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
)
model7

model8 <- train(
  overall_rating ~ .,
  training_embed_topic,
  na.action = na.pass,
  method = "ranger",
  tuneLength = 3,
  preProcess = c("nzv", "medianImpute","center", "scale"),
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
)
model8

model9 <- train(
  overall_rating ~ .,
  training_token,
  na.action = na.pass,
  method = "lm",
  preProcess = c("nzv", "medianImpute", "center", "scale"),
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
)
model9

model10 <- train(
  overall_rating ~ .,
  training_embed,
  na.action = na.pass,
  method = "lm",
  preProcess = c("nzv", "medianImpute", "center", "scale"),
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
)
model10

model11 <- train(
  overall_rating ~ .,
  training_topic,
  na.action = na.pass,
  method = "lm",
  preProcess = c("nzv", "medianImpute", "center", "scale"),
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
)
model11

model12 <- train(
  overall_rating ~ .,
  training_embed_topic,
  na.action = na.pass,
  method = "lm",
  preProcess = c("nzv", "medianImpute", "center", "scale"),
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
)
model12

pred1 <- predict(model1, newdata = holdout_token)
pred2 <- predict(model2, newdata = holdout_embed)
pred3 <- predict(model3, newdata = holdout_topic)
pred4 <- predict(model4, newdata = holdout_embed_topic)
pred5 <- predict(model5, newdata = holdout_token)
pred6 <- predict(model6, newdata = holdout_embed)
pred7 <- predict(model7, newdata = holdout_topic)
pred8 <- predict(model8, newdata = holdout_embed_topic)
pred9 <- predict(model9, newdata = holdout_token)
pred10 <- predict(model10, newdata = holdout_embed)
pred11 <- predict(model11, newdata = holdout_topic)
pred12 <- predict(model12, newdata = holdout_embed_topic)

table_tbl <- tibble(
  algo = c(model1$method, model2$method, model3$method, model4$method, model5$method, model6$method, model7$method, model8$method, model9$method, model10$method, model11$method, model12$method),
  model = rep(c("Tokens", "Embeddings", "Topics", "Embeddings and Topics"), times = 3),
  cv_rsq = c(max(model1$results$Rsquared, na.rm = TRUE),
             max(model2$results$Rsquared, na.rm = TRUE),
             max(model3$results$Rsquared, na.rm = TRUE),
             max(model4$results$Rsquared, na.rm = TRUE),
             max(model5$results$Rsquared, na.rm = TRUE),
             max(model6$results$Rsquared, na.rm = TRUE),
             max(model7$results$Rsquared, na.rm = TRUE),
             max(model8$results$Rsquared, na.rm = TRUE),
             max(model9$results$Rsquared, na.rm = TRUE),
             max(model10$results$Rsquared, na.rm = TRUE),
             max(model11$results$Rsquared, na.rm = TRUE),
             max(model12$results$Rsquared, na.rm = TRUE)
             ),
  ho_rsq = c(
    postResample(pred1, holdout_token$overall_rating)["Rsquared"],
    postResample(pred2, holdout_embed$overall_rating)["Rsquared"],
    postResample(pred3, holdout_topic$overall_rating)["Rsquared"],
    postResample(pred4, holdout_embed_topic$overall_rating)["Rsquared"],
    postResample(pred5, holdout_token$overall_rating)["Rsquared"],
    postResample(pred6, holdout_embed$overall_rating)["Rsquared"],
    postResample(pred7, holdout_topic$overall_rating)["Rsquared"],
    postResample(pred8, holdout_embed_topic$overall_rating)["Rsquared"],
    postResample(pred9, holdout_token$overall_rating)["Rsquared"],
    postResample(pred10, holdout_embed$overall_rating)["Rsquared"],
    postResample(pred11, holdout_topic$overall_rating)["Rsquared"],
    postResample(pred12, holdout_embed_topic$overall_rating)["Rsquared"]
  )
) %>% 
  mutate(across(c(cv_rsq, ho_rsq), ~ formatC(round(.x, 2), format = "f", digits = 2) %>% 
                  str_remove("^0")))
table_tbl
write_csv(table_tbl, "../out/results.csv")

#RQ1: Does the use of embeddings (using the nomic-embed-text LLM embeddings model) improve prediction of satisfaction beyond a rigorous tokenization strategy?
#Comparing the ho_rsq of tokens and embeddings for each model, we can see that embeddings consistently showed improvement over tokens across all three models.
#Therefore, it does seem as though the use of embeddings improves prediction of satisfaction beyond a rigorous tokenization strategy.
#This implies that the semantic meaning of the words (which is defined by the embeddings) shows improvement over the tokens.
#For example, for the ranger model shows an improvement in ho_rsq from .00 to .29, the OLS model from .01 to .32, and the elastic net model from .01 to .32.

#RQ2: Does the use of topics improve prediction of satisfaction beyond a rigorous tokenization strategy?
#This answer is more mixed. The use of topics shows slight improvement in the ho_rsq of ranger model over the rigorous tokenization strategy. However, this wasn't consistent across all the models.
#For the ho_rsq in the OLS model, the use of topics showed very slight improvement over the tokenization strategy, while for the elastic net model, the tokenization strategy showed very slight or negligble improvement over the use of topics.
#However, overall, tokenization showed very low predictive value.

#RQ3: Does the use of embeddings plus topics improve prediction of satisfaction beyond either alone?
#The use of embeddings + topics improved prediction of satisfaction beyond either alone for all models except for the OLS model.
#However, this isn't entirely surprising that this pattern doesn't hold for the OLS model since OLS often doesn't perform as well in ML.

#RQ4: What is the best prediction of overall job satisfaction achievable using text reviews as source data?
#The best prediction of overall job satisfaction achievable using text reviews as source data was found in the elastic net model using embeddings + topics. This showed a ho_rsq of .54.
  
save.image("../out/final.RData") #this saves the workspace to out/
  
  


