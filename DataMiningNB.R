---
  title: "Assigment - Naive Bayes DIY"
author:
  - name author here - Alwin Schra
- name reviewer here - Ilse Akkerman
date: "`r format(Sys.time(), '%d %B, %Y')`"
output:
  html_notebook:
  toc: true
toc_depth: 2
# ---

  ```{r}
#install.packages("tidyverse")
#install.packages("tm")
#install.packages("caret")
#install.packages("wordcloud")
#install.packages("e1071")
#install.packages("readr")
#install.packages("utils")
#install.packages("RColorBrewer")
#install.packages("magrittr")
#install.packages("base")
#install.packages("textclean")
library(tidyverse)
library(tm)
library(caret)
library(wordcloud)
library(e1071)
library(readr)
library(utils)
library(RColorBrewer)
library(magrittr)
library(base)
library(textclean)
 ```
# ---
# Business Understanding
  
  #  Choose a suitable dataset from [this](https://github.com/HAN-M3DM-Data-Mining/assignments/tree/master/datasets) folder and train your own Naive Bayes model. Follow all the steps from the CRISP-DM model.

url <- "https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/assignments/master/datasets/NB-fakenews.csv"
rawDF <- read_csv(url)

head(rawDF)

rawDF$label <- rawDF$label %>% factor %>% relevel("1")
class(rawDF$label)

rawDF$label <- rawDF$label %>% factor %>% relevel("0")
class(rawDF$label) 

Fake <- rawDF %>% filter(label == "1")
NFake <- rawDF %>% filter(label == "0")

wordcloud(Fake$text, max.words = 20, scale = c(4, 0.8), colors= c("indianred1","indianred2","indianred3","indianred"))
wordcloud(NFake$text, max.words = 20, scale = c(4, 0.8), colors= c("lightsteelblue1","lightsteelblue2","lightsteelblue3","lightsteelblue"))

## Data Preparation

rawCorpus <- Corpus(VectorSource(rawDF$text))
inspect(rawCorpus[1:3])

cleanCorpus <- rawCorpus %>% tm_map(tolower) %>% tm_map(removeNumbers)
cleanCorpus <- cleanCorpus %>% tm_map(tolower) %>% tm_map(removeWords, stopwords()) %>% tm_map(removePunctuation)
cleanCorpus <- cleanCorpus %>% tm_map(stripWhitespace)

  # First data check

tibble(Raw = rawCorpus$content[1:3], Clean = cleanCorpus$content[1:3])

cleanDTM <- cleanCorpus %>% DocumentTermMatrix
inspect(cleanDTM)

# Create split indices
set.seed(1234)
trainIndex <- createDataPartition(rawDF$label, p = 0.75, 
                                  list = FALSE,
                                  times = 1)
                          
head(trainIndex)

# Apply split indices to DF
trainDF <- rawDF[trainIndex, ]
testDF <- rawDF[-trainIndex, ]

# Apply split indices to Corpus
trainCorpus <- cleanCorpus[trainIndex]
testCorpus <- cleanCorpus[-trainIndex]

# Apply split indices to DTM
trainDTM <- cleanDTM[trainIndex, ]
testDTM <- cleanDTM[-trainIndex, ]
  
freqWords <- trainDTM %>% findFreqTerms(5)
trainDTM <-  DocumentTermMatrix(trainCorpus, list(dictionary = freqWords))
testDTM <-  DocumentTermMatrix(testCorpus, list(dictionary = freqWords))  
  
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0) %>% factor(levels = c(0,1), labels = c("No", "Yes"))
}

nColsDTM <- dim(trainDTM)[2]
trainDTM <- apply(trainDTM, MARGIN = 2, convert_counts)
testDTM <- apply(testDTM, MARGIN = 2, convert_counts)

head(trainDTM[,1:10])  
  
nbayesModel <-  naiveBayes(trainDTM, trainDF$text, laplace = 1)

predVec <- predict(nbayesModel, testDTM)
confusionMatrix(predVec, testDF$text, positive = "Fake News", dnn = c("Prediction", "True"))

#reviewer adds suggestions for improving the model