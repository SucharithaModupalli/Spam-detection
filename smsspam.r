#spam detection of text messages by using sentimental analysis.
#libraries used as per sentimental analysis

#text mining package from R community, tm_map(), content_transformer()
library(tm) 

#used for stemming, wordStem(), stemDocument()
library(SnowballC) 

#color palletes required to highlight in wordcloud
library(RColorBrewer)

#used forwordcloud generator
library(wordcloud) 

#used for Naive Bayes classifier
library(e1071) 

#CrossTable() also called as contingency table 
library(gmodels)

#ConfusionMatrix() 
library(caret)

#importing the dataset
sms_raw <- read.csv(file.choose())
View(sms_raw)

#data exploration
head(sms_raw)
str(sms_raw)

#features of target variable
table(sms_raw$v1)
round(prop.table(table(sms_raw$v1)),digits = 2)

#spam vs ham wordcloud(before any cleansing of data)
spam <- subset(sms_raw,v1 == "spam")
wordcloud(spam$v2, max.words = 60, colors = brewer.pal(5,"Dark2"),random.order = FALSE)
spam <- subset(sms_raw,v1 == "ham")
wordcloud(spam$v2,max.words = 60,colors = brewer.pal(5,"Dark2"),random.order = FALSE)

#data preprocessing
#Steps to creating a corpus
#Step 1: Prepare a vector source object using VectorSource
#Step 2: Supply the vector source to VCorpus, to import from sources
sms_corpus <- VCorpus(VectorSource(sms_raw$v2))

#To view a message, must use double bracket and as.character()
lapply(sms_corpus[1:2], as.character)

#corpus cleaning
# converts to lowercase
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
sms_corpus_clean
#remove numbers as numbers are unique
sms_corpus_clean <- tm_map(sms_corpus_clean, content_transformer(removeNumbers))

#removing stop words, i.e, to, or, but, and. Use stopwords() as argument, parameter that indicates what words we don't want
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())

#remove punctuation, i,e "", .., ', `
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

#apply stemming, removing suffixes f(learns, learning, learned) --> learn
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

#lastly, strip addtional whitespaces
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

#data preparation
#convert our corpus to a DTM
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

#dimension of DTM
dim(sms_dtm)

#alternate way to data cleanse all in 1 go
sms_dtm2 <- DocumentTermMatrix(sms_corpus_clean, control = 
                                 list(tolower = TRUE,
                                      removeNumbers = TRUE,
                                      stopwords = TRUE,
                                      removePunctuation = TRUE,
                                      stemming = TRUE))
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE, colors=brewer.pal(8, "Dark2"))


#Training set
sms_dtm_train <- sms_dtm[1:4180]

#Test set
sms_dtm_test <- sms_dtm[4181:5574,]
#Preparing Training and Test Labels

#Training Label
sms_train_labels <- sms_raw[1:4180]$v1

#Test Label
sms_test_labels <- sms_raw[4181:5574, ]$v1
#To ensure the train and test sets are representative, both sets should rougly have the same proportion of spam and ham.

#Proportion for train labels
prop.table(table(sms_train_labels))

#Proportion for test labels
prop.table(table(sms_test_labels))

# finding words that appear at least 5 times
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)

#preview of most frequent words, 1166 terms with at least 5 occurences
str(sms_freq_words)

#filter the DTM sparse matrix to only contain words with at least 5 occurence
#reducing the features in our DTM
sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]


# create a function to do , convert zeros and non-zeros into "Yes" or "No"
convert_counts <- function(x){
  x <- ifelse(x > 0, "Yes", "No")
}

#apply to train and test reduced DTMs, applying to column
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

#check structure of both the DTM matrices
str(sms_train)
str(sms_test)


# applying Naive Bayes to training set
sms_classifier <- naiveBayes(sms_train, sms_train_labels, laplace = 0)

#applying to test set
sms_test_pred <- predict(sms_classifier, sms_test)

#preview of output
head(data.frame("actual" = sms_test_labels, "predicted" = sms_test_pred))


CrossTable(sms_test_pred, sms_test_labels, prop.chisq = FALSE, dnn = c("predicted", "actual"))


confusionMatrix(sms_test_pred, sms_test_labels, dnn = c("predicted", "actual"))


#Parameter Tuning
sms_classifier <- naiveBayes(sms_train, sms_train_labels, laplace = 1)

sms_test_pred <- predict(sms_classifier, sms_test)
CrossTable
CrossTable(sms_test_pred, sms_test_labels, prop.chisq = FALSE, dnn = c("predicted", "actual"))
confusionMatrix
confusionMatrix(sms_test_pred, sms_test_labels, dnn = c("predicted", "actual"))

