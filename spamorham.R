# Install all required packages.

install.packages(c("ggplot2", "e1071", "caret", "quanteda", 
                   "irlba", "randomForest", "doSNOW"))

# Set up working directory and Load up the .CSV data and explore in RStudio.

spam.raw <- read.csv("spam.csv", stringsAsFactors = FALSE)
View(spam.raw)

# Clean up the data frame and view

spam.raw <- spam.raw[, 1:2]
names(spam.raw) <- c("Label", "Text")
View(spam.raw)


# Check data to see if there are missing values.
length(which(!complete.cases(spam.raw)))
summary(spam.raw)

# Convert our class label into a factor(categorical).
spam.raw$Label <- as.factor(spam.raw$Label)

#explore the data
table(spam.raw$Label) #no of elements in each category
prop.table(table(spam.raw$Label)) # % of elements in each category

#creating new variable for text length
spam.raw$TextLength = nchar(spam.raw$Text)
summary(spam.raw$TextLength)
View(spam.raw)

# Visualize distribution
boxplot(spam.raw$TextLength)
hist(spam.raw$TextLength)
library(ggplot2)

ggplot(spam.raw, aes(x = TextLength, fill = Label)) +
  theme_bw() +
  geom_histogram(binwidth = 5) +
  labs(y = "Text Count", x = "Length of Text",
       title = "Distribution of Text Lengths with Class Labels")

# split our data into a training set and a test set (sometimes even validation set)
# we'll use caret package(clasification and regression training) for a random stratified split.
library(caret)
help(package = "caret")

#create a 70%/30% stratified split
set.seed(32984) #where to start set
indexes <- createDataPartition(spam.raw$Label, times = 1, p = 0.7, list = FALSE) # 1 split, 70 %

train = spam.raw[indexes,]
test = spam.raw[-indexes,]

#verifying % after split
prop.table(table(train$Label))
prop.table(table(test$Label))

# Text analytics requires a lot of data exploration, data pre-processing and data wrangling. 
#EXCEPTION Example
# HTML-escaped ampersand character.
# HTML-escaped '<' and '>' characters
# A URL.

#The quanteda package IS useful for quickly and easily working with text data.

library(quanteda)
help(package = "quanteda")

#data pre - preocessing pipeline

#Tokenize Text message
train.tokens <- tokens(train$Text, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)

#converting all text to lower case
train.tokens = tokens_tolower(train.tokens)

#removing Stopwords
train.tokens = tokens_select(train.tokens, stopwords(), selection = "remove")

#stemming on tokens(colapsing similar tokens into one term)
train.tokens = tokens_wordstem(train.tokens, language = "english")

#Creating bags of word model
train.tokens.dfm = dfm(train.tokens, tolower= FALSE)

#Transforming data to Matrix or data frame
#matrix
train.tokens.matrix = as.matrix(train.tokens.dfm)
View(train.tokens.matrix[1:20, 1:100])
dim(train.tokens.matrix)

#dataframe
#train.tokens.dataframe = as.data.frame(train.tokens.dfm)
#View(train.tokens.dataframe[1:20, 1:100])

#adding column label to bag of word model for which we are going to predict the lable from the features or variables
train.tokens.df = cbind(Label=train$Label, as.data.frame(train.tokens.dfm))

#Cross Validation
#Aditional preprocessing for such cases
names(train.tokens.df)[c(146, 148, 235, 238)]

#cleaning up column names
names(train.tokens.df) = make.names(names(train.tokens.df))

#Random Stratified Samples
#create stratified folds for 10-fold cross validation repeated 3 times (i.e., create 30 random stratified samples)
set.seed(48743)
cv.folds = createMultiFolds(train$Label, k = 10, times = 3)
cv.cntrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, index = cv.folds)

#run n' train in parallel
library(doSNOW)

#calculating execution time
start.time = Sys.time()

#create clusers to work on diffrent cores
cl = makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

#As the data is non-trivial in size  use a single decision tree alogrithm
rpart.cv.1 = train(Label ~ ., data = train.tokens.df, method = "rpart", trControl = cv.cntrl, tuneLength = 7)

# Processing is done
stopCluster(cl)

# Total time of execution
total.time <- Sys.time() - start.time
total.time

#Check Results
rpart.cv.1

#using Term Frequency(TF) and Inverse Document Frequency(IDF) to improve model prediction 
#By checking how important is the term(if it repeats a lot it is not that important for prediction)

#calculating relative Term Frequency (row/document centric) (normalizes)
term.frequency = function(row)  
  { row / sum(row) }    #every cell in row and divide by total of row which gives percentage of each terms in a doc

#calculating Inverse Document Frequency (column/corpus centric) (penalizes)
inverse.doc.freq = function(col)
{ corpus.size = length(col)         # calculate for each number of columns how many documents are there
doc.count = length(which(col > 0))  #to get number of rows where column is not 0 
log10(corpus.size / doc.count)     #IDF of(term) = no ofdocuments in column divided by count of term in which it shows.  
}

#cauculating TF-IDF
tf.idf = function(tf, idf) { tf * idf }

#normalize using TF (transpose - changeing rows to columns)
train.tokens.df = apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)
dim(train.tokens.matrix)
View(train.tokens.df[1:20, 1:100])

#calculate IDF vector fortraining and test data
train.tokens.idf = apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)

#calculating TF-IDF
train.tokens.tfidf = apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])

#transpose matrix back to original form
train.tokens.tfidf = t(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])

#check for incomplete cases as after preprocessing most would be empty strings.
incomplete.cases = which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases]

#replacing incomplete cases by zeros as it wouldent affect the model
train.tokens.tfidf[incomplete.cases,] = rep(0.0, ncol(train.tokens.tfidf))
dim(train.tokens.tfidf)
sum(which(!complete.cases(train.tokens.tfidf)))

#making a new clean data frame for the IF IDF
train.tokens.tfidf.df = cbind(Label = train$Label, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) = make.names(train.tokens.tfidf.df)

#calculating execution time
start.time = Sys.time()

#create clusers to work on diffrent cores
cl = makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

#As the data is non-trivial in size  use a single decision tree alogrithm
rpart.cv.2 <- train(Label ~ ., data = train.tokens.tfidf.df, method = "rpart", 
                    trControl = cv.cntrl, tuneLength = 7)
# Processing is done
stopCluster(cl)

# Total time of execution
total.time <- Sys.time() - start.time
total.time

#Check Results
rpart.cv.2