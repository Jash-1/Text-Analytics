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
#rpart.cv.1 = train(Label ~ ., data = train.tokens.df, method = "rpart", trControl = cv.cntrl, tuneLength = 7)
rpart.cv.1 = train(Label ~ ., data = train.tokens.df, method = "rpart", trControl = cv.cntrl, tuneLength = 7)
# Processing is done
stopCluster(cl)

# Total time of execution
total.time <- Sys.time() - start.time
total.time

#Check Results
rpart.cv.1
