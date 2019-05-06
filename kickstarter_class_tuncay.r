
Sys.time()

library(tidyverse)
library(caret)
library(janitor) # adorn_percentages
library(ggplot2)
library(wordcloud)
library(tidytext)
library(stringr)
library(text2vec)
library(tm)
library(NLP)
library(psych)
library(randomForest)
#library(party)
library(car)
library(InformationValue)
library(heuristica)

kickstarter <- read.csv("ks-projects-201801.csv")

# additional variables that are used are created here

kickstarter <- kickstarter %>% mutate(launch_year = as.numeric(as.character(format(as.Date(launched), "%Y"))))
kickstarter <- kickstarter %>% mutate(launch_month = format(as.Date(launched), "%B"))
kickstarter <- kickstarter %>% mutate(launch_month_numeric = as.numeric(as.character(format(as.Date(launched), "%m"))))
kickstarter <- kickstarter %>% mutate(launch_weekday = format(as.Date(launched), "%A"))
kickstarter <- kickstarter %>% mutate(launch_weekday_numeric = as.numeric(as.factor(kickstarter$launch_weekday)))
kickstarter <- kickstarter %>% mutate(excess_p=ifelse(((usd_pledged_real/usd_goal_real)-1)>0,((usd_pledged_real/usd_goal_real)-1),0))
kickstarter <- kickstarter %>% mutate(totaldays =  as.numeric(as.Date(deadline) - as.Date(launched)))
kickstarter <- kickstarter %>% mutate(category_numeric = as.numeric(as.factor(kickstarter$category)))
kickstarter <- kickstarter %>% mutate(main_category_numeric = as.numeric(as.factor(kickstarter$main_category)))
kickstarter <- kickstarter %>% mutate(state_numeric = as.numeric(as.factor(kickstarter$state)))

head(kickstarter,1)

str(kickstarter)

anyNA(kickstarter)

kickstarter[!complete.cases(kickstarter),]
# looks like missing values are for the usd.pledged column, which i wont use in this assignment
# so i can ignore them

levels(kickstarter$state)
unique(kickstarter$launch_weekday_numeric)
unique(kickstarter$launch_weekday)

colnames(kickstarter)

levels(kickstarter$category)
n_distinct(kickstarter$category)

levels(kickstarter$main_category)
n_distinct(kickstarter$main_category)

levels(kickstarter$currency)
n_distinct(kickstarter$currency)

df1 <- (kickstarter %>% group_by(main_category,f_s=state_numeric==4) %>%
summarize(successful_count = n()))
 
df2 <- (kickstarter %>% group_by(main_category,f_s=state_numeric==2) %>%
summarize(failed_count = n()))

df3 <- (kickstarter %>% group_by(main_category,f_s=state_numeric==1) %>%
summarize(canceled_count = n()))

df4 <- (kickstarter %>% group_by(main_category,f_s=state_numeric==3) %>%
summarize(live_count = n()))

df5 <- (kickstarter %>% group_by(main_category,f_s=state_numeric==5) %>%
summarize(suspended_count = n()))

df6 <- (kickstarter %>% group_by(main_category,f_s=state_numeric==6) %>%
summarize(undefined_count = n())) 


merged <- Reduce(function(x, y) left_join(x, y, by=c("main_category","f_s"), all=TRUE), list(df1, df2, df3, df4, df5, df6))
merged <- merged %>% mutate_if(is.integer, ~replace(., is.na(.), 0))
merged   <- merged[seq(2,nrow(merged),2),]
merged

# sum(merged$successful_count[seq(2,30,2)]) # sums only TRUE values
# sum(merged$successful_count[seq(1,30,2)])
#sum(merged[,3:8])
#sum(merged$successful_count)
# sum(merged$failed_count)+
# sum(merged$canceled_count)+
# sum(merged$live_count)+
# sum(merged$suspended_count)+
# sum(merged$undefined_count)
# dim(kickstarter)
# sum(merged[1,3:8])
# sum(merged[2,3:8])
# sum(kickstarter$fail_success==4)

merged_p <- merged %>% adorn_percentages()
merged_p %>% arrange(desc(successful_count))

merged_p <- merged %>% adorn_percentages()
merged_p %>% arrange(desc(failed_count))

df7 <- (kickstarter %>% group_by(category,f_s=state_numeric==4) %>%
summarize(successful_count = n()))
 
df8 <- (kickstarter %>% group_by(category,f_s=state_numeric==2) %>%
summarize(failed_count = n()))

df9 <- (kickstarter %>% group_by(category,f_s=state_numeric==1) %>%
summarize(canceled_count = n()))

df10 <- (kickstarter %>% group_by(category,f_s=state_numeric==3) %>%
summarize(live_count = n()))

df11 <- (kickstarter %>% group_by(category,f_s=state_numeric==5) %>%
summarize(suspended_count = n()))

df12 <- (kickstarter %>% group_by(category,f_s=state_numeric==6) %>%
summarize(undefined_count = n())) 


merged2 <- Reduce(function(x, y) left_join(x, y, by=c("category","f_s"), all=TRUE), list(df7, df8, df9, df10, df11, df12))
merged2 <- merged2 %>% mutate_if(is.integer, ~replace(., is.na(.), 0))
merged2   <- merged2[seq(2,nrow(merged2),2),]
head(merged2)

merged2_p <- merged2 %>% adorn_percentages()
head(merged2_p %>% arrange(desc(successful_count)),10)

merged2_p <- merged2 %>% adorn_percentages()
head(merged2_p %>% arrange(desc(failed_count)),10)

# countries that do kickstarter projects the most to the least
# united states, great britain, canada, and australia lead the way in crowd funding.
# weirdly, these are all english spoken countries.
(kickstarter %>% group_by(country) %>%
summarize(n=n())) %>% arrange(desc(n)) 

# most preferred currencies
# dollar, euro, and british pound are the most used currencies in crowd funding.
(kickstarter %>% group_by(currency) %>%
summarize(n=n())) %>% arrange(desc(n)) 

by_excess_p <- (kickstarter %>% group_by(main_category,excess_p>0)) %>%
summarize(by_group_excess_p=sum(excess_p/n()))

by_excess_p   <- by_excess_p[seq(2,nrow(by_excess_p),2),]
by_excess_p

(kickstarter %>% group_by(main_category,f_s=state=="successful")) %>%
summarize(avgdays=sum(totaldays/n()))

head((kickstarter %>% group_by(category, f_s=state=="successful")) %>%
summarize(avgdays=sum(totaldays/n())))

describe(kickstarter$backers)

# 1 mad away from mean backers distribution by main_category
# mad = median absolute deviation
backed <- (kickstarter %>% group_by(backers,main_category) %>%
summarize(n=n()))
           
head((backed %>% arrange(desc(backers>(mean(kickstarter$backers)+mad(kickstarter$backers))))),10)


# 1 mad away from mean backers distribution by sub_category
backed <- (kickstarter %>% group_by(backers,category) %>%
summarize(n=n()))
           
head((backed %>% arrange(desc(backers>(mean(kickstarter$backers)+mad(kickstarter$backers))))),10)

############################################################################################
kickstarter %>% group_by(main_category) %>%
	summarize(n = n(), avg = mean(excess_p), se = sd(excess_p)/sqrt(n())) %>%
	filter(n > 1) %>% 
	mutate(reorder(main_category, avg)) %>%
	ggplot(aes(x = main_category, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
	geom_point() +
	geom_errorbar() + 
	theme(axis.text.x = element_text(angle = 90, hjust = 1))
	ggtitle("excess pledge by main category")

by_excess_p2 <- (kickstarter %>% group_by(category,excess_p>0)) %>%
summarize(by_group_excess_p2=sum(excess_p/n()))

by_excess_p2   <- by_excess_p2[seq(2,nrow(by_excess_p2),2),]
head(by_excess_p2 %>% arrange(desc(by_group_excess_p2)))

kickstarter %>% group_by(category) %>%
	summarize(n = n(), avg = mean(excess_p), se = sd(excess_p)/sqrt(n())) %>%
	filter(avg > 2) %>% 
	mutate(reorder(category, avg)) %>%
	ggplot(aes(x = category, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
	geom_point() +
	geom_errorbar() + 
	theme(axis.text.x = element_text(angle = 90, hjust = 0.2))
	ggtitle("excess pledge by sub-category, filtered by average excess 2-fold")

kickstarter %>% 
group_by(main_category) %>%
summarize(n=n(),howold=2019-first(launch_year),mean=mean(excess_p)) %>%
mutate (hmpay=n/howold)%>% #hmray=how many projects a year
ggplot(aes(hmpay, mean)) +
geom_point() +
geom_smooth() +
ggtitle("")

kickstarter %>% group_by(launch_month_numeric) %>%
	summarize(n = n(), avg = mean(excess_p), se = sd(excess_p)/sqrt(n())) %>%
	filter(avg > 2) %>% 
	mutate(reorder(launch_month_numeric, avg)) %>%
	ggplot(aes(x = launch_month_numeric, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
	geom_line() +
	geom_smooth() +
	ggtitle("")


kickstarter %>% group_by(launch_weekday) %>%
	summarize(n = n(), avg = mean(excess_p), se = sd(excess_p)/sqrt(n())) %>%
	filter(avg > 0) %>% 
	mutate(reorder(launch_weekday, avg)) %>%
	ggplot(aes(x = launch_weekday, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
	geom_point() +
	geom_errorbar() + 
	theme(axis.text.x = element_text(angle = 90, hjust = 1))
	ggtitle("")

############################################################################################
names_of_projects <- (kickstarter$name)
names_of_projects <- tolower(names_of_projects)
names_of_projects <- gsub("[^[A-Za-z]"," ", names_of_projects)
names_of_projects <- gsub("\\b\\w{1,2}\\b", "", names_of_projects)
names_of_projects <- gsub("canceled", "", names_of_projects)
names_of_projects <- gsub("project", "", names_of_projects)

names_of_projects <- str_squish(names_of_projects)
head(names_of_projects)

data("stop_words")

head(stop_words)
which(stop_words$lexicon == 'snowball')[1:10]

stop_words_for_removal <- as.vector((stop_words %>% filter(lexicon=="snowball"))[1])
as.character(stop_words_for_removal)

names_of_projects_tm <- VCorpus(VectorSource(names_of_projects))
names_of_projects_tm <- tm_map(names_of_projects_tm,removeWords,stop_words_for_removal$word)
names_of_projects_tm <- tm_map(names_of_projects_tm,stripWhitespace)
names_of_projects_tm <- tm_map(names_of_projects_tm,removePunctuation)
inspect(names_of_projects_tm[[10]])


NGramTokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), GRAMS), paste, collapse = " "),
  use.names = FALSE)
}

GRAMS <- 1
NGramTokenizer(names_of_projects_tm[[1]])

GRAMS <- 2
NGramTokenizer(names_of_projects_tm[[1]])

GRAMS <- 1
names_of_projects_dtm_1 <- DocumentTermMatrix(names_of_projects_tm,control = list(tokenize=NGramTokenizer))
names_of_projects_dtm_1 <- removeSparseTerms(names_of_projects_dtm_1,0.99)

head(names_of_projects_dtm_1$dimnames$Terms)
tail(names_of_projects_dtm_1$dimnames$Terms)

GRAMS <- 2
names_of_projects_dtm_2 <- DocumentTermMatrix(names_of_projects_tm,control = list(tokenize=NGramTokenizer))
names_of_projects_dtm_2 <- removeSparseTerms(names_of_projects_dtm_2,0.997)

head(names_of_projects_dtm_2$dimnames$Terms)
tail(names_of_projects_dtm_2$dimnames$Terms)

names_of_projects_dtm_freq_1 <- colSums(as.matrix(names_of_projects_dtm_1), na.rm = T)
names_of_projects_dtm_freq_1 <- sort(names_of_projects_dtm_freq_1,decreasing = T)
names_of_projects_dtm_freq_1[1:10]
#barplot(inst2_dtm_freq_1[1:20])

names_of_projects_dtm_freq_2 <- colSums(as.matrix(names_of_projects_dtm_2), na.rm = T)
names_of_projects_dtm_freq_2 <- sort(names_of_projects_dtm_freq_2,decreasing = T)
names_of_projects_dtm_freq_2[1:10]
#barplot(inst2_dtm_freq_1[1:20])

colorlist = c("red","blue","green","red","pink","orange","grey","black","brown","navy","magenta","purple")
wordcloud(names(names_of_projects_dtm_freq_1),names_of_projects_dtm_freq_1, random.order = F, random.color = T, colors = colorlist,scale=c(2,.25))
wordcloud(names(names_of_projects_dtm_freq_2),names_of_projects_dtm_freq_2, random.order = F, random.color = T, colors = colorlist,scale=c(2,.25))

bing <- sentiments %>% filter(lexicon == "bing") %>% dplyr::select(word,
sentiment)

head(bing)
sort(table(bing$sentiment))

tokens <- colnames(names_of_projects_dtm_1)
tokenMat <- cbind.data.frame(names_of_projects_dtm_1$i,names_of_projects_dtm_1$v,tokens[names_of_projects_dtm_1$j])
colnames(tokenMat) <- c("id","freq","word")
tokenMat$id <- as.character(tokenMat$id)
tokenMat$freq <- as.numeric(tokenMat$freq)
tokenMat$word <- as.character(tokenMat$word)
head(tokenMat)
dim(tokenMat)
word_count <- tapply(tokenMat$freq,tokenMat$id,sum)
tokenMat$n <- word_count[match(tokenMat$id,names(word_count))]

by_id_sentiment_bing <- inner_join(bing,tokenMat,by="word")
by_id_sentiment_bing.unique <- by_id_sentiment_bing[!duplicated(by_id_sentiment_bing$id),]
dim(by_id_sentiment_bing.unique)
head(by_id_sentiment_bing.unique)

count_sentiment_by_id <- matrix(NA,length(kickstarter$name),length(unique(bing$sentiment)))
colnames(count_sentiment_by_id) <- unique(bing$sentiment)

for(i in 1:nrow(count_sentiment_by_id)){
    stp <- by_id_sentiment_bing.unique[by_id_sentiment_bing.unique$id == i,]
    sub_mat <- matrix(0,1,ncol(count_sentiment_by_id))
    colnames(sub_mat) <- colnames(count_sentiment_by_id)
    
  for(j in 1:nrow(stp)){
    sub_mat[1,(stp$sentiment[j-i])] <- sub_mat[1,(stp$sentiment[j-i])] + stp$freq[j]
  }
    
  count_sentiment_by_id[i,] <- sub_mat[1,]
}
count_sentiment_by_id<- as.data.frame(count_sentiment_by_id)
head(count_sentiment_by_id)
length(count_sentiment_by_id$negative)
c(negative=sum(count_sentiment_by_id$negative),positive=sum(count_sentiment_by_id$positive))

n <- length(kickstarter$name)

sentiment_hat <- rbinom(n, 1, 0.70) # 1 for positive, 0 for negative

sentiment_hat <- as.data.frame(sentiment_hat)
head(sentiment_hat)

dim(sentiment_hat)
sum(sentiment_hat$sentiment_hat)

classification_bing <- ifelse(count_sentiment_by_id$positive > count_sentiment_by_id$negative, 1, 0)
confusion_matrix_bing <- table(sentiment_hat$sentiment_hat,classification_bing)
confusion_matrix_bing

##Overall Accuracy of Bing Lexicon vs sentiment_hat
accuracy <- sum(diag(confusion_matrix_bing))/sum(confusion_matrix_bing)
print('--Overall Accuracy--')
accuracy
############################################################################################
kickstarter2 <- (kickstarter[,c("state_numeric","category_numeric","main_category_numeric",
                                "backers","totaldays","launch_month_numeric","launch_weekday_numeric")])

# for logistic regression, make it fail=0, success=1
# there is 6 states, 4th is success, convert 4 to 1,
# convert all the other numbers into 0.

kickstarter2$state_numeric[kickstarter2$state_numeric==1] <- 0
kickstarter2$state_numeric[kickstarter2$state_numeric==2] <- 0
kickstarter2$state_numeric[kickstarter2$state_numeric==3] <- 0
kickstarter2$state_numeric[kickstarter2$state_numeric==4] <- 1
kickstarter2$state_numeric[kickstarter2$state_numeric==5] <- 0
kickstarter2$state_numeric[kickstarter2$state_numeric==6] <- 0

head(kickstarter2,1)

str(kickstarter2)

set.seed(1)
test_index <- createDataPartition(y = kickstarter2$state_numeric, times = 1, p = 0.1, list = FALSE)
train <- kickstarter2[-test_index,]
test <- kickstarter2[test_index,]
############################################################################################
# LOGISTIC REGRESSION

# http://r-statistics.co/Logistic-Regression-With-R.html

logitpredtrain <- glm(state_numeric~category_numeric+backers+totaldays+launch_month_numeric+launch_weekday_numeric,
                      train, family=binomial(link="logit"))

predicted <- predict(logitpredtrain, test, type="response")

summary(logitpredtrain)

AIC(logitpredtrain)
BIC(logitpredtrain)

# check for multicollinearity
vif(logitpredtrain)

optCutOff <- optimalCutoff(test$state_numeric, predicted)[1] 
misClassError(test$state_numeric, predicted, threshold = optCutOff)
# The lower the misclassification error, the better is your model

plotROC(test$state_numeric, predicted)
#Receiver Operating Characteristics Curve traces the percentage of true positives accurately 
#predicted by a given logit model as the prediction probability cutoff is lowered from 1 to 0. 
#For a good model, as the cutoff is lowered, 
#it should mark more of actual 1’s as positives and lesser of actual 0’s as 1’s. 
#So for a good model, the curve should rise steeply, 
#indicating that the TPR (Y-Axis) increases faster than the FPR (X-Axis) as the cutoff score decreases. 
#Greater the area under the ROC curve, better the predictive ability of the model.


Concordance(test$state_numeric, predicted)
# the higher the concordance, the better is the quality of model

sensitivity(test$state_numeric, predicted, threshold = optCutOff)
# Sensitivity (or True Positive Rate) is the percentage of 1’s (actuals) correctly predicted by the model
specificity(test$state_numeric, predicted, threshold = optCutOff)
# specificity is the percentage of 0’s (actuals) correctly predicted
# Specificity can also be calculated as 1 − False Positive Rate.

confusMat <- confusionMatrix(test$state_numeric, predicted, threshold = optCutOff)
confusMat

accuracy <- sum(diag(as.matrix(confusMat)))/sum(confusMat)
print('--Overall Accuracy--')
accuracy
############################################################################################
# RANDOMFOREST

 pred_randomforest <- randomForest(state_numeric~category_numeric+backers+totaldays+launch_month_numeric+launch_weekday_numeric,
                      data=train, 
                      importance=TRUE, 
                      ntree=5)

predicted_randomforest <- predict(pred_randomforest, test)

confusMat_rs <- confusionMatrix(test$state_numeric, predicted_randomforest)
confusMat_rs

accuracy <- sum(diag(as.matrix(confusMat_rs)))/sum(confusMat_rs)
print('--Overall Accuracy--')
accuracy


