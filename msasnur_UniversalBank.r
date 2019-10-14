#Assignment 1 - Machine Learning#
#----Email- msasnur@kent.edu----#
#-------Date - 10/13/2019-------#

library(readr)

ubank<-read.csv("UniversalBank.csv")
View(ubank)

library(caret)

library(ISLR)

head(ubank,5)

summary(ubank)

library(dplyr)

m_ubank <- ubank[,c(-1,-5)]
m_ubank$Personal.Loan<-factor(m_ubank$Personal.Loan)

#Data Partition - Training Data = 60% and Validation Data = 40% 

set.seed(20)
Train_Index = createDataPartition(m_ubank$Age,p=0.6, list=FALSE)
Train_Data = m_ubank[Train_Index,]
Val_Data = m_ubank[-Train_Index,]

Test_Index = createDataPartition(m_ubank$Age,p=0.2, list=FALSE)
Test_Data = m_ubank[Test_Index,]
Traval_Data = m_ubank[-Test_Index,]
View(m_ubank)
summary(Train_Data)
summary(Val_Data)
summary(Test_Data)

train.norm.df <- Train_Data[, -8]
valid.norm.df <- Val_Data[, -8]
Test.norm.df <- Test_Data[, -8]
traval.norm.df <- Traval_Data[, -8]

norm.values <- preProcess(train.norm.df, method=c("center", "scale"))

train.norm.df <- predict(norm.values, train.norm.df)
valid.norm.df <- predict(norm.values, valid.norm.df)
traval.norm.df <- predict(norm.values, traval.norm.df)
Test.norm.df <- predict(norm.values, Test.norm.df)
View(train.norm.df)

library(FNN)
k_ubank<-knn(train.norm.df,Test.norm.df,cl=Train_Data$Personal.Loan,k=3,prob = TRUE)

# Confusion Matrix

library(gmodels)
CrossTable(x=Test_Data$Personal.Loan,k_ubank,prop.chisq = FALSE)

accuracy.df <- data.frame(k = seq(1, 55, 1), accuracy = rep(0, 55))
for(i in 1:55) {
  knn.pred <- knn(train.norm.df, valid.norm.df, 
                  cl = Train_Data$`Personal.Loan`, k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, Val_Data$`Personal.Loan`)$overall[1] 
}
accuracy.df

accuracy.df[which.max(accuracy.df$accuracy),]

#Accuracy obtained from train and test data (Confusion Matrix) = 97.3 when k = 3
#Accuracy obtained from train and validation data = 96.4

#----------------------------------------------------------------
# Part 4 - Classifying customer using best K value

x <- data.frame("Age" = 40, "Experience" = 10, "Income" = 84, "Family" = 2, "CCAvg" = 2, "Education"= 1, "Mortgage" = 0, "Personal Loan"= "Deny","Securities Account" = 0, "CD Account" = 0, "Online" = 1,"Credit Card" = 1)
y <- data.frame("Age" = 40, "Experience" = 10, "Income" = 84, "Family" = 2, "CCAvg" = 2, "Education"= 2, "Mortgage" = 0, "Personal Loan"= "Accept","Securities Account" = 0, "CD Account" = 0, "Online" = 1,"Credit Card" = 1)
z <- data.frame("Age" = 40, "Experience" = 10, "Income" = 84, "Family" = 2, "CCAvg" = 2, "Education"= 3, "Mortgage" = 0, "Personal Loan"= "Deny","Securities Account" = 0, "CD Account" = 0, "Online" = 1,"Credit Card" = 1)
test_pre<-as.data.frame(rbind(x,y,z))

test_pre.norm<-test_pre[,-8]
norm.values<- preProcess(test_pre.norm,method = c("center","scale"))
test_pre.norm<-predict(norm.values,test_pre.norm)
nn3<- knn(train.norm.df,test=test_pre.norm,cl=Train_Data$`Personal.Loan`,k=4,prob = TRUE)
CrossTable(x=test_pre$`Personal.Loan`,y=nn3,prop.chisq = FALSE)

#-----------------------------------------------------------------
#Part -5 Repartitioning the data, this time into training, validation, and test sets (50% : 30% : 20%)

library(dplyr)

m_ubank1 <- ubank[,c(-1,-5)]
m_ubank1$Personal.Loan<-factor(m_ubank1$Personal.Loan)

set.seed(20)
Train_Index1 = createDataPartition(m_ubank1$Age,p=0.5, list=FALSE)
Train_Data1 = m_ubank1[Train_Index1,]
Val_Data1 = m_ubank1[-Train_Index1,]

Test_Index1 = createDataPartition(Val_Data1$Age,p=0.2, list=FALSE)
Test_Data1 = Val_Data1[Test_Index1,]
Val_Data1 = Val_Data1[-Test_Index1,]
View(m_ubank1)
summary(Train_Data1)
summary(Val_Data1)
summary(Test_Data1)

train.norm.df1 <- Train_Data1[, -8]
valid.norm.df1 <- Val_Data1[, -8]
Test.norm.df1 <- Test_Data1[, -8]

norm.values1 <- preProcess(train.norm.df1, method=c("center", "scale"))

train.norm.df1 <- predict(norm.values1, train.norm.df1)
valid.norm.df1 <- predict(norm.values1, valid.norm.df1)
Test.norm.df1 <- predict(norm.values1, Test.norm.df1)
View(train.norm.df1)

library(FNN)
k_ubank1<-knn(train.norm.df1,Test.norm.df1,cl=Train_Data1$Personal.Loan,k=3,prob = TRUE)

library(gmodels)
CrossTable(x=Test_Data1$Personal.Loan,k_ubank1,prop.chisq = FALSE)

accuracy.df <- data.frame(k = seq(1, 55, 1), accuracy = rep(0, 55))
for(i in 1:55) {
  knn.pred1 <- knn(train.norm.df1, valid.norm.df1, 
                  cl = Train_Data1$`Personal.Loan`, k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred1, Val_Data1$`Personal.Loan`)$overall[1] 
}
accuracy.df

accuracy.df[which.max(accuracy.df$accuracy),]

#Comparing the accuracies from confusion matrix test set with test and validation set, we found accuracy of 95.2% from confusion matrix and 95.8% for training & validation data
