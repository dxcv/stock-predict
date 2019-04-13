library(class)
library(dplyr)
library(lubridate)
library(scatterplot3d)

stocks <- read.csv(file.choose())
head(stocks)
summary(stocks[,-1])

cl <- stocks$Increase 
colors <- 2-cl
scatterplot3d(stocks[,2:4],color=colors, col.axis=5,            
+   col.grid="lightblue", main="scatterplot3d - stocks", pch=20)
stocks$Date <- ymd(stocks$Date)
stocksTrain <- year(stocks$Date) < 2014
predictors <- cbind(lag(stocks$X16, default = 2.13), lag(stocks$X1, default = 5.10))
colnames(predictors)=c("X16","X1")
train <- predictors[stocksTrain, ] 
test <- predictors[!stocksTrain, ] 
par(mfrow=c(3,2))

acf(stocks$X16)
pacf(stocks$X16)

cl <- stocks$Increase[stocksTrain] 
prediction <- knn(train, test, cl, k = 1) 
table(prediction, stocks$Increase[!stocksTrain])

mean(prediction == stocks$Increase[!stocksTrain])

accuracy <- rep(0, 10)
k <- 1:10
for(x in k){
+ prediction <- knn(predictors[stocksTrain, ], predictors[!stocksTrain, ],
+ stocks$Increase[stocksTrain], k = x)
+ accuracy[x] <- mean(prediction == stocks$Increase[!stocksTrain])
+ }
plot(k, accuracy, type = 'b', col=125,lwd=3)
