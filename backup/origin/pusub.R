library(quantmod)
setSymbolLookup(WANKE=list(name="000002.sz", src="yahoo"))
getSymbols("WANKE")
head(WANKE)
library(lubridate)

library(e1071)

library(quantmod)
setSymbolLookup(WANKE=list(name="000002.sz", src="yahoo"))
getSymbols("WANKE")



head(WANKE)
tail(WANKE)

startDate <- as.Date("2010-01-01")
endDate <- as.Date("2017-01-01")
DayofWeek <- wday(WANKE, label=TRUE)
PriceChange <- Cl(WANKE) - Op(WANKE)
Class <- ifelse(PriceChange > 0, "UP", "DOWN")

DataSet <- data.frame(DayofWeek, Class)

MyModel <- naiveBayes(DataSet[,1], DataSet[,2])
MyModel

W <- na.omit(WANKE)
DayofWeek <- wday(W, label=TRUE)
PriceChange <- Cl(W) - Op(W)
Class <- ifelse(PriceChange > 0, "UP", "DOWN")
EMA5 <- EMA(Op(W), n = 5)
EMA10 <- EMA(Op(W), n = 10)
EMACross <- EMA5 -EMA10
EMACross <- round(EMACross, 2)
DataSet2 <- data.frame(DayofWeek, EMACross, Class)
DataSet2<-DataSet2[-c(1:10),]
head(DataSet2)

tail(DataSet2)
length(DayofWeek)

TrainingSet<-DataSet2[1:200,]
TestSet<-DataSet2[201:270,] 
EMACrossModel<-naiveBayes(TrainingSet[,1:2],TrainingSet[,3]) 
EMACrossModel

table(predict(EMACrossModel,TestSet),TestSet[,3],dnn=list('predicted','actual')) 