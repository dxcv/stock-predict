library(plyr)
library(quantmod)
library(TTR)
library(ggplot2)
library(scales)
library(xts)
library(depmixS4)
library(quantmod)

set.seed(123)
Sys.setenv(tz<-"UTC")
#sp500<- getYahooData("^xx", start<-20100101, end<-20150101,freq<-"daily")
getSymbols("AAPL", src = "yahoo", from = "2010-01-01", to = "2015-01-01")
head(AAPL)
tail(AAPL)
ep <- endpoints(AAPL,on<- "months", k <-1)

AAPLLR<- AAPL[ep[2:(length(ep)-1)]]
AAPLLR<- na.exclude(AAPLLR)

AAPLLRdf<- data.frame(AAPLLR)
mod<-depmix(logret~1,family<-gaussian(),nstates<-4,data<-AAPLLR)
set.seed(1)

summary(fm2)

Classification(inference task)
probs <- posterior(fm2)
head()probs)


rowSums(head(probs)[,2:5])

pBear<-probs[,2]
sp500LRdf$pBear <- pBear

szie <- 60*0.3

bullLowVol <- model1ReturnsFunc(F)
bullHighVol <- model1ReturnsFunc(T)
model1TrainingReturns<-c(bullLowVOl,bullHighVOl)
Model1Fit <- HMM(model1TrainingReturns,nStates<-2)

model1ReturnsFunc <- function(isHighVOl){
returen(rnorm(size,-0.1),if(ifHighVOl){0.15}else{0.02}))
}

bearLowVol <-model2ReturnsFunc(F)
bearHighVol<-model2ReturnsFunc(T)
model2TrainingReturns<-c(bullLowVOl,bullHighVOl)

Model2Fit <- HMM(model2TrainingReturns,nStates<-2)
