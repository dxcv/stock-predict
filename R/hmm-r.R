library(depmixS4)
library(TTR)
library(ggplot2)
library(reshape2)
library(plotly)
  # create the returns stream from this
 shdata<-getSymbols( "000001.ss", from="2004-01-01",auto.assign=F )
 gspcRets = diff( log( Cl( shdata ) ) )
 returns = as.numeric(gspcRets)
 write.csv(as.data.frame(gspcRets),"gspcRets.csv")
 shdata=na.omit(shdata)
 df <- data.frame(Date=index(shdata),coredata(shdata))
 p <- df %>%
   plot_ly(x = ~Date, type="candlestick",
           open = ~X000001.SS.Open, close = ~X000001.SS.Close,
           high = ~X000001.SS.High, low = ~X000001.SS.Low, name = "000001.SS",
           increasing = (~X000001.SS.Open - ~X000001.SS.Close)/~X000001.SS.Open, decreasing = (~X000001.SS.Close - ~X000001.SS.Open)/~X000001.SS.Open) %>%
   add_lines(y = ~up , name = "B Bands",
             line = list(color = '#ccc', width = 0.5),
             legendgroup = "Bollinger Bands",
             hoverinfo = "none") %>%
   add_lines(y = ~dn, name = "B Bands",
             line = list(color = '#ccc', width = 0.5),
             legendgroup = "Bollinger Bands",
             showlegend = FALSE, hoverinfo = "none") %>%
   add_lines(y = ~mavg, name = "Mv Avg",
             line = list(color = '#E377C2', width = 0.5),
             hoverinfo = "none") %>%
   layout(yaxis = list(title = "Price"))
