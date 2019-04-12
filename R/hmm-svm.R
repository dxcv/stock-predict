> library(e1071)
> a<-read.table("C:\\Users/jacky/Desktop/000016.txt")
> xtrain<-seq(1,2500,by=1)
> xtest<-seq(2501,3010,by=1)
> y<-as.vector(a[2:3011,5])
> y<-as.numeric(y)   
> ytrain<-y[2:2501]
> ytest<-y[2502:3011]
> m<-svm(xtrain, ytrain)
> new<-predict(m, xtest)
> s<-0
> for(i in 1:15)
+ {
+ if(abs(ytest[i]-new[i])<1.35)
+ s<-s+1
+ }
> s
[1] 10
> 1.35/mean(new)#相对误差0.149296
[1] 0.149296
> aa<-0
> bb<-0
> for(i in 1:14)
+ {
+ aa[i]<-ytest[i+1]-ytest[i]
+ bb[i]<-new[i+1]-new[i]
+ }
> ss<-0
> for(j in 1:14)
+ {
+ if(aa[j]*bb[j]>0)
+ {
+ ss<-ss+1
+ }
+ }
> ss#趋势正确率为百分之五十
[1] 7