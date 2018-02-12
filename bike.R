library(plyr)
library(dplyr)
library(ggplot2)
library(caTools)
library(rpart)
library(rpart.plot)
library(data.table)
library(mlr)
library(stringr)
library(pROC)
library(caret)
library(xgboost)
library(precrec)


set.seed(123) #random seed


####Charger les donnees
data <- read.csv("2017-Q1-Trips-History-Data.csv")

####data exploration
str(data)

table(is.na(data)) # manque-t-il des données ?
#FALSE 
#5818572 
data=data[-which(data$Bike_number=="?(0x0000000074BEBCE4)"),] ###nettoyage
data$Bike_number=toupper( data$Bike_number)


unique(data$Member.Type)
#[1] "Registered" "Casual" 
# -> probleme logistique/binaire

####data viz
ggplot(data,aes(x=Member.Type))+geom_bar(fill="#FF9999", colour="black")+scale_fill_brewer(palette = "Pastel1")+theme(axis.text.x =element_text(,hjust = 1,size=10))
round(prop.table(table(data$Member_Type))*100)  # proportion
#Casual Registered 
#     18         82

   
# Chercher correlations
task = makeClassifTask(data = data, target = "Member_Type")
var_imp <- generateFilterValuesData(task, method = c("information.gain"))
plotFilterValues(var_imp,feat.type.cols = TRUE)
var_imp$data
                  # name    type information.gain
# 1             Duration integer      0.138995163
# 2           Start.date  factor      0.162134042
# 3             End.date  factor      0.168731865
# 4 Start_station_number integer      0.091141564
# 5        Start.station  factor      0.093930835
# 6   End.station.number integer      0.086516968
# 7          End.station  factor      0.088772250
# 8          Bike.number  factor      0.009434897


# feature engineering: decouper les champs *.date en *date.time et *date.day (separer heures et dates)
splitted_date=str_split_fixed(data$Start.date, " ", 2)
colnames(splitted_date)=c("Start.date.day","Start.date.time")
data=cbind(data, splitted_date)
data=subset(data,select=-c( Start.date))  ## enlever start.date

splitted_date=str_split_fixed(data$End.date, " ", 2)
colnames(splitted_date)=c("End.date.day","End.date.time")
data=cbind(data, splitted_date)
data=subset(data,select=-c( End.date))  ## enlever End.date

data=subset(data,select=-c( Start.station, End.station))  


##### graphiques heures et dates
ggplot(data,aes(x= strftime( as.POSIXct(data$Start.date.time , format="%H:%M"), format="%H:%M")))+geom_histogram(stat="count",fill="#FF9999", colour="black") +scale_x_discrete(name ="Start.date.time",breaks=c("00:00","06:00","12:00","18:00","24:00"), limits=c(sprintf("0%s:00",seq(1:9)) , sprintf("%s:00",seq(10,24))))+scale_fill_brewer(palette = "Pastel1")

ggplot(data,aes(x=as.Date(data$ Start.date.day ,format="%m/%d/%Y ")))+geom_histogram(stat="count",fill="#FF9999")+xlab("Start.date.day")+scale_fill_brewer(palette = "Pastel1")


#########
task = makeClassifTask(data = data, target = "Member_Type")
var_imp <- generateFilterValuesData(task, method = c("information.gain"))
plotFilterValues(var_imp,feat.type.cols = TRUE)
var_imp$data
# name    type information.gain
# 1             Duration integer      0.138995163
# 2 Start.station.number integer      0.091141564
# 3   End.station.number integer      0.086516968
# 4          Bike.number  factor      0.009434897
# 5       Start.date.day  factor      0.056006985
# 6      Start.date.time  factor      0.035879814
# 7         End.date.day  factor      0.055914275
# 8        End.date.time  factor      0.033128906



###### arbre rapide
tr=rpart(Member.Type ~.,data)
summary(tr)
#Variable importance
#       Duration    End.date.day  Start.date.day Start.date.time     Bike.number   End.date.time 
#             78               6               6               4               3               3 

ggplot(data,aes(log10(Duration),fill=data$Member.Type))+geom_histogram(bins=230)+labs(fill="Member.Type") 



#################### #################### #################### 
#################### feature eng
#### transformer les facteurs en donnees numeriques
data <- read.csv("2017-Q1-Trips-History-Data.csv") #refresh
data=subset(data,select=-c( Start.station, End.station))  
colnames(data)=c("Duration" , "Start_date"  , "End_date","Start_station_number" ,"End_station_number" , "Bike_number"  ,"Member_Type")

######conversion des dates en nombre de secondes ecoulees depuis le 1 jan 1970 00:00
data$Start_date=as.numeric(as.POSIXct(strptime(data$Start_date, "%m/%d/%Y %H:%M")))
data$End_date=as.numeric(as.POSIXct(strptime(data$End_date, "%m/%d/%Y %H:%M")))


#### conversion bike number en numeric
data$Bike_number=str_split_fixed(data$Bike_number, "W", 2)[,2]
data$Bike_number=as.numeric(data$Bike_number)

data$Member_Type=as.numeric(data$Member_Type)
data$Member_Type[data$Member_Type==2]=0 ## convertir Member_Type en binaire casual =1 registered =0

## partionner train/test
sample=sample.split(data$Member_Type, SplitRatio=0.7)
train <- data[which(sample==T), ]
test  <- data[ which(sample==F), ]
ytest=subset(test,select=c(Member_Type)) 

round(prop.table(table(train$Member_Type))*100)  # proportion
#    Casual Registered 
#        18         82 


#####arbre rapide
tr=rpart(Member_Type ~.,data=train)
summary(tr)
#Variable importance
#          Duration
#                98              

tree=predict(tr, newdata = test)

threshold=0.5
thresh_tree=tree #casual =1 registered =0
thresh_tree[thresh_tree>threshold]=1
thresh_tree[thresh_tree<=threshold]=0
conf=table(cbind(thresh_tree,ytest))[2:1, 2:1] #matrice de confusion attention il faut inverser la table

           # Member_Type
# thresh_tree      1      0
          # 1  16930   7459
          # 0  18493 151071

confusionMatrix(conf)

# Confusion Matrix and Statistics

           # Member_Type
# thresh_tree      1      0
          # 1  16930   7459
          # 0  18493 151071
                                          
               # Accuracy : 0.8662          
                 # 95% CI : (0.8647, 0.8677)
    # No Information Rate : 0.8174          
    # P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  # Kappa : 0.4902          
 # Mcnemar's Test P-Value : < 2.2e-16       
                                          
            # Sensitivity : 0.47794         
            # Specificity : 0.95295         
         # Pos Pred Value : 0.69417         
         # Neg Pred Value : 0.89094         
             # Prevalence : 0.18264         
         # Detection Rate : 0.08729         
   # Detection Prevalence : 0.12575         
      # Balanced Accuracy : 0.71544         
                                          
       # 'Positive' Class : 1          		  
      
	   
# Accuracy : 0.8662      
# Sensitivity : 0.47794
# Specificity : 0.95295

#    Casual Registered 
#        18         82 	   
		  
rocc=roc( ytest$Member_Type,tree)
plot(rocc,  print.thres = seq(0,1,by=0.1),grid=T)
auc(rocc)
#Area under the curve: 0.7956
plot(rocc,   yaxt="n",xaxt="n",print.auc=T,grid=T, print.thres="best")
axis(2,pos=1.0)
axis(1,pos=0.0)
coords(xrocc, "best", ret = "threshold") #meilleur seuil
#[1] 0.1641969

threshold=0.1641969
thresh_tree=tree #casual =1 registered =0
thresh_tree[thresh_tree>threshold]=1
thresh_tree[thresh_tree<=threshold]=0
conf=table(cbind(thresh_tree,ytest))[2:1, 2:1] #matrice confusion
confusionMatrix(conf)

# Confusion Matrix and Statistics

           # Member_Type
# thresh_tree      1      0
          # 1  25582  32136
          # 0   9841 126394
                                          
               # Accuracy : 0.7836          
                 # 95% CI : (0.7817, 0.7854)
    # No Information Rate : 0.8174          
    # P-Value [Acc > NIR] : 1               
                                          
                  # Kappa : 0.4175          
 # Mcnemar's Test P-Value : <2e-16          
                                          
            # Sensitivity : 0.7222          
            # Specificity : 0.7973          
         # Pos Pred Value : 0.4432          
         # Neg Pred Value : 0.9278          
             # Prevalence : 0.1826          
         # Detection Rate : 0.1319          
   # Detection Prevalence : 0.2976          
      # Balanced Accuracy : 0.7597          
                                          
       # 'Positive' Class : 1                 
                                          
# Accuracy : 0.7836 
# Sensitivity : 0.7222          
# Specificity : 0.7973


############## XGBOOST
############## param arbitraires
dtrain=subset(train,select=c(-Member_Type))
ytrain=subset(train,select=c(Member_Type))

dtest=subset(test,select=c(-Member_Type))
ytest=subset(test,select=c(Member_Type))


ddtrain <- xgb.DMatrix(data = data.matrix(dtrain), label=ytrain$Member_Type)
ddtest <- xgb.DMatrix(data = data.matrix(dtest), label=ytest$Member_Type)
watchlist <- list(train=ddtrain, test=ddtest)


#validation croisee
xgbcv=xgb.cv(params = list(objective = "binary:logistic", eta=0.1,max.depth=6,nthread = 4), ddtrain, nrounds=1000, nfold=5,watchlist=watchlist,early_stopping_rounds=10,error="error")
# [384]	train-error:0.087619+0.000279	test-error:0.093184+0.000565
num_round=384
model <- xgb.train(data=ddtrain, max.depth=5, eta=0.1, nthread = 4, nround=num_round, objective = "reg:logistic",eval_metric="error",watchlist=watchlist,early_stopping_rounds=10)

pred=predict(model, data.matrix(dtest))


threshold=0.5
pred_df=as.data.frame(pred)
pred_df[pred_df>=threshold]=1
pred_df[pred_df<threshold]=0


err=subset(cbind(pred_df,ytest), pred_df!=Member_Type) ### rows fausses
1-nrow(err)/nrow(dtest)  # correctes
#[1] 0.9069053
conf=table(cbind(pred_df,ytest)) [2:1, 2:1]#matrice confusion
confusionMatrix(conf)

# Confusion Matrix and Statistics

    # Member_Type
# pred      1      0
   # 1  21492   4212
   # 0  13931 154318
                                          
               # Accuracy : 0.9065          
                 # 95% CI : (0.9052, 0.9077)
    # No Information Rate : 0.8174          
    # P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  # Kappa : 0.6493          
 # Mcnemar's Test P-Value : < 2.2e-16       
                                          
            # Sensitivity : 0.6067          
            # Specificity : 0.9734          
         # Pos Pred Value : 0.8361          
         # Neg Pred Value : 0.9172          
             # Prevalence : 0.1826          
         # Detection Rate : 0.1108          
   # Detection Prevalence : 0.1325          
      # Balanced Accuracy : 0.7901          
                                          
       # 'Positive' Class : 1                         
 # Accuracy : 0.9065           
 # Sensitivity : 0.6067
 # Specificity : 0.9734  

 
xrocc=roc( ytest$Member_Type,pred)
plot(xrocc,  print.thres = seq(0,1,by=0.1),grid=T)
auc(xrocc)
#Area under the curve:  0.9092
coords(xrocc, "best", ret = "threshold") #best threshold
#[1] 0.1937075
plot(xrocc,   yaxt="n",xaxt="n",print.auc=T,grid=T, print.thres="best")
axis(2,pos=1.0)
axis(1,pos=0.0)


threshold=0.1937075
pred_df=as.data.frame(pred)
pred_df[pred_df>=threshold]=1
pred_df[pred_df<threshold]=0

conf=table(cbind(pred_df,ytest))[2:1, 2:1] #matrice confusion
confusionMatrix(conf)

# Confusion Matrix and Statistics

    # Member_Type
# pred      1      0
   # 1  27664  16625
   # 0   7759 141905
                                          
               # Accuracy : 0.8743          
                 # 95% CI : (0.8728, 0.8758)
    # No Information Rate : 0.8174          
    # P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  # Kappa : 0.6162          
 # Mcnemar's Test P-Value : < 2.2e-16       
                                          
            # Sensitivity : 0.7810          
            # Specificity : 0.8951          
         # Pos Pred Value : 0.6246          
         # Neg Pred Value : 0.9482          
             # Prevalence : 0.1826          
         # Detection Rate : 0.1426          
   # Detection Prevalence : 0.2283          
      # Balanced Accuracy : 0.8380          
                                          
       # 'Positive' Class : 1       
	   
# Accuracy :  0.8743 
# Sensitivity : 0.7810 
# Specificity : 0.8951



 ############## xgboost meilleurs params 	   
 
 
best_param = list()
best_error = Inf
best_error_index = 0

t0=proc.time()
for (iter in 1:100) {
    param <- list(
          max_depth = sample(6:10, 1),
          eta = runif(1, .01, .3),
          colsample_bytree = runif(1, .5, .8)
          )
    cv.nround = 500
    cv.nfold = 5

	
    xgbcv <- xgb.cv(data=ddtrain, params = param,nthread=4, 
                    nfold=cv.nfold, nrounds=cv.nround,
                    verbose = T, early_stop_round=10, maximize=FALSE, error="error",objective="binary:logistic")

    error =  min(xgbcv$evaluation_log$test_error_mean)
    min_error_index = which.min(xgbcv$evaluation_log$test_error_mean)

    if (error < best_error) {
        best_error = error
        best_error_index = min_error_index
        best_param = param
		best_model=xgbcv
    }
}

proc.time()-t0	
# 7.209444 heures
best_param

# max_depth=10
# eta=0.12006
# colsample_bytree=0.7512918


########predictions avec les meilleurs parametres
best_model <- xgb.train(data=ddtrain, eta=0.12006, nthread = 4, nround=500, max_depth=10,objective = "reg:logistic",eval_metric="error",watchlist=watchlist,early_stopping_rounds=10)

pred=predict(best_model, data.matrix(dtest))
xrocc=roc(ytest$Member_Type,pred)
plot(xrocc,  print.thres = seq(0,1,by=0.1),grid=T)
auc(xrocc)
#Area under the curve: 0.9195
coords(xrocc, "best", ret = c("threshold", "precision", "recall"))
# threshold       precision    recall 
#[1] 0.1693627    0.6149493 0.8128899 

plot(xrocc,   yaxt="n",xaxt="n",print.auc=T,grid=T, print.thres="best")
axis(2,pos=1.0)
axis(1,pos=0.0)

threshold= 0.1693627
pred_df=as.data.frame(pred)
pred_df[pred_df>=threshold]=1
pred_df[pred_df<threshold]=0

conf=table(cbind(pred_df,ytest))[2:1, 2:1] #matrice confusion
confusionMatrix(conf)

# Confusion Matrix and Statistics

    # Member_Type
# pred      1      0
   # 1  28795  18030
   # 0   6628 140500
                                          
               # Accuracy : 0.8729          
                 # 95% CI : (0.8714, 0.8743)
    # No Information Rate : 0.8174          
    # P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  # Kappa : 0.6215          
 # Mcnemar's Test P-Value : < 2.2e-16       
                                          
            # Sensitivity : 0.8129          
            # Specificity : 0.8863          
         # Pos Pred Value : 0.6149          
         # Neg Pred Value : 0.9550          
             # Prevalence : 0.1826          
         # Detection Rate : 0.1485          
   # Detection Prevalence : 0.2414          
      # Balanced Accuracy : 0.8496          
                                          
       # 'Positive' Class : 1            
	   
# Accuracy : 0.8729   
# Sensitivity : 0.8129 
# Specificity : 0.8863

# precision : 0.6149493
# recall : 0.8128899
# Mcc : 0.6312289
# F1-score : 0.7001994      

###### graph precision-rappel
sc <- evalmod(scores = pred, labels = ytest$Member_Type)
autoplot(sscurves, "PRC")