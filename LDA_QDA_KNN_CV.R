CV.error.f  =  function(predictors=predictors,response=response,k=10,p1=.01,p2=.99,m=99,
                       Rounds=300) {
    # k-fold CV; 10 folds, 300 reps (rounds)
    # LDA/QDA test out m=100 probs from p1 to p2

    # use a sequence for prob
    # Set the vecotrs for P1 and P2
    P1  =  seq(from=p1, to=p2, length=m)
    P2  =  1-P1
    
    # Make the matrix to store error in
    Err.mat.lda  =  matrix(0,k,m)
    Err.mat.qda  =  matrix(0,k,m)
    Err.mat.knn  =  matrix(0,k,10)
    Final.Err.mat = matrix(0,Rounds,3)
    
    # repeat for each round
    for (i in 1:Rounds) {
        
        # split the data up randomly into 2/3 and 1/3    
        sample_size = floor(2/3*nrow(predictors))
        
        # create folds
        fold = sample(rep(1:k,length=sample_size))
        
        # training set indices
        tr.ind = sample(seq_len(nrow(predictors)), size=sample_size)
        
        # test set
        dat.r.te=predictors[-tr.ind,]
        
        # training set
        dat.r.tr=predictors[tr.ind,]

        # Store the responses for the training set
        y.r.tr = response[tr.ind]
                
        # Store the responses for the test set
        y.r.te = response[-tr.ind]
        
        # repeat for each fold
        for (j in 1:k) {
            # create the indexing
            cond  =  (fold==j)
            
            # use the fold we're on to train
            dat.te=dat.r.tr[cond,]
            
            # don't use the fold we're on test
            dat.tr=dat.r.tr[!cond,]
            
            # Store the responses for the test
            y.tr = y.r.tr[!cond]
                        
            # Store the responses for the test
            y.te = y.r.tr[cond]
            
            # repeat for each Prob
            for (ind in 1:m) {
                
                # Don't use the jth fold
                # LDA model
                lda.obj  =  lda(x = dat.tr,
                                grouping = y.tr,
                                prior=c(P1[ind],P2[ind]))
                
                # QDA model
                qda.obj  =  qda(x = dat.tr,
                                grouping = y.tr,
                                prior=c(P1[ind],P2[ind]))
                
                # predict LDA!
                predict.lda.uni = predict(lda.obj,newdata=dat.te)$class
                
                # predict QDA!
                predict.qda.uni = predict(qda.obj,newdata=dat.te)$class
                
                # Calculate predicted error rate for the chunk
                # average how many don't match 
                # this gives error rate
                Pred.err.lda = mean(predict.lda.uni != y.te)
                
                Pred.err.qda = mean(predict.qda.uni != y.te)
                
                # store the error rates for LDA
                Err.mat.lda[j,ind] = Err.mat.lda[j,ind]+Pred.err.lda
                
                # store the error rates for QDA
                Err.mat.qda[j,ind] = Err.mat.qda[j,ind]+Pred.err.qda
            }
            
            # repeat for each k in knn
            for (l in 1:10) {
                
                # Don't use the jth fold
                # LDA model
                knn.obj  =  knn(k=l,
                                train=as.matrix(dat.tr),
                               test=as.matrix(dat.te), cl=y.tr)
                
                # Calculate predicted error rate for the chunk
                # average how many don't match 
                # this gives error rate
                Pred.err.knn = mean(knn.obj != y.te)
                
                # store the error rates for LDA
                Err.mat.knn[j,l] = Err.mat.knn[j,l]+Pred.err.knn
                
            }
            cat("Replication=",i,"fold=",j,"\n")
        }
# end of all folds loop
 
        ### LDA PRIORS
        
            # find the average per prior combination
            # so average by col
            averages.lda = apply(Err.mat.lda,2,mean)
            
            # combine the prior list with the averages appropriately
            # this was we can easily grab the prior we deem optimal
            Opt.lda = as.data.frame(cbind('average'=averages.lda,
                                         'prior 1'=P1))
            
            # find the minimum error
            min.cv.lda.error = min(Opt.lda$average)
            
            # find the corresponding optimal prior 1
            Opt.p1.lda = Opt.lda[Opt.lda$average==min.cv.lda.error,]$`prior 1`[1]
            Opt.p2.lda = 1-Opt.p1.lda  
        
        ### QDA PRIORS
        
            # find the average per prior combination
            # so average by col
            averages.qda = apply(Err.mat.qda,2,mean)
            
            # combine the prior list with the averages appropriately
            # this was we can easily grab the prior we deem optimal
            Opt.qda = as.data.frame(cbind('average'=averages.qda,
                                         'prior 1'=P1))
            
            # find the minimum error
            min.cv.qda.error = min(Opt.qda$average)
            
            # find the corresponding optimal prior 1
            Opt.p1.qda = Opt.qda[Opt.qda$average==min.cv.qda.error,]$`prior 1`[1]
            Opt.p2.qda = 1-Opt.p1.qda  
        
        ### KNN OPT K
        
            # find the average per k
            # so average by col
            averages.knn = apply(Err.mat.knn,2,mean)
            
            # combine the prior list with the averages appropriately
            # this was we can easily grab the prior we deem optimal
            Opt.knn = as.data.frame(cbind('average'=averages.knn,
                                         'k'=c(1:10)))
            
            # find the minimum error
            min.cv.knn.error = min(Opt.knn$average)
            
            # find optimal k
            Opt.k.knn = Opt.knn[Opt.knn$average==min.cv.knn.error,]$`k`
        
        
        ##### Predictions!
        
            # LDA:
            #  model
            #  predict
            #  find error
            #  store
            lda.fin  =  lda(data = dat.r.tr,
                            grouping = y.r.tr,
                            prior=c(Opt.p1.lda, Opt.p2.lda))
            predict.lda.uni.fin = predict(lda.fin,newdata=dat.r.te)$class
            Pred.err.lda.fin = mean(predict.lda.uni.fin != y.r.te)
            Final.Err.mat[i,1] = Final.Err.mat[i,1]+Pred.err.lda.fin
            
            
            
            # QDA:
            #  model
            #  predict
            #  find error
            #  store
            qda.fin  =  qda(data = dat.r.tr,
                            grouping = y.r.tr,
                            prior=c(Opt.p1.qda, Opt.p2.qda))
            predict.qda.uni.fin = predict(qda.fin,newdata=dat.r.te)$class
            Pred.err.qda.fin = mean(predict.qda.uni.fin != y.r.te)
            Final.Err.mat[i,2] = Final.Err.mat[i,2]+Pred.err.qda.fin
            
            
            
            # KNN:
            #  model w/predictions
            #  find error rate
            #  store
            knn.fin  =  knn(k=Opt.k.knn,train=as.matrix(dat.r.tr),
                           test=as.matrix(dat.r.te), cl=y.r.tr)
            Pred.err.knn.fin = mean(knn.fin != y.r.te)
            Final.Err.mat[i,3] = Final.Err.mat[i,3]+Pred.err.knn.fin
        
        
    }
    # return the final matrix
    return(Final.Err.mat)
    
cat("Replication=",i,"done \n")
} 


#  Find errors for CV on LDA, QDA, and KNN
#  Feel free to change the number of folds and reps depending on data size
CV.error.list = CV.error.f(predictors = predictors, response=response,
                           k=10, Rounds=300)

# save it because you never want to run that again
save(CV.error.list, file="~/Desktop/CV_for_LDA_QDA_KNN.RData")


#### Seeing the results
    # visualize data with boxplots
    # For a quick base boxplot
    boxplot(CV.error.list, names=c('LDA','QDA','KNN'), yaxt='n')
    # make the y axis show up as percents
    axis(2, at=pretty(CV.error.list), 
         lab=paste(pretty(CV.error.list) * 100,'%'), las=TRUE)
    # add a title
    title(main='Boxplot of Test Error')
    # show the Bayes' risk (aka the best error possible)
    abline(h = .067, lty=1.5, lwd=1, col='red')
    
    # Find errors
    # store the averages
    averages = as.data.frame(t(round(apply(CV.error.list,2,mean),3)))
    # give the dataframe column and row names
    colnames(averages) = c('LDA','QDA','KNN')
    row.names(averages) = 'Average Test Error'
    
    print(averages)




