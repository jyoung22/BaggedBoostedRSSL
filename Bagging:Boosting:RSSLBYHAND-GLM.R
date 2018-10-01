# Bagging and RSSL

#For easier reading, I've split up the following ensembles code into 3 chunks:

####################### 1.
##  Set up - aka intializing variables

B = 50 # this is the number of base learners you want to create

# storing a few key variables below 
n = nrow(sim_dat_cont) # number of obs
q = ncol(sim_dat_cont) - 1 # number of attributes
d = min(round(sqrt(q)), round(n/5)) # RSSL attribute selection

# empty matrices to store model parameters
mbeta_bag = matrix(0, nrow=B, ncol=(q+1))
mbeta_rssl = matrix(0, nrow=B, ncol=(q+1))


####################### 2. 
## Replication loop for model creating and storing- 
## this is where you create the base learners!

for(b in 1:B){ # setting up the replication loop

    # storing the sampling IDs
    id.boot = sample(n, replace=T) # bootstrap
    id.bag = 1:q                             # No subset selection: Bagged
    id.rssl = sort(sample(1:q, d, replace=F)) # Equal weight selection: RSSL
    
    # sample the data set
    dat.bag = sim_dat_cont[id.boot, c(1, 1 + id.bag)]    # Bagged data 
    dat.rssl = sim_dat_cont[id.boot, c(1, 1 + id.rssl)]    # RSSL data 
    
    # OLS models from sampled data sets
    bag_mod = glm(y~., data = dat.bag) # model with bagged data
    rssl_mod = glm(y~., data = dat.rssl) # model with rssl data
    
    # Store the model parameters for aggregation
    mbeta_bag[b, c(1, 1 + id.bag)] = coef(bag_mod)   # bagged
    mbeta_rssl[b, c(1, 1 + id.rssl)] = coef(rssl_mod)   # RSSL
}
# the above loop is farily fast with our simulated data so
# I placed the cat statement outside it 
cat("Done with all", B," base learners \n") 


####################### 3.
## Predictions


# to save space, clarify which type (RSSL or Bagged) of ensemble you want to
# make predictions with by commenting out of of the below type statements

type = 'RSSL' # for RSSL
type = 'Bagged' # for Bagged

switch(type, RSSL = {mbeta = mbeta_rssl}, Bagged = {mbeta = mbeta_bag})

#  xnew is the test set normally
xnew = x = cbind(x_1, x_2) # the original attributes
# add a column of all 1s to data frame for the intercept
x = cbind(rep(1, nrow(xnew)),xnew) 
nnew = ncol(xnew) + 1 # number of total columns
alpha = matrix(1/B, nrow=B, ncol=nnew) # Equally weight the base learners
delta = alpha * mbeta # weighted parameters
yhats = colSums(1/(1 + exp(-delta %*% t(x)))) # final overall predictions


#Note that this isn't matrix multiplication! 
#Instead we want to multiply the 1st column with the 1st column, 2nd with 2nd, 
#and 3rd wiht 3rd ONLY! Our goal is to weight the predictions in mbeta by our alpha scheme. 
#Here we're equally weighting, so every point in alpha is the same, 
#so we could replace  alpha * mbeta with .02 * mbeta.


# Boosting
 
#Again, I've split up the code into 3 chunks for easier reading.

####################### 1. 
## Set up - aka intializing variables


# storing a few key variables below 
y = sim_dat_cont$y # response
n = nrow(sim_dat_cont) # no. obs
m = round(0.75*n) # we don't use every obs every time
# so store what should be 75% of them
Tr = 20     # Number of base learners

alpha = numeric(Tr) # initialize 
epsilon = numeric(Tr) # initialize error vector
weight = rep(1/n, n) # initialize obs weighting scheme vector
# start with equal weighting
h = NULL # initialize object to store models



####################### 2. 
## Replication loop for model creating and storing- this is where you create the base learners!
##  Note that the point of only storing decent base learners is because we aim 
##  to create an overall better model by minimizing a loss function. 
##  For the following chunk of code, the loss function used is the exponential 
##  loss function $\frac{1}{n}\sum_{i=1}^{n}\text{exp}(-y_iF(x_i))$
    

for(t in 1:Tr)	
{
    decent_base_learner = 0 # start with no decent base learners
    while(!decent_base_learner) # while there is none,
        # do the following:
    {   
        # bootstrap data with weighting scheme
        # this weighting scheme will update with every model
        # to favor obs. that are harder to predict
        # don't use all attributes
        boost_id = sample(1:n, m, replace=T, prob=weight) # bootstrap
        # create a base learner based on the bootstrap sample above
        base_learner = glm(y~., data=sim_dat_cont, subset=boost_id)
        # predict with the base learner
        yhat = predict(base_learner, subset(sim_dat_cont,select=-y))
        # calulate the exponential loss
        epsilon_candidate = mean(exp(-yhat*y))
        # if our error is less than .5, aka decent,
        # then it's a good learner and we're done
        decent_base_learner = (epsilon_candidate < 0.5)
    }
    
    # store the candidate's error 
    # pad errors of 0 slightly to allow division
    if(epsilon.candidate==0){epsilon_candidate = .0001}
    
    epsilon[t] = epsilon_candidate
    # reconfigure model's weight
    alpha[t] = (1/2)*log((1-epsilon[t])/epsilon[t])
    weight = weight*exp(-alpha[t]*y*yhat)
    # make the weight a probability that sums to 1  
    weight = weight/sum(weight)
    
    # add the base learner to the list
    h = c(h, list(base_learner))	
}
# return the model's weight, base learners, and weight
return(list(alpha=alpha, h=h, weight=weight))


####################### 3. Predictions

# predict from the boosted glm
predict.boosted.glm <- function(h, alpha, xnew)
{
    #  get how many base learners are used
    Tr      <- length(h)
    # find the number of obs
    nnew   <- nrow(xnew)
    # create an empty list
    h.t.x  <- NULL
        # loop through all the learners to predict
        for(t in 1:Tr){h.t.x <- cbind(h.t.x, pm(round(predict(h[[t]],xnew, type='response'))))}
    # put the model weight with the model's prediction for each obs
    m.alpha <- matrix(rep(alpha, nnew), byrow=Tr, nrow=nnew)
    # keep only one prediction per obs based on the highest model weight
    return(ifelse(rowSums(m.alpha*h.t.x)<0,-1,+1))   
} 
