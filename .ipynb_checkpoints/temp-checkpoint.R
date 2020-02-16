library(RCurl)

mydata <- read.csv(text = getURL("https://raw.githubusercontent.com/tristanga/MonteCarlo_ForecastRisk/master/TS.csv"))

# Create time serie
tseries <- ts(mydata$x, frequency = 12, start = c(2000, 1))

# Check time series
start(tseries)
end(tseries)
frequency(tseries)

# remove q4 (the one we want to predict)
tseries_sub <- window(tseries, start=c(2000,1), end=c(2015,9))
tseries_sub

# Define your target
mytarget = 186.0000


# Calculate actuals
actualYTD <- sum(window(tseries, start=c(2015, 1), end=c(2015,9)))
actualYTD

# check the distribution of your time series
hist(tseries_sub)
boxplot(tseries_sub)
tseries_df <- as.data.frame(tseries_sub)


####
# Fit a traditional distribution  to the observed time series
library(fitdistrplus)
fit.norm <- fitdist(as.numeric(tseries_df$x), "norm")
plot(fit.norm)

fit.exp <- fitdist(as.numeric(tseries_df$x), "exp")
fit.weibull <- fitdist(as.numeric(tseries_df$x), "weibull")
fit.lnorm <- fitdist(as.numeric(tseries_df$x), "lnorm")
fit.gamma <- fitdist(as.numeric(tseries_df$x), "gamma")
fit.logistic <- fitdist(as.numeric(tseries_df$x), "logis")
fit.cauchy <- fitdist(as.numeric(tseries_df$x), "cauchy")


# compare goodness-of-fit statistics
gofstat(list(fit.norm, fit.exp, fit.weibull, fit.lnorm, fit.gamma, fit.logistic, fit.cauchy),
        fitnames = c("fit.norm", "fit.exp", "fit.weibull", "fit.lnorm", "fit.gamma", "fit.logistic", "fit.cauchy"))


# the best Goodness-of-fit statistics is for the normal distribution
option1 = fit.norm
summary(option1)


# weibull distribution potentially also good
plot(fit.weibull)


#### fit a supplemental distribution
# Using Supplementary Distributions to fit the second option
library(SuppDists)
parms <- JohnsonFit(as.numeric(tseries_df$x), moment="quant")


# plot the distribution
hist( as.numeric(tseries_df$x) , freq=FALSE)
plot(function(x) dJohnson(x,parms), 0, 20, add=TRUE, col="red")

# let's create samples for october, november and december
option2 <- function(x)qJohnson(x,parms)




new_post(title='My first post.Rmd')
