library(rstanarm)

set.seed(2904)
N <- 5
x <- rnorm(N, 0, 1)

y <- rnorm(N, 2*x + 1, 1)

data <- data.frame(x=x, y=y)

fit <- lm(y ~ x, data=data)

plot(y ~ x)
abline(fit)

vr <- function(x) {
  n <- length(x)
  (1 / (n-1)) * sum( (x - mean(x))^2 )
}

y_hat <- predict(fit)

classical_R_squared <- function(y_hat, y) {
  vr(y_hat) / vr(y)
}

R_squared_rss_tss <- function(y_hat, y) {
  rss <- sum((y_hat - y) ^ 2)  ## residual sum of squares
  tss <- sum((y - mean(y)) ^ 2)  ## total sum of squares
  1 - rss/tss 
}

classical_R_squared(y_hat, y)
R_squared_rss_tss(y_hat, y)

summary(fit)$r.squared 




bayes_prior <- stan_glm(y ~ x, 
                      prior_intercept = normal(1, 0.1, autoscale=F),
                      prior = normal(2, 0.1, autoscale=F),
                      prior_PD = T,
                      data=data)


bayes_fit <- stan_glm(y ~ x, 
                      prior_intercept = normal(1, 0.1, autoscale=F),
                      prior = normal(2, 0.1, autoscale=F),
                      data=data)

bayes_fit
plot(y ~ x, data=data)
abline(fit)
abline(bayes_prior, col="steelblue")
abline(bayes_fit, lty=2, col="steelblue")

by_hat <- predict(bayes_fit)

classical_R_squared(by_hat, y)
R_squared_rss_tss(by_hat, y)

y <- rstanarm::get_y(bayes_fit)
y_pred <- posterior_linpred(bayes_fit, transform = TRUE)




var_pred <- apply(y_pred, 1, var)

res <- t(apply(y_pred, 1, function(x) y - x))
var_res <- apply(res, 1, var)

( bayesian_Rsquared <- var_pred / ( var_pred + var_res ) )
hist(bayesian_Rsquared)
mean(bayesian_Rsquared)

R_squared_rss_tss(by_hat, y)
mean( bayes_R2(bayes_fit)  )
hist(bayes_R2(bayes_fit))


e <- -1 * sweep(y_pred, 2, y)
var_res <- apply(e, 1, var)

( bayesian_Rsquared <- var_pred / ( var_pred + var_res ) )
