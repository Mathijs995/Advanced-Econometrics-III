# Import and pre-process the data.
dataset <- readr::read_csv("fd_X.csv")
fd_X <- dataset$`0`

# Estimate the ARFIMA model.
arfima_model <- fracdiff::fracdiff(fd_X, nar = 1, nma = 1, drange=c(-0.5, 0))
ar <- c(arfima_model$ar)
ma <- -c(arfima_model$ma) # Note that the MA coefficients have inverted signs compared to other parametrizations.
sigma <- arfima_model$sigma
d <- arfima_model$d

# Print model parameters to console.
print(paste0("AR coefficient: ", ar))
print(paste0("MA coefficient: ", ma))
print(paste0("Fractional-differencing parameter: ", d))
print(paste0("Sigma: ", sigma))

# Simulate 10000 observations to plot the theoretical autocorrelation function.
num_sim <- 10000
lag.max <- 50
sim_fd_X <- fracdiff::fracdiff.sim(n=num_sim, ar=ar, ma=-ma, d=d, rand.gen=function(n) (rnorm(n, mean=0, sd=sigma)))
filtered_fd_X <- fracdiff::diffseries(fd_X, d)

# Plot the simulated autocorrelation function for 50 lags.
z <- acf(sim_fd_X$series, lag.max=lag.max)
plot(z$acf[2:lag.max],
     type='h',
     main='Autocorrelation of first differenced X_t',
     xlab="Lag number",
     ylab='acf',
     ylim=c(-0.1, 0.1),
     las=1,
     xaxt="n")
abline(h=0)
axis(1, at=c(1:lag.max), labels=c(1:lag.max))