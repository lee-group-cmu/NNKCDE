context("Tests for univariate responses")

test_that("NNKCDE matches normal density for k=1", {
  set.seed(42)

  n_train <- 1000

  z_grid <- seq(-1, 1, length.out = 100)

  x_train <- runif(n_train)
  z_train <- rnorm(n_train, x_train, 1)

  obj <- NNKCDE$new(x_train, z_train)

  for (h in c(0.1, 0.3, 1.0, 2.0, 3.0)) {
    x_obs <- runif(1)
    nearest <- which.min(abs(x_train - x_obs))
    expected <- dnorm(z_grid, z_train[nearest], sd = h)
    preds <- obj$predict(x_obs, z_grid, k = 1, h = h)

    expect_equal(as.vector(preds), as.vector(expected))
  }
})

test_that("NNKCDE matches density estimate for k=n", {
  set.seed(100)

  n_train <- 1000

  z_grid <- seq(0, 1, length.out = 100)

  x_train <- runif(n_train)
  z_train <- rbeta(n_train, x_train, 2)

  obj <- NNKCDE$new(x_train, z_train)

  for (h in c(0.2, 0.9, 1.0, 3.2, 4.5)) {
    x_obs <- runif(1)
    expected <- ks::kde(z_train, h = h, eval.points = z_grid)$estimate
    preds <- obj$predict(x_obs, z_grid, k = n_train, h = h)

    expect_equal(as.vector(preds), as.vector(expected))
  }
})
