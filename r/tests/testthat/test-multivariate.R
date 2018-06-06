context("Tests for multivariate responses")

test_that("NNKCDE matches density estimate for k=n", {
  set.seed(100)

  n_train <- 1000

  z_grid <- expand.grid(seq(0, 1, length.out = 30),
                        seq(0, 1, length.out = 30))

  x_train <- runif(n_train)
  z_train <- cbind(rbeta(n_train, x_train, 2), rbeta(n_train, 2, x_train))

  obj <- NNKCDE$new(x_train, z_train)

  for (h in c(0.2, 0.9, 1.0, 3.2, 4.5)) {
    x_obs <- runif(1)
    bandwidth <- diag(h, 2)
    expected <- ks::kde(z_train, H = bandwidth, eval.points = z_grid)$estimate
    preds <- obj$predict(x_obs, z_grid, k = n_train, h = bandwidth)

    expect_equal(as.vector(preds), as.vector(expected))
  }

  bandwidth <- matrix(c(0.9, 0.4, 0.4, 0.9), 2, 2)
  expected <- ks::kde(z_train, H = bandwidth, eval.points = z_grid)$estimate
  preds <- obj$predict(x_obs, z_grid, k = n_train, h = bandwidth)
  expect_equal(as.vector(preds), as.vector(expected))
})
