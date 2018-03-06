context("Test example")

test_that("Sample usage case works", {
  set.seed(42)

  n_train <- 1000
  n_validation <- 1000
  n_test <- 100
  n_grid <- 100

  z_grid <- seq(0, 1, length.out = n_grid)

  x_train <- runif(n_train)
  z_train <- runif(n_train, 0, x_train)

  x_validation <- runif(n_validation)
  z_validation <- runif(n_validation, 0, x_validation)

  x_test <- runif(n_test)
  z_test <- runif(n_test, 0, x_test)

  ## Train
  obj <- NNKCDE$new(x_train, z_train, h = 0.05)

  ## Estimate errors
  k_grid <- 1:10
  loss_list <- obj$estimate_loss(x_validation, z_validation, k_grid = k_grid)
  expect_equal(length(loss_list$loss), 10)
  expect_equal(k_grid, loss_list$k)

  ## Tune to minimum loss
  obj$tune(x_validation, z_validation, k_grid = seq(1, 20, 100))
  expect_false(is.null(obj$k))

  cdes <- obj$predict(x_test, z_grid)
  expect_equal(dim(cdes), c(n_test, n_grid))
})
