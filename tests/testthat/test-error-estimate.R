context("Test loss estimation")

if (requireNamespace("cdetools", quietly = TRUE)) {
  test_that("NNKCDE loss matches integral", {
    set.seed(42)

    n_train <- 1000
    n_validation <- 100
    n_grid <- 1000

    z_grid <- seq(-5, 5, length.out = n_grid)

    x_train <- runif(n_train)
    z_train <- rnorm(n_train, x_train, 1)

    x_validation <- runif(n_validation)
    z_validation <- rnorm(n_validation, x_validation, 1)

    obj <- NNKCDE$new(x_train, z_train)

    for (h in c(0.1, 1.0, 3.0)) {
      for (k in c(1, 5, 10)) {
        cde <- obj$predict(x_validation, z_grid, k, h)
        expected <- cdetools::cde_loss(cde, z_grid, z_validation)$loss

        loss <- obj$estimate_loss(x_validation, z_validation, k, h)$loss

        expect_equal(loss, expected, tol = 1e-2)
      }
    }
  })
}
