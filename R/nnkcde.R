#' NNKCDE
#'
#' The NNKCDE class. Provides functions for computing and tuning
#' kernel conditional density estimates using nearest neighbors.
#'
#'
#' @section Usage:
#' \preformatted{fit <- NNKCDE$new(x_train, z_train)
#'
#' fit$tune(x_validation, z_validation, k_grid, h_grid)
#' fit$predict(x_test, z_grid, k)
#' }
#'
#' @section Arguments:
#' \code{x_train} The training covariates; either a vector or a matrix
#'   in which each row corresponds to an observation.
#'
#' \code{z_train} The training responses; either a vector or a matrix
#'   in which each row corresponds to an observation.
#'
#' \code{k} (optional) The number of neighbors.
#'
#' \code{h} (optional) The bandwidth for kernel density estimation.
#'
#' @section Methods:
#' \code{$new(x_train, z_train, k=NULL, h=NULL)} Initializes a new
#'   NNKCDE object.
#'
#'
#' \code{$tune(x_validation, z_validation, k_grid = NULL, h_grid =
#'   NULL)} Selects the parameters which minimize the CDE loss; sets
#'   the attributes k and h accordingly.
#'
#' \code{$predict(x_test, z_grid, k = NULL, h = NULL} Returns a matrix
#'   of conditional density estimates cde[ii, jj] where ii indexes the
#'   observations and jj indexes the z_grid.
#'
#' @name NNKCDE
#' @examples
#' \dontrun{
#' fit <- NNKCDE$new(x_train, z_train)
#' fit$tune(x_validation, z_validation, k_grid = c(5, 10, 15, 20))
#' fit$predict(x_test, z_grid, h = 0.1)
#' }
#' @export
NULL

#' @importFrom R6 R6Class
NNKCDE <- R6::R6Class("NNKCDE", #nolint
                      public = list(
                        x = NULL,
                        z = NULL,
                        k = NULL,
                        h = NULL))

NNKCDE$set("public", "initialize",
function(x_train, z_train, k = NULL, h = NULL) {
  x_train <- as.matrix(x_train)
  z_train <- as.matrix(z_train)

  n_train <- nrow(z_train)
  n_dim <- ncol(z_train)

  stopifnot(nrow(x_train) == n_train)

  self$x <- x_train
  self$z <- z_train
  self$h <- h
  self$k <- k
})

NNKCDE$set("public", "tune",
function(x_validation, z_validation, k_grid = NULL, h = NULL) {
  loss_df <- self$estimate_loss(x_validation, z_validation, k_grid, h)
  self$k <- loss_df$k[which.min(loss_df$loss)]
  self$h <- h
  return(loss_df)
})

NNKCDE$set("public", "predict",
function(x_grid, z_grid, k = NULL, h = NULL) {
  if (is.null(k)) {
    k <- self$k
    stopifnot(!is.null(k))
  }
  if (is.null(h)) {
    h <- self$h
    stopifnot(!is.null(h))
  }

  x_grid <- as.matrix(x_grid)
  z_grid <- as.matrix(z_grid)

  stopifnot(ncol(x_grid) == ncol(self$x))

  cdes <- matrix(NA, nrow(x_grid), nrow(z_grid))
  orders <- FNN::knnx.index(self$x, x_grid, k = k)
  for (ii in seq_len(nrow(x_grid))) {
    subset <- self$z[orders[ii, ], , drop = FALSE] #nolint
    if (ncol(self$z) == 1) {
      cdes[ii, ] <- ks::kde(subset, h = h, eval.points = z_grid)$estimate
    } else {
      cdes[ii, ] <- ks::kde(subset, H = h, eval.points = z_grid)$estimate
    }
  }

  return(cdes)
})

NNKCDE$set("public", "estimate_loss",
function(x_validation, z_validation, k_grid= NULL, h = NULL) {
  x_validation <- as.matrix(x_validation)
  z_validation <- as.matrix(z_validation)

  n_train <- nrow(self$z)
  n_validation <- nrow(z_validation)
  n_dim <- ncol(self$z)

  if (is.null(k_grid)) {
    stopifnot(!is.null(self$k))
    k_grid <- self$k
  }
  k_grid <- sort(k_grid)

  if (is.null(h)) {
    if (n_dim == 1) {
      h <- ks::hpi(self$z)
    } else {
      h <- ks::Hpi(self$z)
    }
  }

  stopifnot(nrow(x_validation) == n_validation)
  stopifnot(ncol(z_validation) == n_dim)

  orders <- FNN::knnx.index(self$x, x_validation, k = n_train)

  if (n_dim == 1) {
    det <- h
    invh <- 1 / h ^ 2
  } else {
    det <- det(h)
    invh <- solve(h)
  }

  d <- matrix(NA, n_train, n_train)
  for (ii in 1:n_train) {
    for (jj in 1:n_train) {
      delta <- self$z[ii, ] - self$z[jj, ]
      d[ii, jj] <- t(delta) %*% invh %*% delta
    }
  }
  d <- exp(-d / 4)

  term1 <- rep(0.0, length(k_grid))
  term2 <- rep(0.0, length(k_grid))
  for (ii in 1:n_validation) {
    cde_est <- 0.0
    integral_est <- 0.0
    last_k <- 0
    z <- z_validation[ii, ]
    for (kk in seq_along(k_grid)) {
      ids <- orders[ii, 1:k_grid[kk]]
      for (mk in (last_k + 1):k_grid[kk]) {
        off <- sum(d[orders[ii, seq_len(mk - 1)], orders[ii, mk]])
        diag_term <- 1.0
        integral_est <- integral_est + 2 * off + diag_term

        vec <- self$z[ids[mk], ]
        tmp <- t(vec - z) %*% invh %*% (vec - z)
        cde_est <- cde_est + dnorm(sqrt(tmp), 0, 1) / det
      }
      last_k <- k_grid[kk]
      term1[kk] <- term1[kk] + integral_est / (k_grid[kk] ^ 2)
      term2[kk] <- term2[kk] + cde_est / k_grid[kk]
    }
  }

  term1 <- term1 * (2 * pi) ^ (-n_dim / 2) / (det * sqrt(2))

  return(data.frame(loss = (term1 - 2 * term2) / n_validation,
                    k = k_grid))
})
