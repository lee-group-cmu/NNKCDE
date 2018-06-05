NNKCDE: Nearest Neighbor Conditional Density Estimation
===
[![Build Status](https://travis-ci.org/tpospisi/NNKCDE.svg?branch=master)](https://travis-ci.org/tpospisi/NNKCDE)

Estimates nearest neighbor kernel conditional densities tuned using
CDE loss.

Installation
---
Use the `devtools` package to install from Github

```r
devtools::install_github("tpospisi/NNKCDE")
```

Usage
---

```r
fit <- NNKCDE$new(x_train, z_train)
fit$estimate_error(x_validation, z_validation)
fit$tune(x_validation, z_validation, k_grid = c(5, 10, 15, 20))
fit$predict(x_test, z_grid, h = 0.1)
```