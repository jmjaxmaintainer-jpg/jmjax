test_that("build_long_arrays pads correctly and preserves values", {
  data_long <- data.frame(
    id = c(1, 1, 1, 2, 2, 3),
    time = c(0, 1, 2, 0, 1, 0),
    y = c(1.1, 2.2, 3.3, 4.4, 5.5, 6.6)
  )

  arr <- build_long_arrays(y ~ time, data_long, id_var = "id")

  expect_equal(dim(arr$X_long), c(3, 3, 2))  # 3 subjects, max_obs=3, intercept+time
  expect_equal(arr$n_obs, c(3L, 2L, 1L))
  expect_equal(arr$subj_ids, c(1, 2, 3))

  # Subject 1's three observations, in order, should match the raw data
  expect_equal(arr$y_long[1, 1:3], c(1.1, 2.2, 3.3))
  expect_equal(arr$X_long[1, 1:3, 2], c(0, 1, 2))  # time column

  # Padded slots for subject 3 (only 1 obs) should be zero, not garbage
  expect_equal(arr$y_long[3, 2:3], c(0, 0))

  # Intercept column should be all 1s wherever populated
  expect_equal(arr$X_long[1, 1:3, 1], c(1, 1, 1))
})

test_that("build_long_arrays handles subject ids that are non-contiguous / out of order", {
  data_long <- data.frame(
    id = c(5, 5, 2, 2, 2),
    time = c(0, 1, 0, 1, 2),
    y = c(1, 2, 3, 4, 5)
  )
  arr <- build_long_arrays(y ~ time, data_long, id_var = "id")

  expect_equal(arr$subj_ids, c(2, 5))  # sorted, defines row order
  expect_equal(arr$n_obs, c(3L, 2L))    # subject 2 (row 1) has 3 obs, subject 5 (row 2) has 2
  expect_equal(arr$y_long[1, 1:3], c(3, 4, 5))
  expect_equal(arr$y_long[2, 1:2], c(1, 2))
})

test_that("build_surv_arrays reorders data_surv to match subj_ids and builds correct quadrature grid", {
  data_long <- data.frame(id = c(1, 1, 2), time = c(0, 1, 0), y = c(1, 2, 3))
  data_surv <- data.frame(id = c(2, 1), time = c(3.0, 5.0), event = c(1, 0))  # deliberately out of order

  arr_long <- build_long_arrays(y ~ time, data_long, id_var = "id")
  arr_surv <- build_surv_arrays(survival::Surv(time, event) ~ 1, data_surv,
                                 arr_long$subj_ids, id_var = "id")

  # subj_ids = c(1, 2), so T_surv/event should be reordered to match
  expect_equal(arr_surv$T_surv, c(5.0, 3.0))
  expect_equal(arr_surv$event, c(0, 1))

  expect_equal(dim(arr_surv$t_quad), c(2, 10))
  expect_true(all(arr_surv$t_quad[1, ] <= 5.0))
  expect_true(all(arr_surv$t_quad[2, ] <= 3.0))
  expect_equal(sum(arr_surv$gk_weights), 1.0, tolerance = 1e-3)
})

test_that("build_time_design evaluates the formula correctly at arbitrary times", {
  X <- build_time_design(y ~ time, "time", c(0, 1, 2.5))
  expect_equal(unname(X[, "time"]), c(0, 1, 2.5))
  expect_equal(unname(X[, "(Intercept)"]), c(1, 1, 1))

  # Matrix input (quadrature grid): [N_sub, n_quad] -> [N_sub, n_quad, p]
  t_quad <- matrix(c(0, 1, 2, 3), nrow = 2, ncol = 2)
  X_arr <- build_time_design(y ~ time, "time", t_quad)
  expect_equal(dim(X_arr), c(2, 2, 2))
  expect_equal(X_arr[1, , "time"], t_quad[1, ])
  expect_equal(X_arr[, , "(Intercept)"], matrix(1, 2, 2))
})
