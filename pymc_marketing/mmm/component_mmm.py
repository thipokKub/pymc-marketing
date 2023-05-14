# TODO
# - Conform to `from pymc_experimental.model_builder import ModelBuilder`
# - Deal with forecast
#   - Many of Component didn't define how to extrapolate -> May need explicit revision
# - Deal with Spline changepoint detection
#   - Privitive (evenly split -> constran prior to Laplace)
#   - Use third-party libaray (probably going to use `https://github.com/deepcharles/ruptures/`)
# - Remove "dim" in favour of multiple observation to direct calibration from lift test
# - Handle media transformation through `transform` component
# - For Trend, and Seasonal: Make sure to decouple `t`, and `dates` (no duplicates)
#   The duplication happen after via indexing, before feeding into `Likelihood`
