# TODO
# - Conform to `from pymc_experimental.model_builder import ModelBuilder`
# - Deal with forecast
#   - Many of Component didn't define how to extrapolate -> May need explicit revision
# - Deal with Spline changepoint detection
#   - Privitive (evenly split -> constran prior to Laplace)
#   - Use third-party libaray (probably going to use `https://github.com/deepcharles/ruptures/`)
# - Make `dim` optional
# - Handle media transformation through `transform` component
# - For multi observation modeling - Only some of the methods seems to work
#   So I think I would make an example of it but as a separate custom model
