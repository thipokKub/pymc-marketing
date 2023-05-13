import pymc as pm

class ComponentBase(pm.Model):
    pass

# Issues
# - Didn't define how set_data will interact with sub modules (probably going to be recursive)
# - Prediction behaviour might be different (Especially for GP, need to look at HSGP prediction method)
