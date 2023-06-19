from pymoo.core.algorithm import Algorithm




#The optimization process itself is as follows:

# 1. Define the black box function f(x), the acquisition function a(x) 
# and the search space of the parameter x.

# 2. Generate some initial values of x randomly, and measure the corresponding outputs from f(x).
# 3. Fit a Gaussian process model m(X, y) onto X = x and y = f(x). 
# In other words, m(X, y) serves as a surrogate model for f(x)!

# 4. The acquisition function a(x) then uses m(X, y) to generate new values of x as follows. 
# Use m(X, y) to predict how f(x) varies with x. 
# The value of x which leads to the largest predicted value in m(X, y) is then suggested 
# as the next sample of x to evaluate with f(x).

# 5. Repeat the optimization process in steps 3 and 4 until we finally 
# get a value of x that leads to the global optimum of f(x). 
# Note that all historical values of x and f(x) should be used to train 
# the Gaussian process model m(X, y) in the next iteration — as the number 
# of data points increases, m(X, y) becomes better at predicting the optimum of f(x).

class BayesianOptimiztion(Algorithm):
    def __init__(self,
                 pop_size=25,
                 sampling=LHS(),
                 w=0.9,
                 c1=2.0,
                 c2=2.0,
                 adaptive=True,
                 initial_velocity="random",
                 max_velocity_rate=0.20,
                 pertube_best=True,
                 repair=NoRepair(),
                 output=PSOFuzzyOutput(),
                 **kwargs):
