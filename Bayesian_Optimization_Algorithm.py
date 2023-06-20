from pymoo.core.algorithm import Algorithm
from sklearn.gaussian_process import GaussianProcessRegressor
from numpy.random import uniform
from numpy import argmax
from numpy import vstack
from pymoo.core.initialization import Initialization
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.repair import NoRepair
from pymoo.core.population import Population
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
                 sample_size = 100, # The number of initial samples to be generated from the problems and 
                                    #used in the acquisition function
                 repair=NoRepair(),
                 **kwargs):
        
        super().__init__(**kwargs)

        # the surrogate model to be used in the optimization
        self.model = None
        #the data set to be used in the optimization
        self.sample_size = sample_size
        self.Y = None
        self.initialization = Initialization(FloatRandomSampling())
        self.repair = repair

    
        
    def _setup(self, problem, **kwargs):
        self.model = GaussianProcessRegressor()
    
    def _initialize_infill(self):
        initial_pop = self.initialization.do(self.problem, self.sample_size, algorithm=self)
        self.Y = initial_pop.get("F")
        return initial_pop
    
    def _infill(self):
        X_new = self.optimize_acquisition_function()
        off = Population.new(X=X_new)
        return off
    
    def _advance(self, infills=None, **kwargs):
        return self.update_data_set()
    
    def _finalize(self):
        return super()._finalize()
    
    # define the surrogate function
    def surrogate_function(self,X):
        return self.model.predict(X, return_std=True)

    # sample from the search space
    def sample(self):
        if self.problem.has_bounds():
            xl, xu = self.problem.bounds()
            X = uniform(xl, xu, size=(self.sample_size, self.problem.n_var))
            return X
        
    
    # define the acquisition function   
    def optimize_acquisition_function(self):
        if self.problem.has_bounds():
            # Randomly draw 1000 sample points from the search space
            X_sample = self.sample()
            # calculate the current best surrogate score 
            mu, std = self.surrogate_function(self.pop.get("X"))
            # calculate mean and std of the sample in the surrogate function
            mu_sample, std_sample = self.surrogate_function(X_sample)
            current_best = max(mu)
            # calculate the probability of improvement
            pi = (mu_sample - current_best) / std_sample
            # find the index of the greatest scores
            index = argmax(pi)
            return X_sample[index, :]
    
    # generate initial samples and evaluate them
    def generate_initial_samples(self):
        X = self.sample()
        Y = self.problem.evaluate(X)
        return X, Y
    
    # update the model with new samples
    def update_model(self, X, Y):
        self.model.fit(X, Y)

    # optimize the acquisition function
    def update_data_set(self):
        # find the best point
        x = self.optimize_acquisition_function(self.problem)
        # calculate the new target value
        y = self.problem.evaluate(x)
        # update the 
        self.X = vstack((self.pop.get("X"), x))
        self.Y = vstack((self.pop.get("Y"), y))
        self.update_model(self.X, self.Y)
    