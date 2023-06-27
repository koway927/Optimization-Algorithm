from pymoo.core.algorithm import Algorithm
from sklearn.gaussian_process import GaussianProcessRegressor
from numpy.random import uniform
from numpy import argmax
from numpy import argmin
from numpy import vstack
from pymoo.core.initialization import Initialization
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.repair import NoRepair
from pymoo.core.population import Population
from scipy.stats import norm
from scipy.optimize import minimize as scipy_minimize
from numpy import array

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
                 sample_size = 10, # The number of samples to be generated from the problems and 
                                    #used in the acquisition function
                 sampling=FloatRandomSampling(),        
                 repair=NoRepair(),
                 model = GaussianProcessRegressor(),# the surrogate model to be used in the optimization
                 **kwargs):
        
        super().__init__(**kwargs)

        self.sample_size = sample_size
        self.initialization = Initialization(sampling)
        self.model = model
        self.repair = repair
        #the data set to be used in the optimization
        self.data_set_X = None
        self.data_set_Y = None

    
        
    def _setup(self, problem, **kwargs):
        self.model = GaussianProcessRegressor()
    
    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.sample_size, algorithm=self)

    def _initialize_advance(self, infills=None, **kwargs):
        self.data_set_X = self.pop.get("X")
        self.data_set_Y = self.problem.evaluate(self.data_set_X)
        self.update_model(self.data_set_X,self.data_set_Y)

        super()._initialize_advance(infills=infills, **kwargs)

    def _infill(self):
        X_new = self.optimize_acquisition_function()
        self.update_data_set(X_new)
        self.update_model(self.data_set_X,self.data_set_Y)

        off = Population.new(X=self.data_set_X)
        self.pop = off
        return off
        
    
    def _advance(self, infills=None, **kwargs):
        self.update_model(self.data_set_X,self.data_set_Y)
        print("n_eval",self.evaluator.n_eval)
    
    def _finalize(self):
        return super()._finalize()
    
    # define the surrogate function
    def surrogate_function(self,X):
        return self.model.predict(X, return_std=True)

    # sample from the search space
    def sample(self, sample_size = 10):
        if self.problem.has_bounds():
            xl, xu = self.problem.bounds()
            X = uniform(xl, xu, size=(sample_size, self.problem.n_var))
            
            return X
        
    
    # define the acquisition function
    def acquisition_function(self, X_sample):
        # calculate the current best surrogate score 
        predicted_y, _ = self.surrogate_function(self.data_set_X)
        current_best = min(predicted_y)
        
        # calculate mean and std of the sample in the surrogate function
        mu_sample, std_sample = self.surrogate_function(X_sample.reshape(-1, self.problem.n_var) )
        mu_sample = mu_sample[:,0]

        # calculate the probability of improvement
        probability_of_improvement = self.find_probability_of_improvement(mu_sample, std_sample, current_best)

        # calculate the expected improvement
        return ((current_best - mu_sample) * probability_of_improvement + std_sample * norm.pdf((current_best - mu_sample) / (std_sample + 1**-16)))

    # find the next point to sample
    def optimize_acquisition_function(self):
        fun_negative_acquisition = lambda X_sample: -1.0 * self.acquisition_function(X_sample)
        xl, xu = self.problem.bounds()
        initials = self.sample()
        list_next_point = []

        bounds_list = []
        for l, u in zip (xl, xu):
            bounds_list.append((l,u))

        # attempt to find the minimum of the acquisition function
        for arr_initial in initials:
            next_point = scipy_minimize(fun_negative_acquisition,
                                  x0=arr_initial,
                                  bounds = bounds_list,
                                  method="L-BFGS-B",
                                  options={'disp': False}
                                )
            next_point_x = next_point.x
            list_next_point.append(next_point_x)
                
        next_points = array(list_next_point)
        acquisition_value = fun_negative_acquisition(next_points)
        index_best = argmin(acquisition_value)
        return next_points[index_best, :]
    
    def find_probability_of_improvement(self, mean, std, current_best):
        return norm.cdf((current_best - mean) / (std + 1**-16))
    
    # update the model with new samples
    def update_model(self, X, Y):
        self.model.fit(X, Y)

    # optimize the acquisition function
    def update_data_set(self, X_new):
        # update the model with new samples
        self.data_set_X = vstack((self.data_set_X, X_new))
        self.data_set_Y = vstack((self.data_set_Y, self.problem.evaluate(X_new)))
    