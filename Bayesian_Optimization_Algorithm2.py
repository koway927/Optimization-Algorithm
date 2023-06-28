from pymoo.core.algorithm import Algorithm
from sklearn.gaussian_process import GaussianProcessRegressor
from numpy.random import uniform
from numpy import argmax
from numpy import sum
from numpy import vstack
from pymoo.core.initialization import Initialization
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.repair import NoRepair
from pymoo.core.population import Population
from scipy.stats import norm

#The optimization process itself is as follows:


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
        self.X_new = None



    def _setup(self, problem, **kwargs):
        self.model = GaussianProcessRegressor()

    def _initialize_infill(self):
        initial_pop = self.initialization.do(self.problem, self.sample_size, algorithm=self)
        #print("initial_pop X",initial_pop.get("X"))
        #print("initial_pop F",self.problem.evaluate(initial_pop.get("X")))
        initial_X = initial_pop.get("X")
        initial_Y = self.problem.evaluate(initial_pop.get("X"))
        self.update_model(initial_X,initial_Y)
        return initial_pop

    def _infill(self):
        self.X_new = self.optimize_acquisition_function()
        off = Population.new(X=vstack((self.pop.get("X"), self.X_new)))
        return off


    def _advance(self, infills=None, **kwargs):
        self.update_data_set()

    def _finalize(self):
        return super()._finalize()
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
            # Randomly draw 100 sample points from the search space
            X_sample = self.sample()
            # calculate the current best surrogate score 
            mu, _ = self.surrogate_function(self.pop.get("X"))
            #print("mu",mu)
            # calculate mean and std of the sample in the surrogate function
            mu_sample, std_sample = self.surrogate_function(X_sample)
            #print("mu_sample",mu_sample)
            current_best = max(mu)
            print("mu",mu)
            # calculate the probability of improvement
            probability_of_improvement = mu_sample - current_best / (std_sample + 1**-16)
            # calculate the expected improvement
            #expected_improvement = (mu - current_best) * probability_of_improvement + std_sample * norm.pdf(probability_of_improvement)
            # find the index of the greatest scores
            #print("pi",pi)
            sum_probability_of_improvement = sum(probability_of_improvement, axis = 1)
            index = argmax(sum_probability_of_improvement)
            #print("probability_of_improvement",probability_of_improvement)
            return X_sample[index, :]

    def get_next_points(self):
        return self.optimize_acquisition_function()
    # update the model with new samples
    def update_model(self, X, Y):
        self.model.fit(X, Y)

    # optimize the acquisition function
    def update_data_set(self):
        y = self.problem.evaluate(self.X_new)
        # update the 
        updated_X = vstack((self.pop.get("X"), self.X_new))
        #print("updated_X",updated_X)
        updated_Y = vstack((self.pop.get("F"), y))
        self.update_model(updated_X, updated_Y)