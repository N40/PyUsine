import pymc3 as pm
from pymc3 import  *
print('Running on PyMC3 v{}'.format(pm.__version__))

import theano
import theano.tensor as tt

import numpy as np

import PyProp as PP

from time import time
t0 = time()

import datetime
now = datetime.datetime.now()

import sys

run = PP.PyRunPropagation()
InitVals = []

# define a theano Op for our likelihood function
class TheanWrapper(tt.Op):
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value

    def __init__(self, likefct):
        """
        loglike:
            The log-likelihood (or whatever) function we've defined
        """
        # add inputs as class attributes
        self.likelihood = likefct

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables
        # call the log-likelihood function
        ret_like = self.likelihood(theta)
        outputs[0][0] = np.array(ret_like) # output

def loglike_chi2(theta):
    global run
    global InitVals
    global t0

    InBoundary = True
    for par,val in zip(theta,InitVals):
        if (val[1] > par or par > val[2]):
            chi2 = 2.0e40
            InBoundary = False

#        if(chi2 > 1000):
#        	  print(val[1], par, val[2], InBoundary)

    if InBoundary:
        chi2 = run.PyChi2(theta)

    result = (-0.5*chi2)

    f = open("logger.txt",'a+')
    f.write("{:15}  {:15}  {:15}   {:8}   {}\n".format(round(time()-t0,3), round(chi2,3),  round(result,3), InBoundary, theta))

    if (result < -900000.0 and InBoundary):
        f.write('# - - Warning: Class ist beeing reinitialized due probable crash - -\n')
        global ParFile
        run.PySetClass(ParFile, 0, "OUT")

    f.close()

    return result

def main():
    # LOADING USINE CONFIGURATION
    global ParFile
    ParFile = sys.argv[1]
    print ('loading configuration from {}'.format(ParFile))

    global run
    run.PySetLogFile("run.log")
    run.PySetClass(ParFile, 1, "OUT")

    global InitVals
    InitVals = run.PyGetInitVals()
    VarNames = run.PyGetFreeParNames().split(",")
    for i in range(len(VarNames)):
        if bool(InitVals[i][4]):  #This position is the bool output of IsLogSampling
            VarNames[i] = "LOG10_" + VarNames[i]


    # SETTING PYMC3 PARAMETERS
    basic_model = pm.Model()

    ext_fct = TheanWrapper(loglike_chi2)

    with basic_model:
        ProScale = 2. # Scale for the sampling normal

        # Priors for unknown model parameters
        Priors = []
        print ('found {} free parameters, using the following priors:'.format(len(VarNames)))
        for name, vals in zip(VarNames, InitVals):
            if (vals[3] == 0.):
                VarNames.remove(name)
                InitVals.remove(vals)
                continue
            print('{:10}{:10}{:10}'.format(name, vals[0], vals[3]*ProScale))
            P = pm.Normal(name, mu=vals[0], sd=vals[3]*1.5)
            Priors.append(P)

        theta = tt.as_tensor_variable(Priors)

        # Likelihood (sampling distribution) of observations
        likelihood = pm.DensityDist('likelihood',  lambda v: ext_fct(v), observed={'v': theta})

    # RUNNING PYMC3
    f = open("logger.txt",'w+')

    N_run  = 500
    N_tune = 50
    N_chains = 3
    if sys.argv[2]:
        N_run = int(sys.argv[2])
    if sys.argv[3]:
        N_tune = int(sys.argv[3])
    if sys.argv[4]:
        N_chains = int(sys.argv[4])

    IsProgressbar = int(sys.argv[5])



    print ('\n using configuration N_run = {}, N_tune = {}, N_chains = {}\n'.format(N_run,N_tune,N_chains))
    print (IsProgressbar)

    with basic_model:
        # draw 500 posterior samples
        step = pm.Metropolis(S = ProScale* np.diag([var[3] for var in InitVals]))
        global t0
        t0 = time()
        trace = pm.sample(N_run,
                        step = step,
                        progressbar = IsProgressbar,
                        chains = N_chains,
                        tune = N_tune)

    f.close()

    output_filename = 'result_{}:{}:{}_'.format(N_run*N_chains,N_tune,N_chains) + datetime.datetime.now().strftime("%d.%m.%Y_%H:%M")
    print ('saving results as numpy array in {}'.format(output_filename))

    post_data = np.array([trace[VarNames[i]] for i in range(len(VarNames))]).T
    np.savetxt(output_filename, post_data, delimiter=',', header = '#'+ str(VarNames))

if __name__ == "__main__":
    # execute only if run as a script
    main()
