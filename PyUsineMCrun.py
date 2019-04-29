import pymc3 as pm
from pymc3 import  *
print('Running on PyMC3 v{}'.format(pm.__version__))

import theano
import theano.tensor as tt

import numpy as np
import scipy.optimize

import PyProp as PP

from time import time
import sys
import datetime


# define a theano Op for our likelihood function
class Storage_Container():
    '''
    Storage in order to prevent from unnessesary Chi2 calculations
    by re-using allready calculated values.
    Most recenly used parameters are beeing kept while not regarded are beeing eliminated
    when new ones are saved
    '''
    A_Chi2 = []
    A_theta = []
    n_ = -1

    def __init__(self, n_ =5):
        self.A_Chi2 = [0 for i in range(n_)]
        self.A_theta = [[] for i in range(n_)]

    def Add(self, theta, Chi2, i = 0):
        self.A_Chi2.pop(i)
        self.A_theta.pop(i)
        self.A_Chi2.append(Chi2)
        self.A_theta.append(theta)

    def Check(self,theta):
        chi2 = 0
        try:
            i = self.A_theta.index(theta)
            chi2 = self.A_Chi2[i]
            self.Add(theta, chi2, i)
        except ValueError:
            chi2 = False
        return chi2

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
        ret_like = self.likelihood(theta, 1)
        outputs[0][0] = np.array(ret_like) # output
        
class TheanWrapperGrad(tt.Op):
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value
    
    def __init__(self, likefct, STDs = None):
        """
        loglike:
            The log-likelihood (or whatever) function we've defined
        """
        # add inputs as class attributes
        self.likelihood = likefct
        
        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood, STDs)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables
        # call the log-likelihood function
        ret_like = self.likelihood(theta, 1)
        #print(ret_like)
        outputs[0][0] = np.array(ret_like) # output
        
    def grad(self, inputs, g ):
        # the method that is automaticaly searched for when using NUTS or HMC
        theta, = inputs
        return [g[0]*self.logpgrad(theta)]
    
class LogLikeGrad(tt.Op):
    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike, STDs = None):
        """
        loglike:
            The log-likelihood (or whatever) function we've defined
        """
        # add inputs as class attributes
        self.likelihood = loglike
        self.STDs = STDs
        self.GradVerbose = 0
        

    def perform(self, node, inputs, outputs):
        theta, = inputs
        if not np.array(self.STDs).any():
            print('\n >> no deviations found, setting to 10\% ')
            self.STDs = 0.1*np.abs(theta)
        # calculate gradients
        grads = scipy.optimize.approx_fprime(theta, self.likelihood, self.STDs*0.05, self.GradVerbose) # arguments other than theta on last position 
        outputs[0][0] = grads

# - Global varible definitions

# -

class Chi2Eval():
    def __init__(self, run, InitVals, t0, log_file_name, S):
        self.run = run
        self.InitVals = InitVals
        self.t0 = t0
        self.log_file_name = log_file_name
        self.S = S
        
    def five(self, theta, IsVerb = 1):
        print(theta, IsVerb)
        return 5
    
    def HalfNegChi2(self, theta, IsVerb = 1):
        theta = list(theta)
        chi2 = 0
        InBoundary = True
        Flag = str(int(IsVerb))
        
        for par,val in zip(theta, self.InitVals):
            if (val[1] > par or par > val[2]):
                chi2 = 2.0e20
                InBoundary = False

        if InBoundary == False:
            Flag = "*"+Flag
                
        stock = self.S.Check(theta)
        if (stock):
            Flag += "X"
            chi2 = stock
        elif InBoundary:
            Flag += " "
            chi2 = self.run.PyChi2(theta)
            self.S.Add(theta,chi2)

        result = (-0.5*chi2)
        if (bool(IsVerb)):
            f = open(self.log_file_name,'a+')
            f.write("{:10}  {:15}  {:6}  ".format(round(time()-self.t0,3), round(chi2,3),  Flag))
            f.write('[ ' + ' '.join(["{:10},".format(round(p,6)) for p in theta]) + '  ] \n')
            f.close()

        """
            if (result < -900000.0 and InBoundary):
                f.write('# - - Warning: Class ist beeing reinitialized due probable crash - -\n')
                global ParFile
                run.PySetClass(ParFile, 0, "OUT")
        """

        return result

def main():
    # GENERAL INITIALIZATION
    now = datetime.datetime.now()
    t0 = time()
    log_file_name = "logger_" + datetime.datetime.now().strftime("%H_%M_%S")
    S = Storage_Container()
    
    # LOADING USINE CONFIGURATION
    ParFile = sys.argv[1]
    print ('\n >> Loading configuration from {}'.format(ParFile))

    run = PP.PyRunPropagation()
    run.PySetLogFile("run.log")
    run.PySetClass(ParFile, 1, "OUT")

    InitVals = run.PyGetInitVals()
    VarNames = run.PyGetFreeParNames()
    FixedVarNames = run.PyGetFixedParNames()
    for i in range(len(VarNames)):
        if bool(InitVals[i][4]):  #This position is the bool output of IsLogSampling
            VarNames[i] = "LOG10_" + VarNames[i]

    # SORTING OUT FIXED VARIABLES
    print ('\n >> Not regarding the following FIXED parameters:')
    for name in FixedVarNames:
        print('{:25}'.format(name))


    # SETTING PYMC3 PARAMETERS
    basic_model = pm.Model()
    
    CE = Chi2Eval(run, InitVals, t0, log_file_name, S)
    ext_fct = TheanWrapper(CE.HalfNegChi2)

    with basic_model:
        ProScale = 1. # Scale for the sampling normal

        # Priors for unknown model parameters
        Priors = []
        print ('\n >> Found {} free parameters, using the following values:'.format(len(VarNames)))
        for name, vals in zip(VarNames, InitVals):
            print('{:25}  [{:10}, {:15} +- {:10} ,{:10}]'.format(name, vals[1], vals[0], vals[3], vals[2]))
            P = pm.Normal(name, mu=vals[0], sd=vals[3]*1.5)
            Priors.append(P)

        theta = tt.as_tensor_variable(Priors)

        # Likelihood (sampling distribution) of observations
        likelihood = pm.DensityDist('likelihood',  lambda v: ext_fct(v), observed={'v': theta})

    # RUNNING PYMC3
    print('\n >> Saving calculation steps in {}'.format(log_file_name) )
    f = open(log_file_name,'w+')

    N_run  = 500
    N_tune = 50
    N_chains = 3
    N_cores = 1
    IsProgressbar = 0
    print ("\n >> len(sys.argv) = ",len(sys.argv))
    if len(sys.argv) >= 3:
        N_run = int(sys.argv[2])
    if len(sys.argv) >= 4:
        N_tune = int(sys.argv[3])
    if len(sys.argv) >= 5:
        N_chains = int(sys.argv[4])
    if len(sys.argv) >= 6:
        IsProgressbar = int(sys.argv[5])
    if len(sys.argv) >= 7:
        N_cores = int(sys.argv[6])


    print ('\n >> using configuration N_run = {}, N_tune = {}, N_chains = {}, N_cores = {}\n'.format(N_run,N_tune,N_chains,N_cores))

    CE.S = Storage_Container(2*N_chains*len(VarNames))

    with basic_model:

        try:
            cov = np.loadtxt(sys.argv[7] , delimiter = ',' )
            print(cov.shape)
            print("\n >> Covariance matrix {} found".format(sys.argv[7]))
            S_ = cov
            print("\n >> Using Cov Matrix  {} ".format(sys.argv[7]))
        except:
            print("\n >> Not using Cov Matrix")
            S_ = np.diag([(ProScale*var[3])**2 for var in InitVals])
        step = pm.Metropolis(S = S_, proposal_dist = pm.MultivariateNormalProposal )
        
        print("\n >> Starting sampler")
        CE.t0 = time()
        trace = pm.sample(N_run,
                        step = step,
                        progressbar = IsProgressbar,
                        chains = N_chains,
                        cores = N_cores,
                        tune = N_tune ,
                        parallelize=True)

    f.close()

    time_stamp = datetime.datetime.now().strftime("%d.%m.%Y_%H:%M")
    for i_c in range(N_chains):
        output_filename = 'result_C{}_{}_{}_'.format(i_c, N_run,N_tune) + time_stamp
        print ('saving chain {} as numpy array in {}'.format(i_c, output_filename))

        post_data = np.array([trace.get_values(VarNames[j_v], chains = i_c) for j_v in range(len(VarNames))]).T
        np.savetxt(output_filename, post_data, delimiter=',', header = str(VarNames))

def Gen_Cov():
    print("\n >> Generating covariance matrix from data")
    result_file = sys.argv[2]
    print(" >> Loading results from {}".format(result_file))

    if len(sys.argv) > 5:
        cov_file = sys.argv[4]
    else:
        cov_file = "Cov_" + result_file.split(":")[0].split("_")[1] +  '_' + sys.argv[1].split(".par")[0]

    post_data = np.loadtxt(result_file, delimiter = ',')
    covariance = np.cov(post_data.T)

    with open(result_file,'r') as r_file:
        VarNames = r_file.readline()[2:]

    print(" >> Sving covariance matrix in {}".format(cov_file))
    np.savetxt(cov_file, covariance, delimiter = ', ', header = VarNames)



if __name__ == "__main__":
    # execute only if run as a script
    try:
        int(sys.argv[2])
        main()
    except:
        Gen_Cov()
