import pymc3 as pm
from pymc3 import  *
print('Running on PyMC3 v{}'.format(pm.__version__))

import theano
import theano.tensor as tt

import numpy as np

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
        ret_like = self.likelihood(theta)
        outputs[0][0] = np.array(ret_like) # output

# - Global varible definitions
t0 = time()

now = datetime.datetime.now()

log_file_name = "logger_" + datetime.datetime.now().strftime("%H_%M_%S")

run = PP.PyRunPropagation()
InitVals = []

S = Storage_Container()
# -

def loglike_chi2(theta):
    theta = list(theta)
    global run
    global InitVals
    global t0
    global log_file_name
    global S
    chi2 = 0

    InBoundary = True
    for par,val in zip(theta,InitVals):
        if (val[1] > par or par > val[2]):
            chi2 = 2.0e20
            InBoundary = False

    Flag = str(int(InBoundary))

    stock = S.Check(theta)
    if (stock):
        Flag += "X"
        chi2 = stock
    elif InBoundary:
        Flag += " "
        chi2 = run.PyChi2(theta)
        S.Add(theta,chi2)

    result = (-0.5*chi2)

    f = open(log_file_name,'a+')
    f.write("{:10}  {:15}  {:6}  ".format(round(time()-t0,3), round(chi2,3),  Flag))
    f.write('[ ' + ' '.join(["{:10},".format(round(p,6)) for p in theta]) + '  ] \n')

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
    print ('\n >> Loading configuration from {}'.format(ParFile))

    global run
    run.PySetLogFile("run.log")
    run.PySetClass(ParFile, 1, "OUT")

    global InitVals
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

    ext_fct = TheanWrapper(loglike_chi2)

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
    global log_file_name
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

    global S
    S = Storage_Container(5*N_chains)

    with basic_model:
        step = pm.Metropolis(S = np.diag([(ProScale*var[3])**2 for var in InitVals]))
        global t0
        t0 = time()
        trace = pm.sample(N_run,
                        step = step,
                        progressbar = IsProgressbar,
                        chains = N_chains,
                        cores = N_cores,
                        tune = N_tune  )

    f.close()

    output_filename = 'result_{}:{}:{}_'.format(N_run*N_chains,N_tune,N_chains) + datetime.datetime.now().strftime("%d.%m.%Y_%H:%M")
    print ('saving results as numpy array in {}'.format(output_filename))

    post_data = np.array([trace[VarNames[i]] for i in range(len(VarNames))]).T
    np.savetxt(output_filename, post_data, delimiter=',', header = '# '+ str(VarNames))






if __name__ == "__main__":
    # execute only if run as a script
    main()
