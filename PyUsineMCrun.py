import numpy as np
import scipy.optimize

import pymc3 as pm
from pymc3 import  *

import theano
import theano.tensor as tt

import PyProp as PP

from time import time
import sys
import datetime

import argparse
import os

class Storage_Container():
    '''
    Cache designed to prevent unnessesary Chi2 calculations by re-using
    allready calculated values. For each new value, the least recently
    called one is overwritten.
    '''
    A_Chi2 = []
    A_theta = []
    n_ = -1

    def __init__(self, n_ =5):
        '''
        n_ :
            size of the cache
        '''
        self.n_ = n_
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

class TheanWrapperGrad(tt.Op):
    """
    A generic Class, necessary in order to call custom log-probability function
    (blame the internal PyMC3 design for this!)
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value

    def __init__(self, likefct, STDs = None):
        """
        loglike:
            The log-likelihood (or whatever) function we've defined
        """
        self.likelihood = likefct

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood, STDs)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling for a Chi2 value
        theta, = inputs
        ret_like = self.likelihood(theta, 1)
        outputs[0][0] = np.array(ret_like)

    def grad(self, inputs, g ):
        # This is executed if the gradient of the log-likelihood function is needed
        # it is automaticaly searched for when using NUTS or HMC; not needed for Metropolis
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
        self.likelihood = loglike
        # Vector of the uncertainties
        self.STDs = STDs
        # This defines, whether or not to write the gradient calculations in the log-file
        # often overwritten externaly
        self.GradVerbose = 0

    def perform(self, node, inputs, outputs):
        theta, = inputs
        if not np.array(self.STDs).any():
            print('\n >> no deviations found, setting to 10\% ')
            self.STDs = 0.1*np.abs(theta)
        # calculation of gradients using steps in each direction (5% of init. uncertainty)
        # arguments other than theta on position 3 and beyond
        grads = scipy.optimize.approx_fprime(theta, self.likelihood, self.STDs*0.05, self.GradVerbose)
        outputs[0][0] = grads

class PyUsine():
    """
    Class in order to mediate the execution and documentation of the Chi2 function calls
    """
    def __init__(self, ParFile = None):
        self.PRP = PP.PyRunPropagation()
        self.t0 = time()
        self.log_file_name = 'logger'
        self.S = Storage_Container(5)
        self.Cov = None

        if ParFile:
            self.InitPar(ParFile)


    def InitPar(self, ParFile, log_file_name = None, Theta0 = None):
        """
        Loading Usine Configuration and setting all intern parameters correspondingly
        ParFile         : USINE init file
        log_file_name   : log file for the Chi2 calls
        Theta0          : Starting point or point to sample around the starting points
        """

        if log_file_name:
            self.log_file_name = log_file_name
        print (" >> Saving Chi2 calculations in '{}' ".format(self.log_file_name))
        open(self.log_file_name,'w+').close() # wiping the logfile

        self.ParFile = ParFile
        print (" >> Loading USINE configuration and parameters from '{}' ".format(self.ParFile))

        self.PRP.PySetLogFile("run.log") # this is the USINE log file, not the MCMC one
        self.PRP.PySetClass(self.ParFile, 1, "OUT") # "OUT" is a dummy location without use, needed for USINE

        self.InitVals = self.PRP.PyGetInitVals()            # Parameter-wise list of [start, low_bound, up_bound, std]
        self.VarNames = self.PRP.PyGetFreeParNames()        #
        self.FixedVarNames = self.PRP.PyGetFixedParNames()

        if Theta0:
            for i in range(len(self.VarNames)):
                self.InitVals[i][0] = Theta0[i]

        self.Theta0 = [ V[0]           for V in self.InitVals]    # Initial parameters
        self.STDs   = [ V[3]           for V in self.InitVals]    # Initial uncertainties
        self.Bounds = [ [ V[1], V[2] ] for V in self.InitVals]    # Parameter bounds

        self.S.__init__(5 * len(self.VarNames))            # updating the cache size


        print (' >> Not regarding the following FIXED parameters:')
        for name in self.FixedVarNames:
            print('    {:25}'.format(name))

        print (' >> Using {} free parameters with the following values:'.format(len(self.VarNames)))
        for i_V, name, T, S, B in zip(range(100),self.VarNames, self.Theta0, self.STDs, self.Bounds):
            print(' {:2} {:30}  [{:7.3f},   {:7.3f} +- {:6.3f}   ,{:7.3f}]'.format(i_V,name[:27], B[0], T, S, B[1] ))


    def SetCovMatrix(self, **kwargs):
        '''
        Setting covariance matrix of the parameters
            Cov     : file location of a comma-separated covariance matrix
            Scale   : The fraction of init. uncertainties to use for a diagonal matrix
        '''
        try:
            self.Cov = np.loadtxt(kwargs["Cov"] , delimiter = ',' )
            print(" >> Valid Covariance matrix {} found".format(kwargs["Cov"]))
        except:
            Scale = kwargs.get("Scale", 1.0)
            print(#" >> No valid Covariance matric found \n",
                  " >> Creating diagnonal covariance matrix with scale {}".format(Scale))
            self.Cov = np.diag([(Scale*var[3])**2 for var in self.InitVals])


    def HalfNegChi2(self, theta, Option = 1):
        return self.Chi2(theta, Option)*(-0.5)

    def Chi2(self, theta, Option = 1):
        '''
        This function returns -0.5 Chi^2(theta), the logarithm of the (gaussian) posterior probability

        Option:
            1 : Normal function call
            0 : Do not write to log-file
            2 : Normal function call (meant for gradient calculation)
            3 : No penalty for out-of-bound parameters (meant for gradient calculation)
        theta:
            list or array of parameters to probe
        '''
        theta = list(theta)
        InBoundary = True
        Flag = str(int(Option)) # This indicates in the log file the type of Chi2 call

        for par, (B_lo, B_up) in zip(theta, self.Bounds):
            # The bounds, defined in the init file, are in position 1,2 (lower,upper) for each parameter
            if ((B_lo > par or par > B_up) and Option != 3):
                chi2 = 2.0e20
                InBoundary = False
                Flag = "*"+Flag
                break

        # Checking if parameter was recently called to avoid USINE-call
        stock = self.S.Check(theta)
        if (stock):
            Flag += "X"
            chi2 = stock
        elif InBoundary:
            Flag += " "
            chi2 = self.PRP.PyChi2(theta) # Calculate the Chi2 with USINE
            self.S.Add(theta,chi2)        # Cach resent Chi2 call

        # Writing the Time, Chi2, Flag and theta to the log-file
        if (bool(Option)):
            lf = open(self.log_file_name,'a+')
            lf.write("{:10.3f}  {:12}   {:3}  ".format(time()-self.t0, (round(chi2,4)),  Flag))
            lf.write('[ ' + ' '.join(["{:10.4f},".format( p ) for p in theta]) + '  ] \n')
            lf.close()

        return chi2

# PyMC3 Routine
class MCU(PyUsine):
    """
    This class contains all the steps and routines for generating MCMC Fits
    """
    def __init__(self, **kwargs):
        self.basic_model = None
        super().__init__()


    def Gen_Start_Points(self, sigma = 0.1):
        '''
        Calculate a dictionary of points of departure, sampled with respect to Theta0
        sigma :
            Sampling scale for the points (relative to init. uncertainties)
        '''
        start_points = []
        for i_C in range(self.Custom_sample_args['chains']):
            start = dict()
            for V,S,T,P in zip(self.VarNames,self.STDs ,self.Theta0, self.InitVals ):
                new_V = T + np.random.normal()*S*sigma
                while (new_V < P[1] or new_V > P[2]):
                    new_V = T + np.random.normal()*S
                start.update({V:new_V})
            start_points.append(start)
        return start_points


    def InitPyMCBasic(self, **kwargs):
        '''
        PyMC3 initialisation: Free parameters, Log-Likelihood function
        '''
        # Check if PyUsine is properly initialised
        if not self.VarNames:
            self.InitPar(kwargs["ParFile"])

        self.basic_model = pm.Model()
        # Setting the extern blackbox function
        ext_fct = TheanWrapperGrad(self.HalfNegChi2, np.array(self.STDs))
        ext_fct.logpgrad.GradVerbose = 2  # =2: gradient calculation steps are shown in logfile

        with self.basic_model:
            # Setting up the Parameters for MCMC to use (no priors here!)
            Priors = []
            print (' >> Setting up PyMC3 with the {} free parameters and flat priors'.format(len(self.VarNames)))
            for name, B in zip(self.VarNames, self.Bounds):
                P = pm.Uniform(name, lower=B[0], upper=B[1])
                Priors.append(P)

            theta = tt.as_tensor_variable(Priors)
            # Likelihood (sampling distribution) of observations. Specified as log_like
            likelihood = pm.DensityDist('likelihood',  lambda v: ext_fct(v), observed={'v': theta})

    def InitPyMCSampling(self, **kwargs):
        '''
        PyMC3 initialisation: Sampler
            N_tune :
            N_chains :
            N_cores :
            IsProgressbar :
            Sampler_Name :
        '''
        #Checking if all the necessary stuff is loaded
        if not self.basic_model:
            self.InitPyMCBasic()
        try: self.Cov[0][0]
        except TypeError: self.SetCovMatrix(Scale = 1.2)


        # Further initialisation
        with self.basic_model:
            Sampler_Name    = kwargs.get("Sampler_Name","Metropolis")
            N_tune          = kwargs.get("N_tune" , 0)
            N_chains        = kwargs.get("N_chains" , 1)
            N_cores         = kwargs.get("N_cores" , min(4,N_chains) )
            IsProgressbar   = kwargs.get("IsProgressbar" , 1)
            print ('\n >> using configuration :  {:12}, N_tune = {}, N_chains = {}, N_cores = {}'.format(Sampler_Name,N_tune,N_chains,N_cores))

            self.S.__init__(3*N_chains*len(self.VarNames)) # updating the cach size

            # Setting up the samplers
            #   Calling S = self.Cov[::-1,::-1] is a neccessary hack in order to avoid a problem in the PyMC3 code:
            #   The order of the variables is inverted (by accident?) durint the BlockStep().__init__() (see terminal promts)
            if Sampler_Name == "DEMetropolis":
                step = pm.DEMetropolis(S = self.Cov[::-1,::-1], proposal_dist = pm.MultivariateNormalProposal )

            elif Sampler_Name == "Metropolis":
                step = pm.Metropolis(S = self.Cov[::-1,::-1], proposal_dist = pm.MultivariateNormalProposal , blocked = True)

            elif Sampler_Name == "Hamiltonian":
                # the settings for HMC are very tricky. allowing adapt_step_size=True may lead to very small step sizes causing the method to stuck.
                #length = max(0.3, 1.5*np.sqrt(np.sum(np.array(self.STDs)**2)))  # this is the length in the parameter-space to travel between two points
                length = np.sqrt(len(self.STDs)) * np.mean(self.STDs)
                sub_l  = length/7                                               # setting substeps
                step = pm.HamiltonianMC(scaling = self.Cov[::-1,::-1], adapt_step_size= 0, step_scale = sub_l, path_length = length, is_cov = True )

                self.step = step
                self.step.adapt_step_size = False   # workaround for PyMC3 bug ( 'adapt_step_size= 0' is ignored)

                print(' >> Hamiltonian settings: {:7.4f} / {:7.4f}  = {:4} substeps between points'.format(length, sub_l/(len(self.STDs)**0.25), int(length / (sub_l/(len(self.STDs)**0.25)) )))

            else:
                print(' >> Unknown Sampler_Name = {:20}, Using Metropolis instead'.format(Sampler_Name))
                step = pm.Metropolis(S = self.Cov[::-1,::-1], proposal_dist = pm.MultivariateNormalProposal , blocked = True )

            # To be passed to the PyMC3 'sample' function
            self.Custom_sample_args = {
                "step"        : step,
                "progressbar" : IsProgressbar,
                "chains"      : N_chains,
                "cores"       : N_cores,
                "tune"        : N_tune ,
                #"parallelize" : True,
                }

        self.trace = None
        self.Prev_End = None


    def Sample(self, N_run = 50):
        '''
        Running the MCMC Sampling
            N_run : Number of Points to sample (The true numer of Chi2 calls is higher, depending on sampler)
        '''
        # Check for starting points
        try:
            trace = self.trace
            self.start = [trace.point(-1,i_C) for i_C in range(self.Custom_sample_args['chains'])]
            self.Custom_sample_args['tune'] = 0
            print(" >> Continouing previous trace")
        except:
            self.t0 = time()

            if self.Prev_End:
                self.start = self.Prev_End
                print(" >> Continouing previous trace from results")

            else:
                try: DD = self.Departure_Deviation
                except AttributeError: DD = 0.0

                trace = None
                self.start = self.Gen_Start_Points(DD)
                if DD >0.0:
                    print(" >> Using departure points for sampled each chain around given starting parameters with {} sigma".format(DD))
                else:
                    print(" >> Using exact departure points from USINE input file")


        print(" >> Sampling {} elememnts".format(N_run))
        with self.basic_model:
            # This is the actual MCMC run
            trace = pm.sample(N_run,
                trace = trace,
                start = self.start,
                **self.Custom_sample_args,
                )

            # No more tuning after first iteration
            self.Custom_sample_args['tune'] = 0

        post_data = np.array([
            [trace.get_values(self.VarNames[j_V], chains = i_C) for j_V in range(len(self.VarNames))]
                        for i_C in range(self.Custom_sample_args['chains'])]  )
        print(post_data.shape)
        self.trace = trace
        return post_data

    def SaveResults(self, **kwargs):
        Result_Key = kwargs.get("Result_Key", '')
        Result_Loc = kwargs.get("Result_Loc", '')

        # Saving the Chains independantly or accumulated
        Combined = kwargs.get("Combined", False)
        if Combined == False:
            for i_c in range(self.Custom_sample_args['chains']):
                output_filename = Result_Loc+'result_C{}_{}'.format(i_c, Result_Key)
                print (' >> saving chain {} as numpy array in {}'.format(i_c, output_filename))

                post_data = np.array([self.trace.get_values(self.VarNames[j_v], chains = i_c) for j_v in range(len(self.VarNames))]).T
                np.savetxt(output_filename, post_data, delimiter=',', header = str(self.VarNames))
        else:
            output_filename = Result_Loc+'result_{}'.format(Result_Key)
            print (' >> saving as numpy array in {}'.format(output_filename))

            post_data = np.array([self.trace.get_values(self.VarNames[j_v]) for j_v in range(len(self.VarNames))]).T
            np.savetxt(output_filename, post_data, delimiter=',', header = str(self.VarNames))

    def GetCovMatrix(self):
        post_data = np.array([self.trace.get_values(self.VarNames[j_v]) for j_v in range(len(self.VarNames))]).T
        return np.cov(post_data.T)

def RunMC(args):
    # now = datetime.datetime.now()
    log_file_name = "logger" # + '_' + datetime.datetime.now().strftime("%H_%M_%S")

    ParFile = args['P']
    if ParFile == None:
        print(" >> Not parameter file specified, aborting!\n")
        return

    Result_Loc      = args['O']
    N_run, N_tune, N_chains = args['C']
    IsProgressbar   = args['V']
    N_cores         = args['M']
    Sampler_Name    = args['S']
    N_I             = args['I']

    Key             = args['R']
    if len(Key) > 0:
        Key = '_'+Key

    if Result_Loc != '':
        Result_Loc += '/'

    log_file_name = Result_Loc + log_file_name

    Theta0 = None
    if args['T']:
        file     = args['T'][0]
        line_no  = int(args['T'][1])
        Theta0 = open(file).readlines()[line_no][:-1]
        Theta0 = list(eval(Theta0))
        print (' >> Using custom Theta0: ', Theta0)

    try:    os.mkdir(Result_Loc)
    except: pass

    MC = MCU()
    MC.InitPar(ParFile, log_file_name, Theta0)
    MC.InitPyMCBasic()

    if float(args['D']) > 0.0:
        MC.Departure_Deviation = float(args['D'])

    L_I = args['L']
    if L_I >= 0:
        MC.Prev_End = []

        for i_C in range(N_chains):
            last_res_file = Result_Loc+'result_C{}_I{}{}'.format(i_C,L_I,Key)
            try:
                last_vals = np.loadtxt(last_res_file, delimiter = ",", unpack = False)[-1]
                print(" >> loaded points from file {}".format(last_res_file))
            except:
                print(" >> aborting during loading of last results in {}\n".format(last_res_file))
                return
            _start = dict()
            for name, val in zip(MC.VarNames,last_vals):
                _start.update({name: val})
            MC.Prev_End.append(_start)

        if len(MC.Prev_End) != N_chains:
            print(" >> number of chains {} not equal to loaded last results {}\n".format(N_chains,len(MC.Prev_End) ))
            return

    MC.InitPyMCSampling(
        N_tune = N_tune,
        N_chains = N_chains,
        N_cores = N_cores,
        IsProgressbar = IsProgressbar,
        Sampler_Name = Sampler_Name,
    )

    # Main Loop over iterations
    for i_I in range(L_I+1, N_I + L_I+1):
        print('\n >> Starging interation {}'.format(i_I))

        #if Sampler_Name == 'Hamiltonian':
        #    log_file_name = Result_Loc + 'logger_I{}'.format((i_I))
        #    MC.log_file_name = log_file_name
        #    print(' >> Changing logfile to {}'.format(log_file_name))
        
        lf = open(MC.log_file_name,'a')
        lf.write('\n >> Starting Iteration  {:4} \n'.format(i_I) )
        lf.close()

        data = MC.Sample(N_run)
        MC.SaveResults(Result_Loc = Result_Loc, Result_Key = "I{}{}".format(i_I,Key))
        MC.SaveResults(Result_Loc = Result_Loc, Result_Key = "F", Combined = True)


        # Covariance matrix update (optional)
        if( (i_I+1)%args['U'] == 0 and args['U'] > 0 and i_I>0 ):
            New_Cov = MC.GetCovMatrix()*1.5
            cov_file = Result_Loc +'Cov_I{}{}'.format(i_I,Key)

            try:
                # Check Cov-Matrix if positive definite
                np.linalg.cholesky(New_Cov)
                MC.Cov = New_Cov*1.5
                print(' >> Updating Covariance Matrix from present Results, Saving in {}'.format(cov_file))

            except np.linalg.LinAlgError:
                if np.prod(np.diag(New_Cov)) > 0.0:
                    print(' >> Covariance Matrix is not positive definite; Updating diagnonal only')
                    MC.Cov = np.abs(np.diag(np.diag(New_Cov)) *1.5)   # killing off-diagnonal elements
                else:
                    print(' >> No update of Covariance Matrix due to zero-value in diagonal!')

            np.savetxt(cov_file, New_Cov, delimiter = ', ', header = ',  '.join(MC.VarNames))

            if (Sampler_Name != 'Hamiltonian'):
                MC.Custom_sample_args['step'].proposal_dist.__init__(MC.Cov[::-1,::-1])



if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', type=int, nargs = 3, default=[50, 0, 1] , help='N_Run, N_Tune, N_Chains')
    parser.add_argument('-M', type=int, default=1 , help='Multicore')
    parser.add_argument('-I', type=int, default=1 , help='N_Iterations')
    parser.add_argument('-P', type=str, help='Input arameter file')
    parser.add_argument('-O', type=str, default='OUT', help='Output directory')
    parser.add_argument('-R', type=str, default='', help='Result key')
    parser.add_argument('-V', type=int, default=1, help='Verbose progressbar')
    parser.add_argument('-S', type=str, default='DEMetropolis', help='Sampler Name')
    parser.add_argument('-T', nargs = 2, help='Theta0: File, line number')
    parser.add_argument('-D', type=float, default=0.0, help='sigma radius to sample starting points around Theta0')

    parser.add_argument('-L', type=int, default = -1, help='Last iteration index to load files from')
    parser.add_argument('-U', type=int, default=-1 , help='number of Iterations after which the Cov Matrix is to be updated')

    args = vars(parser.parse_args())
    print(args)
    RunMC(args)
