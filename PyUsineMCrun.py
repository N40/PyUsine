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
    Storage in order to prevent from unnessesary Chi2 calculations
    by re-using allready calculated values.
    Most recenly used parameters are beeing kept while not regarded are beeing eliminated
    when new ones are saved
    '''
    A_Chi2 = []
    A_theta = []
    n_ = -1

    def __init__(self, n_ =5):
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

# define a theano Op for our likelihood function
# class TheanWrapper(tt.Op):
#     '''
#     This function is no longer used.
#     It only serves for non-gradient based methods.
#     '''
#     itypes = [tt.dvector] # expects a vector of parameter values when called
#     otypes = [tt.dscalar] # outputs a single scalar value
#
#     def __init__(self, likefct):
#         """
#         loglike:
#             The log-likelihood (or whatever) function we've defined
#         """
#         # add inputs as class attributes
#         self.likelihood = likefct
#
#     def perform(self, node, inputs, outputs):
#         # the method that is used when calling the Op
#         theta, = inputs  # this will contain my variables
#         # call the log-likelihood function
#         ret_like = self.likelihood(theta, 1)
#         outputs[0][0] = np.array(ret_like) # output

class TheanWrapperGrad(tt.Op):
    """
    A generic Class, necessary in order to call custom log-probability function
    """
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

class Chi2Eval():
    """
    Class in order to mediate the execution and documentation of the Chi2 function calls
    Could be merged into the MCU class in future
    """
    def __init__(self):
        self.run = PP.PyRunPropagation()
        self.InitVals = []
        self.t0 = time()
        self.log_file_name = 'logger'
        self.S = Storage_Container(5)
        self.step = None  # introduced for debugging reasons concerning HamiltonianMC

    def Chi2(self, theta, Option = 1):
        return self.HalfNegChi2(theta, Option)*(-2.0)

    def HalfNegChi2(self, theta, Option = 1):
        '''
        Option:
            1 : Normal function call
            0 : Call, without logging
            2 : Used for gradient calculation
            3 : Usde for gradient calculation, ignoring boundaries
        '''
        theta = list(theta)
        chi2 = 0
        InBoundary = True
        Flag = str(int(Option))

        for par,val in zip(theta, self.InitVals):
            if ((val[1] > par or par > val[2]) and Option != 3):
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

        if (bool(Option)):
            lf = open(self.log_file_name,'a+')

            # HamiltonianMC debugging by displaying the present step size
            try:
                lf.write(' {:12}'.format(round(self.step.step_size,6)))
                #lf.write(' {:6} '.format((self.step.adapt_step_size)))
            except: pass

            lf.write("{:10}  {:15}  {:6}  ".format(round(time()-self.t0,3), round(chi2,3),  Flag))
            lf.write('[ ' + ' '.join(["{:10},".format(round(p,6)) for p in theta]) + '  ] \n')
            lf.close()

        return result

class MCU(Chi2Eval):
    """
    This class contains all the steps and routines for generating MCMC Fits
    """
    def __init__(self, **kwargs):
        self.Cov = None
        self.basic_model = None
        super().__init__()

    def InitPar(self, ParFile, log_file_name = None, Theta0 = None):
        """
        Loading Usine Configuration and setting all intern parameters
        correspondingly
        """
        now = datetime.datetime.now()
        if log_file_name:
            self.log_file_name = log_file_name
        else:
            self.log_file_name = "logger_" + datetime.datetime.now().strftime("%H_%M_%S")
        print (' >> Saving futher Calculations in {}'.format(self.log_file_name))
        open(self.log_file_name,'w+').close() # wiping the logfile

        self.ParFile = ParFile
        print (' >> Loading configuration from {}'.format(self.ParFile))

        self.run.PySetLogFile("run.log") # this is the USINE log file, not the MCMC one
        self.run.PySetClass(self.ParFile, 1, "OUT")

        self.InitVals = self.run.PyGetInitVals()
        self.VarNames = self.run.PyGetFreeParNames()
        self.FixedVarNames = self.run.PyGetFixedParNames()
        self.Theta0 = []
        self.STDs = []
        for i in range(len(self.VarNames)):
            if Theta0:
                self.InitVals[i][0] = Theta0[i]

            self.Theta0.append(self.InitVals[i][0])
            self.STDs.append(self.InitVals[i][3])

        # SORTING OUT FIXED VARIABLES
        print (' >> Not regarding the following FIXED parameters:')
        for name in self.FixedVarNames:
            print('{:25}'.format(name))

        # INIT PARAMETERS USED
        print (' >> Using {} free parameters with the following values:'.format(len(self.VarNames)))
        for name, vals in zip(self.VarNames, self.InitVals):
            print('{:25}  [{:6.3f},   {:6.3f} +- {:6.3f}   ,{:6.3f}]'.format(name[:22], vals[1], vals[0], vals[3], vals[2]))
        print(' ')

    def Gen_Start_Points(self, sigma = 0.1):
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

    def SetCovMatrix(self, **kwargs):
        try:
            self.Cov = np.loadtxt(kwargs["Cov"] , delimiter = ',' )
            print(" >> Valid Covariance matrix {} found".format(kwargs["Cov"]))
        except:
            Scale = kwargs.get("Scale",0.5)
            print(" >> No valid Covariance matric found",
                  "\n >> creating diagnonal one with scale {}".format(Scale))
            self.Cov = np.diag([(Scale*var[3])**2 for var in self.InitVals])


    def InitPyMC(self):
        self.basic_model = pm.Model()
        # Setting the extern blackbox function
        ext_fct = TheanWrapperGrad(self.HalfNegChi2, np.array(self.STDs))
        ext_fct.logpgrad.GradVerbose = 2  # =2: gradient calculation steps are shown in logfile

        with self.basic_model:
            # Setting up the Parameters for MCMC to use (no priors here!)
            Priors = []
            print (' >> Using {} free parameters with the following values:'.format(len(self.VarNames)))
            for name, vals in zip(self.VarNames, self.InitVals):
                #P = pm.Normal(name, mu=vals[0], sd=vals[3]*1.0)
                P = pm.Uniform(name, lower=vals[1], upper=vals[2])
                Priors.append(P)

            theta = tt.as_tensor_variable(Priors)

            # Likelihood (sampling distribution) of observations. Specified as log_like
            likelihood = pm.DensityDist('likelihood',  lambda v: ext_fct(v), observed={'v': theta})

    def InitPyMCSampling(self, **kwargs):
        '...'
        #Checking if all the necessary stuff is loaded
        if not self.VarNames:
            self.InitPar(kwargs["ParFile"])
        try: self.Cov[0][0]
        except TypeError: self.SetCovMatrix(Scale = 1.2)
        if not self.basic_model:
            self.InitPyMC()

        # RUNNING PYMC3
        print(' >> Logging calculation steps in {}'.format(self.log_file_name) )
        open(self.log_file_name,'w+').close()
        with self.basic_model:
            Sampler_Name = kwargs.get("Sampler_Name","Metropolis")
            #N_run  = kwargs.get("N_run" , 500)
            N_tune = kwargs.get("N_tune" , 0)
            N_chains = kwargs.get("N_chains" , 1)
            N_cores = kwargs.get("N_cores" , min(4,N_chains))
            IsProgressbar = kwargs.get("IsProgressbar" , 1)
            print ('\n >> using configuration :  {:12}, N_tune = {}, N_chains = {}, N_cores = {}'.format(Sampler_Name,N_tune,N_chains,N_cores))

            self.S = Storage_Container(2*N_chains*len(self.VarNames))
            """
            calling S = self.Cov[::-1,::-1] is a neccessary hack in order to avoid a problem in the PyMC3 code
            the order of the variables is inverted (by accident?) durint the BlockStep().__init__()
            """
            if Sampler_Name == "DEMetropolis":
                step = pm.DEMetropolis(S = self.Cov[::-1,::-1], proposal_dist = pm.MultivariateNormalProposal )

            elif Sampler_Name == "Metropolis":
                step = pm.Metropolis(S = self.Cov[::-1,::-1], proposal_dist = pm.MultivariateNormalProposal , blocked = True)

            elif Sampler_Name == "Hamiltonian":
                # these settings for HMC are very tricky. allowing adapt_step_size=True may lead to extr. small step sizes causing the method to stuck.
                length = max(0.3, 1.5*np.sqrt(np.sum(np.array(self.STDs)**2)))
                sub_l  = length/7
                step = pm.HamiltonianMC(adapt_step_size= 0, step_scale = sub_l, path_length = length, is_cov = True,  scaling = self.Cov[::-1,::-1] )
                #step_scale = 0.01, path_length = 0.1, target_accept = 0.85)
                self.step = step  # debugging feature for HamiltonianMC
                #print(self.step.adapt_step_size)
                self.step.adapt_step_size = False
                print(' >> Hamiltonian settings: {:7.4f} / {:7.4f}  = {:4}'.format(length, sub_l/(len(self.STDs)**0.25), int(length / (sub_l/(len(self.STDs)**0.25)) )))

            else:
                print(' >> Unknown Sampler_Name = {:20}, Using Metropolis instead'.format(Sampler_Name))
                step = pm.Metropolis(S = self.Cov[::-1,::-1], proposal_dist = pm.MultivariateNormalProposal , blocked = True )

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
        # Check for starting points
        try:
            trace = self.trace
            self.start = [trace.point(-1,i_C) for i_C in range(self.Custom_sample_args['chains'])]
            self.Custom_sample_args['tune'] = 0
            print(" >> Continouing previous trace")
        except:
            if self.Prev_End:
                self.start = self.Prev_End
                print(" >> Continouing previous trace from results")
            elif self.Custom_sample_args['chains'] > 1:
                self.t0 = time()
                trace = None
                self.start = self.Gen_Start_Points()
                print(" >> Using departure points for sampled each chain around given starting parameters")
            else:
                # Default case: none given, only one chain; taking starting position from init-file
                self.t0 = time()
                self.start = self.Gen_Start_Points(0.0)
                trace = None
                print(" >> Using departure points from USINE input file")


        print(" >> Sampling {} elememnts".format(N_run))
        with self.basic_model:
            trace = pm.sample(N_run,
                trace = trace,
                start = self.start,
                **self.Custom_sample_args,
                )

            self.Custom_sample_args['tune'] = 0

        post_data = np.array([
            [trace.get_values(self.VarNames[j_V], chains = i_C) for j_V in range(len(self.VarNames))]
                        for i_C in range(self.Custom_sample_args['chains'])]  )
        print(post_data.shape)
        self.trace = trace
        return post_data

    def SaveResults(self, **kwargs):
        time_stamp = datetime.datetime.now().strftime("%d.%m.%Y_%H:%M")
        Result_Key = kwargs.get("Result_Key", time_stamp)
        Result_Loc = kwargs.get("Result_Loc", '')

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
    now = datetime.datetime.now()
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
        print (Theta0)

    try:
        os.mkdir(Result_Loc)
    except:
        pass

    MC = MCU()
    MC.InitPar(ParFile, log_file_name, Theta0)
    MC.InitPyMC()

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

    for i_I in range(L_I+1, N_I + L_I+1):
        print('\n >> Starging interation {}'.format(i_I))

        if Sampler_Name == 'Hamiltonian':
            log_file_name = Result_Loc + 'logger_I{}'.format((i_I))
            MC.log_file_name = log_file_name
            print(' >> Changing logfile to {}'.format(log_file_name))

        data = MC.Sample(N_run)
        MC.SaveResults(Result_Loc = Result_Loc, Result_Key = "I{}{}".format(i_I,Key))


        if(Sampler_Name != 'Hamiltonian' and (i_I+1)%args['U'] == 0 and args['U'] > 0 and i_I>0):
            New_Cov = MC.GetCovMatrix()*1.5
            cov_file = Result_Loc +'Cov_I{}{}'.format(i_I,Key)

            try:
                # Check Cov-Matrix if positive definite
                np.linalg.cholesky(New_Cov)
                MC.Cov = New_Cov*1.5
                print(' >> Updating Covariance Matrix from present Results, Saving in {}'.format(cov_file))

            except np.linalg.LinAlgError:
                print(' >> Covariance Matrix is not positive definite; Updating diagnonal only')
                MC.Cov = np.diag(np.diag(New_Cov)) *1.5   # killing off-diagnonal elements

            MC.Custom_sample_args['step'].proposal_dist.__init__(MC.Cov[::-1,::-1])
            np.savetxt(cov_file, New_Cov, delimiter = ', ', header = ',  '.join(MC.VarNames))




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
    parser.add_argument('-L', type=int, default = -1, help='Last iteration index to load files from')
    parser.add_argument('-U', type=int, default=-1 , help='number of Iterations after which the Cov Matrix is to be updated')

    args = vars(parser.parse_args())
    print(args)
    RunMC(args)
