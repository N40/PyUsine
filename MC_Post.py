
from copy import deepcopy
import numpy.random as nr
import os

import numpy as np

class MCRes():
    def __init__(self, Out_Dir, N_C, N_I, Res_Key = ''):
        Files = os.listdir(Out_Dir)

        self.All_Chains_Cum_Traces = []
        self.Combined_Cum_Traces = []
        for i_C in range(N_C):
            Sub_Cum_Traces = []
            for j_I in range(N_I):
                f = Out_Dir+'/result_C{}_I{}{}'.format(i_C, j_I, Res_Key)

                Data = np.loadtxt(f, delimiter = ',')

                try:
                    if Sub_Cum_Traces[-1][0] in Data:
                        pass
                    else:
                        Data = np.concatenate((Sub_Cum_Traces[-1], Data))
                except IndexError:
                    pass
                Sub_Cum_Traces.append(Data)
            self.All_Chains_Cum_Traces.append(Sub_Cum_Traces)

        for j_I in range(N_I):
            Cum_Traces = []
            for i_C in range(N_C):
                try:
                    Cum_Traces = np.concatenate((Cum_Traces, deepcopy(self.All_Chains_Cum_Traces[i_C][j_I])))
                except:
                    Cum_Traces = deepcopy(self.All_Chains_Cum_Traces[i_C][j_I])
            self.Combined_Cum_Traces.append(Cum_Traces)

        self.Len_I = len(self.All_Chains_Cum_Traces[0][0])
        self.N_C = N_C
        self.N_V = len(self.Combined_Cum_Traces[0][0])


    def Combined_Mean(self, **kwargs):
        if kwargs.get("Start", False):
            i = kwargs["Start"]
            return np.array([ np.mean(Trace[:], axis = 0)  for Trace in self.Combined_Cum_Traces[i:]])
        else:
            i = kwargs.get("Last", 0)
            return np.array([ np.mean(Trace[-i*self.Len_I:], axis = 0)  for Trace in self.Combined_Cum_Traces[i:]])

    def Combined_Std(self, **kwargs):
        if kwargs.get("Start", False):
            i = kwargs["Start"]
            return np.array([ np.std(Trace[:], axis = 0)  for Trace in self.Combined_Cum_Traces[i:]])
        else:
            i = kwargs.get("Last", 0)
            return np.array([ np.std(Trace[-i*self.Len_I:], axis = 0)  for Trace in self.Combined_Cum_Traces[i:]])

    def Get_Rscore(self, kV, n):
        x = np.array([self.All_Chains_Cum_Traces[i_C][-1][:n].T[kV] for i_C in range(self.N_C )])
        num_samples = len(self.All_Chains_Cum_Traces[0][-1].T[kV])
        # Calculate between-chain variance
        B = num_samples * np.var(np.mean(x, axis=1), axis=0, ddof=1)
        # Calculate within-chain variance
        W = np.mean(np.var(x, axis=1, ddof=1), axis=0)
        # Estimate of marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples
        return np.sqrt(Vhat / W)


def Combine_Trace(Out_Dir = '.', Res_Key = ''):
    for i_C in range(20):
        Chain = []
        for j_I in range(50):
            input_file = Out_Dir+'/result_C{}_I{}{}'.format(i_C, j_I, Res_Key)

            try:
                Data = np.loadtxt(input_file, delimiter = ',')
            except OSError:
                if len(Chain)>0:
                    print(' >> Combined trace ',i_C, '  ',len(Chain))
                    output_file = Out_Dir+'/result_C{}_X{}'.format(i_C, Res_Key)
                    np.savetxt(output_file, Chain, delimiter = ', ', header = str(VarNames))
                break

            VarNames = eval(open(input_file).readline()[2:])
            try:
                if (Chain[0] in Data and Chain[1] in Data):
                    pass
                else:
                    Data = np.concatenate((Chain, Data))
            except:
                pass

            Chain = Data
