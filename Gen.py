import numpy as np

from copy import deepcopy
import numpy.random as nr
from pymc3.stats import autocorr
import os
import argparse

import datetime

import scipy.optimize

from time import time
t0 = time()

from PyUsineMCrun import MCU

def Sigmoid(x):
    return 1/(1 + np.exp(x))

class Genetic_Routine():
    def __init__(self,ParFile,log_file_name, Out_Dir):
        #initialization
        self.MC = MCU()
        self.MC.InitPar(ParFile = ParFile, log_file_name = log_file_name)
        self.log_file_name = log_file_name
        self.Out_Dir = Out_Dir
        
        try: os.mkdir(Out_Dir)
        except: pass
        
        open(log_file_name, "w+").close()
        
        self.N_Pop = 50
        self.N_Par = 15
        self.N_New = 5
        
        Gen0 = []
        for i in range(self.N_Pop):
            th = self.Mutate(self.MC.Theta0,0.1)
            Gen0.append((th, (-2)*self.MC.CE.HalfNegChi2(th)))

        self.Gens = [deepcopy(Gen0)]
        self.Scores = [deepcopy(np.sort(Gen0, axis = 0)[0][1])]
        self.N_Calc = [self.N_Pop]
        self.IG = 1
        
    def Run(self,N_G):
        """
        Main routine
        """
        for g in range(self.IG ,self.IG + N_G):
            print('\n >> Calculating generation {} / {} '.format(g,N_G))
            self.F = 0.1 * 5/(5 + g)
            NewGen = self.MakeNextGen(self.Gens[-1], N_Par = self.N_Par, Method = "Crossover", SFac = self.F, N_New = self.N_New)
            self.Gens.append(NewGen)
            f = open(self.log_file_name, "a+")
            BestScore = np.sort(NewGen, axis = 0)[0][1]
            OutString = ('generation {}   ->   best Chi2 {}\n\n'.format(g, BestScore  ))
            f.write(OutString)
            f.close()

            self.Scores.append(BestScore)
            self.N_Calc.append(self.N_Calc[-1] + self.N_Pop)
            header = ('N_Pop = {}, N_Par_Cross = {}, N_New = {}'.format(self.N_Pop, self.N_Par, self.N_New) 
                      + '\n F = 0.1 * 5/(5 + g)   , New Scale = 2*F'
                      + '\n Init = init_2019019_BCpaper_Mod_B_NSS.par '  )
            np.savetxt(self.Out_Dir+"/GS_{}_F5_".format(self.N_Pop),
                       np.array([self.N_Calc,self.Scores]).T, delimiter = ' , ', header = header )
            print('  - > Best Chi2: {}   -  NCalls : {} '.format(BestScore,self.N_Calc[-1]))

        IG = len(self.Gens)    
        
        


    def Crossover(self, Parent1, Parent2, Weights = True, **kwargs):
        Newtheta = deepcopy(Parent1[0])
        Diff = (Parent2[1] - Parent1[1])*float(Weights)
        P2 = Sigmoid(Diff)
        B2 = nr.random(len(Newtheta)) < P2
        for i, b in enumerate(B2):
            if b:
                Newtheta[i] = Parent2[0][i]
        return Newtheta

    def Mutate(self, theta, SFac = 0.1):
        Delta = nr.normal(scale = np.array(self.MC.STDs)*SFac)
        return list(theta + Delta)

    def MakeNextGen(self, LastGen, N_Par = -1, Method = "Survival", SFac = 0.1, N_New = 0, **kwargs):
        #print(Method)
        if N_Par <1:
            N_Par = int(len(LastGen)/3)

        Parents = sorted(LastGen, key=lambda x: x[1])
        Parents = Parents[:N_Par]
        #print(' '.join([str(round(g[1],2)) for g in Parents]))

        NewGen = []
        for i in range(self.N_Pop-self.N_New):
            if Method == 'Crossover':
                k0, k1 = [int(x%N_Par) for x in -np.log(nr.random(2))*N_Par*0.8]
                NewTheta = self.Crossover(Parents[k0], Parents[k1], True)
                #print('[ {}  {} ]'.format(round(Parents[k0][1],3), round(Parents[k1][1],3)))
            elif Method == 'Survival':
                k0 = [int(x%N_Par) for x in -np.log(nr.random(1))*N_Par*0.8][0]
                NewTheta = deepcopy(Parents[k0][0])
                #print('[ {} - {} ]'.format(k0,round(Parents[k0][1],2)), end = '  ')

            NewTheta = self.Mutate(NewTheta, SFac)
            NewChi2 = (-2)*self.MC.CE.HalfNegChi2(NewTheta)
            NewGen.append((NewTheta, NewChi2 ))

        if self.N_New >0:
            m = np.zeros(len(self.MC.Theta0))
            for i,P in enumerate((Parents)):
                m += np.array(P[0])*P[1]
            m/= np.sum(np.array(Parents).T[1])

            for i in range(self.N_New):
                NewTheta = self.Mutate(m,self.F*2)
                NewChi2 = (-2)*self.MC.CE.HalfNegChi2(NewTheta)
                NewGen.append((NewTheta, NewChi2 ))

        return NewGen
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', type=str, help='Input arameter file')
    parser.add_argument('-O', type=str, default='OUT_X', help='Output directory')
    parser.add_argument('-N', type=int, default=10, help='Number of Generations')
    args = vars(parser.parse_args())
    
    GR = Genetic_Routine(args['P'],'logger_N{}'.format(args['O'][4:]),args['O'])
    GR.Run(args['N'])
    
    