import numpy as np
import matplotlib.pyplot as plt
import pdb
import argparse

from numpy import convolve
 
def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input loss')
    args, unparsed = parser.parse_known_args()
    
    lll = np.load(args.input)
    lll_avg = [movingaverage(lll[:,x],2000) for x in range(lll.shape[-1])]
    #pdb.set_trace()


    lll_avg_norm = [(np.array(lll_avg[x])-np.min(lll_avg[x]))/np.max(np.array(lll_avg[x])-np.min(lll_avg[x])) for x in range(lll.shape[-1])]
    #lll_avg_norm = [np.log(lll_avg[x]) for x in range(lll.shape[-1])]
    
    #lll_avg_norm = lll_avg    

    #plt.plot(lll_avg_norm[1])
    
    sm, = plt.plot(lll_avg_norm[0][:17000],label='sm')
    tmp, = plt.plot(lll_avg_norm[1][:17000],label='tmp')
    mm, = plt.plot(lll_avg_norm[2][:17000],label='mm')
    tt, = plt.plot(lll_avg_norm[3][:17000],label='total')
    plt.legend(handles=[sm,tmp,mm,tt])
    plt.xlabel('Iterations')
    plt.ylabel('Norm Loss')
    '''
    plt.plot(lll_avg_norm[0])
    plt.plot(lll_avg_norm[1])
    plt.plot(lll_avg_norm[2])
    plt.plot(lll_avg_norm[3])
    '''

    #plt.show()
    plt.savefig(args.input.split('/')[-2]+'_loss.jpg')
    #pdb.set_trace()

if __name__ == '__main__':
    main()
