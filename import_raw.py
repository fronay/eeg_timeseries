"""load sample datasets, plot an overview..."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# dataset downloaded from http://www.tcts.fpms.ac.be/~devuyst/Databases/DatabaseSpindles/

EEG_PATH = "data/excerpt1.txt"
HYPNO_PATH = "data/Hypnogram_excerpt1.txt"
SPINDLE_PATH = "data/Visual_scoring1_excerpt1.txt"
TIC_LIST = [100,200,50,200,200,200,200,200]

eeg = pd.read_csv(EEG_PATH)
hypno = pd.read_csv(HYPNO_PATH)
expert = pd.read_csv(SPINDLE_PATH)

# wait, there are 8 files in total, so import (and perhaps concat) more series
path_gen = lambda pth: ["{}{}.txt".format(pth,i) for i in range(1,9)]
eeg_paths = path_gen("data/excerpt")
hypno_paths = path_gen("data/Hypnogram_excerpt")


# sanity check shapes of eeg and hypno files
def check_shapes():
	es, hs = [],[]
	for pth in eeg_paths:
		es.append(pd.read_csv(pth).shape[0])
	for pth in hypno_paths:
		hs.append(pd.read_csv(pth).shape[0])

	for ind, shp in enumerate(es):
		rat = es[ind]/float(hs[ind])
		tiggo = TIC_LIST[ind]
		print "eeg: {}, hypno: {}, eeg/hypno ratio: {}, tic: {}, tic/conv ratio: {}".format(es[ind], hs[ind], rat, tiggo, rat/tiggo )

# check_shapes()

# print eeg.describe(), hypno.describe()
# print "eeg sample timing is {} ms, hypno samples cover {} ms".format(eeg.shape[0], hypno.shape[0]*500)

if __name__ == "__main__":
	# plot if called as main module
	# get multiple 5 second samples for plotting
	# at 100 Hz sampling in first sample, each hypno-score covers 5000ms
	tic = TIC_LIST[0]
	def plotty():

		num = 500
		sample = eeg[:(num*tic)]
		hyp = hypno[:num]
		# 2 separate axis for eeg and scoring
		fig, ax = plt.subplots(2,1)
		ax_score = ax[0].twinx()
		ax[0].plot(np.arange(sample.shape[0]), sample, 'b') 
		ax_score.plot(np.arange(hyp.shape[0])*tic, hyp, 'r')
		ax[0].set_xlabel('Time (ms)')
		ax[0].set_xlabel('Amplitude EEG (mV)')
		ax_score.set_ylabel('Sleep stage (1-5-X)')

		# attempt fourier transform
		n = sample.shape[0]
		k = np.arange(n)
		T = float(n)/tic
		frq = k/T # two sides frequency range
		frq = frq[range(n/2)] # one side frequency range
		Y = np.fft.fft(sample)/n # fft computing and normalization
		Y = Y[range(n/2)]
		ax[1].plot(frq,abs(Y),'g') # plotting the spectrum
		ax[1].set_xlabel('Freq (Hz)')
		ax[1].set_ylabel('|Y(freq)|')
		plt.show()
	plotty()

