"""preprocess data for actual fitting script"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# previous script...
from import_raw import *
from scipy import stats, signal
from peakdetect import peakdet
# import scipy.fftpack

def length_fix(arr_list, len):
	# get rid of non-standard length samples because they mess up the np.stack
	for ind, arr in enumerate(arr_list):
		# print arr.shape[0] == tic
		if arr.shape[0] != len:
			arr_list.pop(ind)
			print "removed {}".format(arr)

	return arr_list

def main_peaks(vector):
	num_peaks = 10
	delta = 0.3
	maxtab, mintab = peakdet(vector, delta, x=None)
	main_peaks = np.sort(maxtab[:,0])[:num_peaks]
	return main_peaks

def transform_sample(eeg, hypno, tic, sub_sample=True, peaks=False):
	# --- merge data, split up by score group, sub-sample higher-Hz signals to match lower one if required
	# convert to series
	hypno = hypno[hypno.columns[0]]
	eeg = eeg[eeg.columns[0]]
	# extra line to make sure we don't go over max length of all samples (long story)
	eeg = eeg[:360000]
	# get classes
	classes = pd.unique(hypno)
	# pad hypno scores for same length as eeg array using 'tic' i.e. freq(Hz) * timewindow (seconds)
	convert_tic = int(5*tic)
	hypno_pad = np.repeat(hypno, convert_tic)
	# print "hypno_shape:", hypno_pad.shape
	# print "eeg shape", eeg.shape
	# ergo create array with s1 = (1) hypno, s2 = (100,1) sample array  
	data = np.stack([eeg, hypno_pad], axis=1)

	# --- select by class, then split into 1 second samples 
	# split by class; as well as keep final complete dataset
	raw_class_dict = {}
	fourier_class_dict =  {}
	FULL_CLASS = []
	FULL_RAW = []
	FULL_FOURIER = []
	PEAK_FOURIER = []

	print classes

	# for cl in classes:
	for cl in classes:
		# (EEG has frequencies 2-20Hz, roughly, so that should do it)
		sel = data[data[:,1] == cl]
		##  now that split by class is done, only interested in signal column, so index sel as [:,0]
		raw_class_data = np.split(sel[:,0], sel.shape[0]/tic)
		raw_class_data = length_fix(raw_class_data, len=tic)
		fourier_class_data = np.fft.fft(raw_class_data)
		if sub_sample and tic != 50:
			sample_ratio = tic/50
			raw_class_data = [signal[::sample_ratio] for signal in raw_class_data]
			fourier_class_data = np.abs(np.fft.fft(raw_class_data))
		# can reduce dimensionality with peak detection for fourier sample:
		if peaks:
			fourier_peak_data = []
			for signal in fourier_class_data:
				fourier_peak_data.append(main_peaks(signal))
		"""
		for i, arr in enumerate(np.abs(fourier_class_data)):
			print i
			max_tab, _ = peakdet(arr, 0.01)
			# MAXTAB consists of two columns. Column 1
			# contains indices in V, and column 2 the found values.
			main_peaks = np.sort(max_tab[:,0])[:10] # [np.argsort(max_tab[1])]				
			print main_peaks
			fourier_peak_data.extend(main_peaks)
		"""
		# freq_range_data = np.fft.fftfreq()
		##  add data to dictionary of classes
		raw_class_dict.update({cl : raw_class_data})
		fourier_class_dict.update({cl : fourier_class_data})

		## ----  add data to complete dataset
		FULL_RAW.extend(raw_class_data)
		FULL_FOURIER.extend(fourier_class_data)
		FULL_CLASS.extend([cl]*fourier_class_data.shape[0])
		PEAK_FOURIER.extend(fourier_peak_data)
		# FULL_CLASS.append(np.repeat(cl, fourier_class_data.shape[0])

	# print [x.shape[0] for x in FULL_FOURIER]
	return FULL_RAW, FULL_FOURIER, FULL_CLASS, PEAK_FOURIER

FULL_RAW, FULL_FOURIER, FULL_CLASS, PEAK_FOURIER = [],[],[],[]
#  only use 200 Hz for now:
# indices = [i for i, x in enumerate(TIC_LIST) if x == 200]
"""
for i in range(1,9):
	eeg = pd.read_csv("data/excerpt{}.txt".format(i))
	hypno = pd.read_csv("data/Hypnogram_excerpt{}.txt".format(i))
	tic = TIC_LIST[i-1]
	# print "\n \n working on file {}, tic {}".format(i+1, tic)
	FR,FF,FC,PF = transform_sample(eeg, hypno, tic, sub_sample=True, peaks=True)
	FULL_RAW.extend(FR)
	FULL_FOURIER.extend(FF)
	FULL_CLASS.extend(FC)
	PEAK_FOURIER.extend(PF)
"""

for pos, tigger in enumerate(TIC_LIST):
	# for this run, only use the tic=200 samples
	i = pos + 1
	if tigger == 200:
		eeg = pd.read_csv("data/excerpt{}.txt".format(i))
		hypno = pd.read_csv("data/Hypnogram_excerpt{}.txt".format(i))
		tic = tigger
		# print "\n \n working on file {}, tic {}".format(i+1, tic)
		FR,FF,FC,PF = transform_sample(eeg, hypno, tic, sub_sample=False, peaks=True)
		FULL_RAW.extend(FR)
		FULL_FOURIER.extend(FF)
		FULL_CLASS.extend(FC)
		PEAK_FOURIER.extend(PF)


# print [len(i) for i in (FULL_FOURIER, PEAK_FOURIER)]


# hint: when accessing dict afterwards, can do it as fourier_class_dict['class_value'][column,row]

# --- bonus: plot one of the fouriers to check if sensible
def plot_fourier(fft_vals):
	# freq=np.fft.fftfreq(fft_vals.shape[0], d=1.0/tic)
	N = 50
	T = 1.0 / 50
	xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
	fig,ax = plt.subplots()
	plt.plot(xf, np.abs(fft_vals[:N//2]))
	plt.show()

some_sample = FULL_FOURIER[0]
#print some_sample
#plot_fourier(some_sample)






"""
# Number of samplepoints
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()

"""