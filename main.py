"""
@author: Maik Simke

Automatically pitches timed lyrics for UltraStar Deluxe using SPICE.

SPICE License: Apache-2.0

Ressources:
https://tfhub.dev/google/spice/2
https://blog.tensorflow.org/2020/06/estimating-pitch-with-spice-and-tensorflow-hub.html
https://tensorflow.google.cn/hub/tutorials/spice?hl=en
"""

import argparse
import numpy as np
from audio2numpy import open_audio
from scipy.signal import resample
from time import time
# import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow_hub as hub


def initArgs():
	parser = argparse.ArgumentParser(description="Automatically pitches timed lyrics for UltraStar Deluxe using SPICE.")

	parser.add_argument("filename", type=str,
						help="Name or path of the karaoke file")
	parser.add_argument("audiofile", type=str,
						help="Name or path of the audio file")
	parser.add_argument("-c", "--confidence", type=float, default=0.85,
						help="How confident the model has to be. Default: 0.85")
	parser.add_argument("-gpu", "--gpu", action='store_true', default=False,
						help="Use GPU instead of CPU. Default: False")

	args = parser.parse_args()

	return args


def loadFile():
	src = args.filename

	try:
		with open(src, "r") as f:
			lines = f.readlines()
	except OSError:
		exit("Karaoke file not found! Please make sure if the filename/path is spelled correctly.")
	except:
		exit("Something went wrong with reading the karaoke file.")

	metaData = []
	for line in lines:
		if line[0] == "#":
			metaData.append(line)

	lyrics = []
	for line in lines:
		if line[0] not in ["#"]:
			split = line.split(" ")									# Note: this keeps the line-breaks for the last string.

			if len(split) == 6:
				split[4] = " " + split.pop()						# Appends space character if part of the lyrics (word divider)
			elif len(split) > 6:
				exit("Invalid format. Failed on: " + line + "Please make sure that there are no unnecessary spaces in the line.")

			lyrics.append(split)

	return metaData, lyrics


def loadAudio():
	try:
		audioData, samplerate = open_audio(args.audiofile)
	except OSError:
		exit("Audio file not found! Please make sure if the filename/path is spelled correctly.")
	except:
		exit("Something went wrong with reading the audio file.")

	return audioData, samplerate


def prepareAudio(audioData, samplerate):
	if len(audioData.shape) > 1:
		audioData = np.mean(audioData, axis=1)						# Conversion to mono channel

	targetSamplerate = 16000										# SPICE requires a samplerate of 16kHz
	resampleFactor = targetSamplerate/samplerate
	processedAudio = resample(audioData, int(np.ceil(len(audioData)*resampleFactor)))

	# Pad length to multiples of 512
	processedAudioPadded = np.zeros(int(np.ceil(len(processedAudio)/512)) * 512)
	processedAudioPadded[:len(processedAudio)] = processedAudio

	return processedAudioPadded


def analyze(audioData):
	if not args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'					# Use CPU instead of GPU for tensorflow (Somehow faster in this case...)
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"						# Suppress tensorflow information
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)	# Suppress tensorflow errors

	model = hub.load("spice_2")										# "https://tfhub.dev/google/spice/2" to load model from server, instead of local files

	modelOutput = model.signatures["serving_default"](tf.constant(audioData, tf.float32))

	return modelOutput


def pitch(metaData, lyrics, modelOutput):
	bpm = None
	gap = None														# In milliseconds

	for line in metaData:
		if line[0:4] == "#BPM":
			bpm = float(line[5:])
		if line[0:4] == "#GAP":
			gap = float(line[5:])

	tps = int(bpm)*4/60												# Ticks per second
	
	if bpm == None:
		exit("No #BPM-Tag in header found!")
	if gap == None:
		exit("No #GAP-Tag in header found!")

	pitch = np.array(modelOutput["pitch"])
	uncertainty = np.array(modelOutput["uncertainty"])
	confidence = 1.0 - uncertainty

	"""
	# Creates scatter graph of pitch and confidence
	testPitch = []
	testConfidence = []
	testX = []
	for i in range(len(pitch)):
		if confidence[i] > args.confidence:
			testPitch.append(pitch[i])
			testConfidence.append(confidence[i])
			testX.append(i*32/1000)

	fig, ax = plt.subplots()
	fig.set_size_inches(20, 10)
	plt.scatter(testX, testPitch)
	plt.scatter(testX, testConfidence)
	plt.savefig('test.png')
	"""

	lyricsPitched = []
	for line in lyrics:
		if line[0].startswith("-"):
			lyricsPitched.append([line[0], line[1]])

		elif line[0].startswith("E"):
			lyricsPitched.append([line[0]])

		else:
			if len(line) < 5:
				exit("Invalid line: " + str(line) + ".")

			start = int(((gap/1000) + int(line[1])/tps - 0.5/tps) * 1000/32)
			end = int(((gap/1000) + (int(line[1]) + int(line[2]))/tps + 0.5/tps) * 1000/32)

			pitches = pitch[start:end]
			pitches = pitches[confidence[start:end] >= args.confidence]

			if len(pitches) == 0:
				note = 0
			else:
				hz = pitch2hz(np.median(pitches))
				note = hz2note(hz)

			lyricsPitched.append([line[0], line[1], line[2], str(note), line[4]])

	return lyricsPitched


def pitch2hz(pitch):
	# Constants taken from https://tfhub.dev/google/spice/2
	PT_OFFSET = 25.58
	PT_SLOPE = 63.07
	FMIN = 10.0
	BINS_PER_OCTAVE = 12.0
	cqt_bin = pitch * PT_SLOPE + PT_OFFSET

	return FMIN * 2.0 ** (1.0 * cqt_bin / BINS_PER_OCTAVE)


def hz2note(hz):
	# 9 is Kammerton A4 at 440Hz, scale is shifted by 60 compared to the MIDI scale, so MIDI 60 (C4) is 0 and MIDI 69 is 9 (A4)
	# f = 440*2^((n-69)/12), for f in Hz and n in MIDI number
	# n = 12 * log(4f/55 * 2^3/4)/log(2)
	
	return int(np.round(12 * np.log(4*hz/55 * 2**(3/4))/np.log(2))) - 60


def writeFile(metaData, lyricsPitched):
	dest = args.filename[:-4] + "_pitched.txt"

	with open(dest, "w") as f:
		pass

	lyricsString = []
	for line in lyricsPitched:
		if line[0].startswith("-"):
			lyricsString.append(line[0] + " " + line[1])
		elif line[0].startswith("E"):
			lyricsString.append(line[0])
		else:
			if len(line) < 5:
				exit("Invalid line: " + str(line) + ".")
			
			lyricsString.append(line[0] + " " + line[1] + " " + line[2] + " " + line[3] + " " + line[4])

	file1 = open(dest, "a")
	file1.writelines(metaData)
	file1.writelines(lyricsString)
	file1.close()


if __name__ == '__main__':
	startTime = time()

	args = initArgs()

	print("Reading file...")
	metaData, lyrics = loadFile()

	print(f"Loading audio...")
	audioData, samplerate = loadAudio()

	print(f"Preparing audio...")
	audioData = prepareAudio(audioData, samplerate)

	print(f"Analyzing audio...")
	modelOutput = analyze(audioData)

	print("Pitching...")
	lyricsPitched = pitch(metaData, lyrics, modelOutput)

	print("Writing file...")
	writeFile(metaData, lyricsPitched)

	processTime = time() - startTime
	print("Completed in " + str(format(processTime, ".3f")) + " seconds.")