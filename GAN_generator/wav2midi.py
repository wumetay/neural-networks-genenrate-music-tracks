import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from multiprocessing import Pool

from music21 import *
from scipy.signal import savgol_filter


def peaks(seq):
    data = []
    for i, x in enumerate(seq):
        if i <= 1 or i >= len(seq) - 2:
            continue
    #    if seq[i - 2] < seq[i - 1] < x and seq[i + 2] < seq[i + 1] < x:
        data.append((i, x))
    return sorted(data, key=lambda x: -x[1])


def db_to_vol(db,freq):
    transformed = np.fix(np.float((127 - 3 * abs(db))) * np.exp(-freq / 2000))
    return transformed if transformed > 32 else 0


def keydiff(freq1, freq2):
    return abs(12 * np.log2(freq1 / freq2))


def make_note(freq, dur, vol):
    n = note.Note()
    p = pitch.Pitch()
    p.frequency = freq
    n.pitch = p
    n.duration = duration.Duration(dur)
    n.volume.velocity = vol
    return n


def make_stream(top_freqs, keydiff_threshold=1):
    s = stream.Stream()

    freqs = np.array([f for (f, i) in top_freqs])
    intensities = np.array([i for (f, i) in top_freqs])
    default_dur = 0.25  # 16th note

    for voice, ints in zip(freqs.T, intensities.T):
        par = stream.Part()
        last_freq = voice[0]
        dur = default_dur
        vol = db_to_vol(ints[0],last_freq)

        for note_idx in range(1, len(voice)):
            if keydiff(voice[note_idx], last_freq) >= keydiff_threshold:
                n = make_note(last_freq, dur, vol)
                par.append(n)

                # reset
                last_freq = voice[note_idx]
                dur = default_dur
                vol = db_to_vol(ints[note_idx], last_freq)
            else:
                dur += default_dur

        n = make_note(last_freq, dur, vol)
        par.append(n)
        s.insert(0, par)
    return s  # .chordify()


ZERO_VOLUME = -80  # dB

def mute_low_volume(seq):
    return [x if x > -60 else ZERO_VOLUME for x in seq]


def make_bin2freq(sr, n_fft):
    return dict(enumerate(librosa.fft_frequencies(sr=sr, n_fft=n_fft)))


def compute_top_frequencies(spec, n_peaks):
    bin2freq = make_bin2freq(sr=48000, n_fft=4096)
    top_freqs = []
    for time_slice in spec.T:
        pitches = []
        intensities = []

        # remove high frequencies
        # 4096: 256 = 3000 Hz, 172 = 2015 Hz
        time_slice = time_slice[:172]

        # silence murmurs
        time_slice = mute_low_volume(time_slice)

        # filter out frequencies < 70 Hz
        for i in range(6):
            time_slice[i] = ZERO_VOLUME

        # smooth the frequencies
        time_slice = savgol_filter(time_slice, 9, 3)

        # store with intensity
        for (idx, value) in peaks(time_slice)[:n_peaks]:
            hz = bin2freq[idx]
            if hz not in pitches:
                pitches.append(hz)
                intensities.append(value)

        # account for not enough peaks (silence)
        while len(pitches) < n_peaks:
            pitches.append(1)
            intensities.append(ZERO_VOLUME)

        top_freqs.append((pitches, intensities))
    return top_freqs


def write_stream(path, s):
    s.insert(0, tempo.MetronomeMark(number=1500))
    s.write("midi", path)

def generate_midi(data, sample_rate, output_file, params):
    print(params)
    spec = librosa.stft(np.asfortranarray(data).T[0], n_fft=4096, hop_length=512)
    db = librosa.amplitude_to_db(spec, ref=np.max)
    # plot_db(db.T[100])
    top_freqs = compute_top_frequencies(db, params['n_peaks'])
    s = make_stream(top_freqs, params['keydiff_threshold'])
    write_stream(output_file, s)


def plot_spec(spec):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max), y_axis='log', x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='stft')
    plt.savefig("plot.png")


def plot_db(timeslice):
    bin2freq = dict(enumerate(librosa.fft_frequencies(sr=48000, n_fft=2048)))
    plt.figure()
    plt.plot([bin2freq[b] for b in range(0, 128)], timeslice[:128])
    plt.xlabel('freq Hz')
    plt.ylabel('dB')
    plt.xscale('log')
    plt.show()
    plt.savefig("power.png")



def wav2midi(input_wav, output_midi, params):
    data, sample_rate = sf.read(input_wav, dtype='float32')
    generate_midi(data, sample_rate, output_midi, params)


def run(args):
    return func(*args)
def func(path_a_temp,path_b_temp):
    data, sample_rate = sf.read(path_a_temp, dtype='float32')
    generate_midi(data, sample_rate,path_b_temp, {"n_peaks": 12, "keydiff_threshold": 2})


def wav_2_midi(path_):
    path_a = path_
    path_b = path_
    args_list=[]
    for i in os.listdir(path_a):
        if i[-3:] == 'wav':
            path_a_temp = path_a + '\\%s' % i
            path_b_temp = path_b + "\\%s.mid" % i.split('.wav')[0]
            args_list.append([path_a_temp,path_b_temp])
    pool = Pool(2)
    pool.map(run, args_list)
out_oath='./song/midi'
def wav2midi(p):
    if not os.path.exists(out_oath):
        os.mkdir(out_oath)
    if p[-3:] == 'wav':
        head,tail = os.path.split(p)
        name,ext = os.path.splitext(tail)
        pool = Pool(2)
        pool.map(run,[[p,p.replace('.wav','.mid').replace('wav','midi')]])
if __name__=='__main__':
    # wav_2_midi(r'song/wav/NoNoNo.wav')
    wav2midi(r'song/wav/NoNoNo.wav')