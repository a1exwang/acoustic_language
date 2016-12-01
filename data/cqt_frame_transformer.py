import numpy as np
import scipy
import scipy.io.wavfile
import librosa
from .maps_constants import *


class Transformer:
    MAPS_WAV_FILE_MAX_VALUE = 2**15 - 1

    @staticmethod
    def midis_to_one_hot(midi_list):
        one_hot = np.zeros([SEMITONES_ON_PIANO], dtype='float32')
        for midi in midi_list:
            one_hot[midi - FIRST_PIANO_KEY_MIDI_VALUE] = 1
        return one_hot

    def __init__(self, sample_rate=44100, frame_size=2048, bins_per_semitone=3):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.bins_per_semitone = bins_per_semitone
        self.frame_rate = float(sample_rate) / frame_size

    def midis_to_freqs(self, frame_count, midi_seq):
        midi_freqs = np.zeros([frame_count, SEMITONES_ON_PIANO])
        frame_length = 1 / self.frame_rate

        for (start, end, midi) in midi_seq:
            start_frame_index = int(start / frame_length)
            end_frame_index = int(end / frame_length)
            for frame_index in range(start_frame_index, end_frame_index):
                midi_freqs[frame_index, midi - FIRST_PIANO_KEY_MIDI_VALUE] = 1

        return midi_freqs

    def get_data(self, file_path):
        (sample_rate, data) = scipy.io.wavfile.read(file_path, mmap=True)
        data = data.astype('float32')
        assert(sample_rate == self.sample_rate)
        # Merge into 1 channel
        part = np.sum(data, axis=1)

        # CQT, cqt.shape = (bins, time)
        cqt = librosa.cqt(part,
                          sr=sample_rate,
                          hop_length=self.frame_size,
                          fmin=librosa.midi_to_hz(FIRST_PIANO_KEY_MIDI_VALUE),
                          bins_per_octave=self.bins_per_semitone * SEMITONES_PER_OCTAVE,
                          n_bins=self.bins_per_semitone * SEMITONES_ON_PIANO,
                          real=True)
        cqt = cqt.swapaxes(0, 1)

        frame_count = cqt.shape[0]
        txt_file_path = file_path[:-3] + 'txt'
        midi_seq = []
        with open(txt_file_path, 'r') as f:
            lines = f.read().split("\n")
            for line_no, line in enumerate(lines):
                if line_no != 0 and len(line) > 0:
                    [start_time, end_time, midi_val] = map(lambda x: float(x), line.split())
                    midi_seq.append((start_time, end_time, int(midi_val)))

        midi_freqs = self.midis_to_freqs(frame_count, midi_seq)
        return cqt, midi_freqs

