import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def read_audio(file_path):
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data

def dominant_frequency(data, sample_rate):
    n = len(data)
    fft = np.fft.fft(data)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)
    
    # Only consider positive frequencies
    fft = fft[:n//2]
    freqs = freqs[:n//2]
    
    idx = np.argmax(np.abs(fft))
    dom_freq = freqs[idx]
    return abs(dom_freq)

def frequency_to_midi(frequency):
    return 69 + 12 * np.log2(frequency / 440.0)

def midi_to_hue(midi_note, min_midi=21, max_midi=108):
    midi_note = np.clip(midi_note, min_midi, max_midi)
    hue = (midi_note - min_midi) / (max_midi - min_midi)
    return hue

def frequency_to_color(frequency):
    min_midi = 21  # A0
    max_midi = 108  # C8
    
    midi_note = frequency_to_midi(frequency)
    hue = midi_to_hue(midi_note, min_midi, max_midi)
    
    saturation = 1.0  # Full saturation
    value = 1.0  # Full brightness
    color = hsv_to_rgb((hue, saturation, value))
    
    # Debug output
    print(f"Frequency: {frequency}, MIDI Note: {midi_note}, Hue: {hue}, Color: {color}")
    
    return color

def sound_to_color(file_path):
    sample_rate, data = read_audio(file_path)
    
    if len(data.shape) > 1:  # Stereo to mono conversion if needed
        data = np.mean(data, axis=1)
    
    dom_freq = dominant_frequency(data, sample_rate)
    
    # Debug output
    print(f"Dominant frequency: {dom_freq} Hz")
    
    color = frequency_to_color(dom_freq)
    
    return color

def visualize_color(color):
    plt.figure(figsize=(2, 2))
    plt.imshow([[color]])
    plt.axis('off')
    plt.show()

# Example usage:
file_path = '/Users/auxni/Documents/Science/Programming/Synesthesia/Spring.wav'  # Replace with the path to your audio file
color = sound_to_color(file_path)
visualize_color(color)


