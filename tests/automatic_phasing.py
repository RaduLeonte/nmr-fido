import numpy as np
import nmr_fido as nf
import nmrglue as ng
from pprint import pprint
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit, least_squares
import random


dic, data = ng.pipe.read("tests/test2d.fid")

data = nf.NMRData(
    data,
    axes= [
        {
            "label": "15N",
            "SW": 5555.55615234375,
            "ORI": 3333.448974609375,
            "OBS": 50.64799880981445,
            "interleaved_data": True,
        },
        {
            "label": "13C",
            'SW': 50000.0,
            'ORI': -18053.66015625,
            'OBS': 125.69100189208984,
        },
    ]
)


def simulate_signal(t: np.ndarray, oscillators_info: list) -> np.ndarray:
    signal = np.zeros(t.size, dtype=np.complex64)
    for amplitude, freq, damping in oscillators_info:
        signal += amplitude * np.exp(-damping * t) * np.exp(1j * 2 * np.pi * freq * t)
    return signal

def complex_lorentzian(f, A, f0, gamma, phi):
    """
    A: amplitude
    f0: center frequency
    gamma: linewidth (HWHM)
    phi: phase in degrees
    """
    return A / (1j * (f - f0) + gamma) * np.exp(1j * phi * np.pi / 180)

def real_lorentzian(f, A, f0, gamma, phi):
    """
    Real part of phase-shifted Lorentzian
    """
    c = complex_lorentzian(f, A, f0, gamma, phi)
    return c.real


def residuals_real(params, f, data_real):
    A, f0, gamma, phi = params
    model_real = real_lorentzian(f, A, f0, gamma, phi)
    res = model_real - data_real
    return res


def get_phase(freq: np.ndarray, ft: np.ndarray, fit_range_hz: tuple, max_iter: int = 5) -> float:
    idx_start = np.abs(freq - fit_range_hz[0]).argmin()
    idx_end = np.abs(freq - fit_range_hz[1]).argmin()
    xdata = freq[idx_start:idx_end]
    corrected_ft = ft.copy()
    ext_phases = []
    ext_freq_centers = []
    bounds = [
            (     0, min(xdata),                               0, -180),
            (np.inf, max(xdata), np.abs(max(xdata) - min(xdata)),  180)
        ]
    print(xdata[0], xdata[-1])
    prev_guess = None
    for i in range(max_iter):
        ydata_real = corrected_ft.real[idx_start: idx_end]

        p0 = [
            prev_guess[0] if prev_guess is not None else float(np.max(np.abs(ydata_real))), # Amplitude
            prev_guess[1] if prev_guess is not None else float(min(xdata) + (max(xdata) - min(xdata))/2), # Center frequency
            prev_guess[2] if prev_guess is not None else float(np.abs(max(xdata) - min(xdata)) / 2), # Width
            0 # Phase in degrees
        ]
        result = least_squares(residuals_real, p0, bounds=bounds, args=(xdata, ydata_real))
        popt = result.x
        print(popt)
        ext_phase = float(popt[3])
        ext_phases.append(ext_phase)
        ext_freq_centers.append(float(popt[1]))

        corrected_ft = nf.PS(corrected_ft, p0=ext_phase)
        prev_guess = popt

    print(ext_phases)
    total_phase = float(sum(ext_phases) % 360)
    freq_center = float(np.average(ext_freq_centers))

    return total_phase, freq_center
    

t = np.linspace(0, 10, 5000).astype(np.complex64)
oscillators_info = [
    (1, 2,  3),
    (2, 100, 5),
]

signal = simulate_signal(t, oscillators_info)
ft = nf.FT(signal)
phase_0 = random.uniform(-360, 360)
phase_1 = random.uniform(-360, 360)
ft = nf.PS(ft, p0=phase_0, p1=phase_1)
#ft = nf.DI(ft)

freq = np.fft.fftshift(np.fft.fftfreq(t.size, d=(t[1] - t[0]).real))

margin = 25
target_phase_0, target_freq_0 = get_phase(
    freq, ft,
    (-oscillators_info[0][1] - margin, -oscillators_info[0][0] + margin)
)

target_phase_1, target_freq_1 = get_phase(
    freq, ft,
    (-oscillators_info[1][1] - margin, -oscillators_info[0][0] + margin)
)
min_x = min(freq)
max_x = max(freq)
target_idx_0 = (target_freq_0 - min_x) / (max_x - min_x)
target_idx_1 = (target_freq_1 - min_x) / (max_x - min_x)
p1 = (target_phase_1 - target_phase_0) / (target_idx_1  - target_idx_0)
p0 = target_phase_0 - p1*target_idx_0

#corrected_ft = nf.PS(ft, p0=p0, p1=p1)
print(p0, p1)
p0 = (p0 - 360)
p1 = (p1)
print(p0, p1)
corrected_ft = nf.PS(ft, p0=p0, p1=p1)


phase_corr_1 = p0 + p1*(np.arange(freq.size) / freq.size)


fig, axs = plt.subplots(2,1, figsize=(7,5), layout="constrained")

axs[0].plot(freq, ft.real, label=f'Phase: p0={float(phase_0):.2f}  p1={float(phase_1):.2f}')
axs[0].legend()

axs[1].plot(freq, corrected_ft.real, label=f"Phase: p0={p0:.2f} p1={p1:.2f}")
axs[1].legend()


for ax in axs:
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")


""" data = nf.SP(data, off=0.35, end=0.98, pow=1, c=1.0)
data = nf.ZF(data, size=4096)
data = nf.FT(data)
data = nf.PS(data, p0=0.0, p1=0.0)
data = nf.DI(data)
data = nf.EXT(data, x1="70ppm", xn="40ppm")

data = nf.TP(data)

data = nf.SP(data, off=0.35, end=0.9, pow=1, c=0.5)
data = nf.ZF(data, size=2048)
data = nf.FT(data)
data = nf.PS(data, p0=0.0, p1=0.0)
data = nf.DI(data)
data = nf.EXT(data, x1="135ppm", xn="100ppm")

data = nf.TP(data) 

fig, ax = plt.subplots(1,1, figsize=(7,5), layout="constrained")

levels = 30_000 * 1.2 ** np.arange(16)

for contour_colour, contour_levels in zip(["crimson", "dodgerblue"], [levels[::-1] * -1, levels]):
    ax.contour(
        data,
        levels = contour_levels,
        extent = data.extent(),
        colors = contour_colour,
    )

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
ax.set_xlabel(data.axes[-1]["label"])
ax.set_ylabel(data.axes[-2]["label"]) """


plt.show()