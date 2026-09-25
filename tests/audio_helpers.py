"""Signal-analysis helpers shared by the audio test suites.

Test-only code: nothing in the shipped package uses these. They live here —
importable as ``tests.audio_helpers`` from both ``tests/unit_tests`` and
``tests/hardware`` (the empty root ``conftest.py`` puts the repo root on
``sys.path``) — rather than in ``reachy_mini.media.audio_utils``, to keep the
public API surface down.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def log_sweep(
    duration_s: float,
    f0: float,
    f1: float,
    rate: int,
    amplitude: float,
) -> npt.NDArray[np.float64]:
    """Logarithmic sine sweep with 10 ms raised-cosine fades (no click).

    The standard excitation for acoustic measurements: matched filtering it
    (``correlation_peak``) gives a sharp lag estimate, and its spectrum against
    a capture's gives the frequency response.
    """
    t = np.arange(int(duration_s * rate)) / rate
    ratio = f1 / f0
    phase = (
        2 * np.pi * f0 * duration_s / np.log(ratio) * (ratio ** (t / duration_s) - 1)
    )
    sweep = amplitude * np.sin(phase)
    fade = int(0.01 * rate)
    window = np.hanning(2 * fade)
    sweep[:fade] *= window[:fade]
    sweep[-fade:] *= window[fade:]
    return sweep


def noise_floor(noise: npt.NDArray[np.floating]) -> float:
    """Transient-resistant noise level, as a Gaussian-equivalent RMS.

    Plain RMS over a short noise window is dominated by whatever transient
    happened to land in it — a chair, a voice, a fan. The median of |x|
    ignores brief spikes; the 1.2533 factor (sqrt(pi/2)) converts it back to
    the RMS of an equivalent Gaussian so ratios stay meaningful.
    """
    return float(np.median(np.abs(noise)) * 1.2533)


def loudest_block_rms(track: npt.NDArray[np.floating], block: int) -> float:
    """RMS of the loudest ``block``-sample window of ``track``.

    A short sound inside a longer capture is diluted by the silence around it
    in a whole-capture RMS; the loudest block isolates the sound itself.
    """
    usable = len(track) // block * block
    if usable == 0:
        return float(np.sqrt(np.mean(track**2))) if len(track) else 0.0
    blocks = np.asarray(track[:usable]).reshape(-1, block)
    return float(np.sqrt((blocks**2).mean(axis=1)).max())


def correlation_peak(
    capture: npt.NDArray[np.floating],
    reference: npt.NDArray[np.floating],
    samplerate: int,
) -> tuple[float, float]:
    """Locate ``reference`` inside ``capture`` by normalized cross-correlation.

    A matched filter: the peak height says how much of the reference's
    *waveform* is present, the peak position says when it starts.  Robust to
    level and delay, and — unlike :func:`spectral_cosine` — to heavy spectral
    coloration, which makes it the right presence detector for an *acoustic*
    path (small speaker, EQ, room, mic DSP).  Measured on a real robot
    speaker→mic loopback: ~0.28 with the sound present vs ~0.02-0.04 for echo
    cancellation eating it or unrelated noise.

    Note it is scale invariant: a 50x quieter copy scores identically, so it
    answers "is the right sound there", never "is it loud enough" — pair it
    with a level check.

    Both signals must be at ``samplerate`` and ``capture`` must be at least as
    long as ``reference``.

    Args:
        capture: The recording to search in, shape ``(N,)``.
        reference: The signal to look for, shape ``(M,)``, ``M <= N``.
        samplerate: Common sample rate in Hz (used only for the lag).

    Returns:
        ``(peak, lag_s)`` — peak of the normalized cross-correlation in
        ``[0, 1]``, and the reference's start offset within the capture in
        seconds.

    """
    c = np.asarray(capture, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    c = c - c.mean()
    r = r - r.mean()

    # numpy-only on purpose: the SDK dropped its scipy runtime dependency
    # (#1342), so a plain `pip install reachy-mini` test client has no scipy.
    n_valid = len(c) - len(r) + 1
    if n_valid < 1:
        raise ValueError("capture must be at least as long as reference")

    # FFT cross-correlation. corr[k] = sum_i c[i + k] * r[i], i.e. the
    # correlation of the capture window starting at k against the reference.
    size = 1 << (len(c) + len(r) - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(c, size) * np.conj(np.fft.rfft(r, size)), size)
    corr = corr[:n_valid]

    # Per-position energy of the capture window, so the normalization is local:
    # a loud noise burst elsewhere in the capture can't deflate the peak.
    # A cumsum gives the exact sliding-window sum in one pass.
    cumulative = np.concatenate(([0.0], np.cumsum(c**2)))
    window_energy = cumulative[len(r) :] - cumulative[:n_valid]
    ncc = corr / (np.linalg.norm(r) * np.sqrt(np.clip(window_energy, 1e-12, None)))

    k = int(np.argmax(np.abs(ncc)))
    return float(np.abs(ncc[k])), k / samplerate


def spectral_cosine(
    a: npt.NDArray[np.floating],
    b: npt.NDArray[np.floating],
    n: int | None = None,
) -> float:
    """Cosine similarity of the Hann-windowed magnitude spectra of two signals.

    Frequency-domain so it's timing-invariant — a partial capture or a start
    offset doesn't matter, only whether the same sound is present.

    Best suited to digitally clean paths (measured 0.75-0.84 on the virtual
    audio loopback vs ~0.10 for noise).  On a real *acoustic* path the
    speaker/EQ/room coloration compresses the separation to the point of
    uselessness (~0.18 present vs ~0.12 for white noise, measured on-robot) —
    use :func:`correlation_peak` there instead.

    Both signals must be at the **same sample rate**.  The FFT size is shared,
    so a given frequency lands in a different bin at a different rate:
    identical audio compared across 16 kHz and 44.1 kHz scores near zero.
    Resample one to the other's rate first.

    The reference also needs spectral *structure* for the score to
    discriminate.  Speech and music do; a sweep or noise does not — two
    unrelated broadband signals score ~0.5, so this is the wrong metric for
    those.

    Args:
        a: First signal, shape ``(N,)``.
        b: Second signal, shape ``(M,)``.  Need not match ``a`` in length, but
            must match in sample rate.
        n: FFT size.  Defaults to the next power of two covering the longer
            signal, so neither is truncated — an ``n`` shorter than a signal
            silently crops it to its first ``n`` samples, making the score
            depend on *when* the sound occurs, exactly what this metric is
            meant to be invariant to.

    Returns:
        Similarity in ``[0, 1]``.

    """
    if n is None:
        n = 1 << (max(len(a), len(b)) - 1).bit_length()

    def spectrum(x: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        xf = np.asarray(x, dtype=np.float64)
        mag = np.abs(np.fft.rfft(xf * np.hanning(len(xf)), n))
        return np.asarray(mag / (np.linalg.norm(mag) + 1e-9), dtype=np.float64)

    return float(np.dot(spectrum(a), spectrum(b)))
