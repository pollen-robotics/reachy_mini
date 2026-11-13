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
    from scipy.signal import fftconvolve

    c = np.asarray(capture, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    c = c - c.mean()
    r = r - r.mean()

    corr = fftconvolve(c, r[::-1], mode="valid")
    # Per-position energy of the capture window, so the normalization is local:
    # a loud noise burst elsewhere in the capture can't deflate the peak.
    window_energy = fftconvolve(c**2, np.ones(len(r)), mode="valid")
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
