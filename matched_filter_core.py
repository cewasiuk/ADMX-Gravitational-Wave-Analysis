import numpy as np
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.psd import interpolate
from pycbc.filter import matched_filter_core


def normalize_template_fd(template_fd, psd_fd, f_low=None, f_high=None, psd_floor=1e-60):
    df = float(template_fd.delta_f)
    freqs = np.arange(len(template_fd)) * df

    psd = np.asarray(psd_fd.numpy(), dtype=np.float64)
    psd = np.where(np.isfinite(psd), psd, psd_floor)
    psd = np.maximum(psd, psd_floor)

    t = np.asarray(template_fd.numpy(), dtype=np.complex128)

    m = np.ones_like(freqs, dtype=bool)
    if f_low is not None:
        m &= freqs >= f_low
    if f_high is not None:
        m &= freqs <= f_high

    norm2 = 4.0 * df * np.sum((np.abs(t[m])**2) / psd[m])
    t_unit = t / np.sqrt(norm2)

    return FrequencySeries(t_unit, delta_f=df)


def matched_filter_on_common_grid(data_fd, template_fd, psd_fd,
                                 f_low=None, f_high=None,
                                 psd_floor=1e-60,
                                 return_time_domain_snr=True):
    """
    Matched filter using pycbc.filter.matched_filter_core, assuming:
      - data_fd, template_fd, psd_fd are all on the same frequency grid (same delta_f, same length)
      - data_fd and template_fd are complex FrequencySeries (rFFT-style, DC..Nyquist)
      - psd_fd is a real (positive) FrequencySeries

    Inputs
    ------
    data_fd : FrequencySeries (complex)
    template_fd : FrequencySeries (complex)
    psd_fd : FrequencySeries (real)
    f_low : float or None (Hz)
    f_high: float or None (Hz)
    psd_floor : float
        Floor applied to PSD to avoid divide-by-zero and insane whitening.
    return_time_domain_snr : bool
        If True, returns the time-series SNR (usual for peak finding).
        If False, returns the complex frequency-domain matched-filter output (advanced use).

    Returns
    -------
    snr_ts : TimeSeries (complex)  (if return_time_domain_snr=True)
    peak   : dict with peak info
    """

    # ---- Coerce inputs to FrequencySeries and sanity check grids
    if not isinstance(data_fd, FrequencySeries):
        raise TypeError("data_fd must be a pycbc.types.FrequencySeries")
    if not isinstance(template_fd, FrequencySeries):
        raise TypeError("template_fd must be a pycbc.types.FrequencySeries")
    if not isinstance(psd_fd, FrequencySeries):
        raise TypeError("psd_fd must be a pycbc.types.FrequencySeries")

    if data_fd.delta_f != template_fd.delta_f:
        raise ValueError("data_fd.delta_f != template_fd.delta_f")
    if len(data_fd) != len(template_fd):
        raise ValueError("len(data_fd) != len(template_fd)")

    df = float(data_fd.delta_f)

    # PSD must match grid; if not, interpolate it
    if psd_fd.delta_f != df or len(psd_fd) != len(data_fd):
        psd_fd = interpolate(psd_fd, df, len(data_fd))

    # ---- PSD safety
    psd_arr = np.asarray(psd_fd.numpy(), dtype=np.float64)
    psd_arr = np.where(np.isfinite(psd_arr), psd_arr, psd_floor)
    psd_arr = np.maximum(psd_arr, psd_floor)
    psd_fd = FrequencySeries(psd_arr, delta_f=df)

    # ---- Optional: enforce DC = large PSD so DC doesn’t dominate
    if len(psd_fd) > 0:
        psd_fd[0] = max(psd_fd[0], psd_floor)

    # ---- Call matched_filter_core
    # This returns the matched-filter SNR time series by default in PyCBC workflows.
    # We request return_complex=True to preserve complex SNR (phase info).
    snr = matched_filter_core(
        template_fd,
        data_fd,
        psd=psd_fd,
        low_frequency_cutoff=f_low,
        high_frequency_cutoff=f_high,
    )
    print(type(snr))
    if isinstance(snr, tuple):
        print([type(x) for x in snr])


    # Some PyCBC versions return (snr, corr, norm) or similar; handle robustly:
    if isinstance(snr, tuple):
        snr = snr[0]

    # Ensure time domain (usual usage) unless you explicitly want the FD output
    if return_time_domain_snr:
        out = snr.to_timeseries()
        snr_ts = out[0] if isinstance(out, tuple) else out
    else:
        snr_ts = snr  # frequency-domain object

    # ---- Peak finding (time-domain)
    if return_time_domain_snr:
        snr_abs = np.abs(snr_ts.numpy())
        i_peak = int(np.argmax(snr_abs))
        peak = {
            "snr_peak": float(snr_abs[i_peak]),
            "snr_peak_complex": complex(snr_ts[i_peak]),
            "idx_peak": i_peak,
            "t_peak_sec": float(i_peak * snr_ts.delta_t),
            "delta_t": float(snr_ts.delta_t),
            "delta_f": df,
        }
    else:
        # If you keep it in frequency domain, “peak” is not meaningful in the same way
        peak = {
            "delta_f": df,
            "note": "return_time_domain_snr=False, no time-domain peak computed",
        }

    return snr_ts, peak


# -------------------------
# Minimal example call
# -------------------------
# snr_ts, peak = matched_filter_on_common_grid(
#     data_fd=data_fd,            # complex FrequencySeries (your processed FFT_BinnedFlat)
#     template_fd=template_fd,    # complex FrequencySeries (your voltage template with Lorentzian applied)
#     psd_fd=psd_fd,              # real FrequencySeries (your stacked PSD)
#     f_low=10.0,                 # choose appropriate band for your signal/template
#     f_high=None,
# )
# print("Peak SNR =", peak["snr_peak"], "at t =", peak["t_peak_sec"], "sec")