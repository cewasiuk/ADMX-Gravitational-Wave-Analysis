import os
import h5py
import numpy as np
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.psd import welch, interpolate


def list_binned_files(folder, suffix="_binned.h5"):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(suffix)
    )


def pick_group(f: h5py.File):
    root_keys = set(f.keys())
    if any(k in root_keys for k in ("FFT_Binned_Re", "FFT_BinnedFlat_Re", "FFT_Binned_Flat_Re")):
        return f
    keys = list(f.keys())
    if len(keys) == 1 and isinstance(f[keys[0]], h5py.Group):
        return f[keys[0]]
    raise KeyError(f"Could not auto-detect group. Keys: {keys}")


def load_fft_fd_and_meta(h5_path, prefer_flat=True):
    flat_re_names = ("FFT_BinnedFlat_Re", "FFT_Binned_Flat_Re")
    flat_im_names = ("FFT_BinnedFlat_Im", "FFT_Binned_Flat_Im")

    with h5py.File(h5_path, "r") as f:
        g = pick_group(f)
        gkeys = set(g.keys())

        if prefer_flat and any(n in gkeys for n in flat_re_names) and any(n in gkeys for n in flat_im_names):
            re_name = next(n for n in flat_re_names if n in gkeys)
            im_name = next(n for n in flat_im_names if n in gkeys)
            re_ds = g[re_name]
            im_ds = g[im_name]
            used = "flat"
        else:
            if ("FFT_Binned_Re" not in gkeys) or ("FFT_Binned_Im" not in gkeys):
                raise KeyError(f"Missing FFT datasets. Keys present: {sorted(gkeys)}")
            re_ds = g["FFT_Binned_Re"]
            im_ds = g["FFT_Binned_Im"]
            used = "raw"

        re = re_ds[:].astype(np.float64, copy=False)
        im = im_ds[:].astype(np.float64, copy=False)
        X = re + 1j * im

        f_baseband = g["f_baseband"][:] if "f_baseband" in g else None

        dt = float(f.attrs.get("delta_t", re_ds.attrs.get("delta_t", np.nan)))
        df = float(f.attrs.get("delta_f_binned", re_ds.attrs.get("delta_f", np.nan)))
        N_time = int(f.attrs.get("N_time", re_ds.attrs.get("N_time", -1)))

        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("delta_t missing or invalid")
        if not np.isfinite(df) or df <= 0:
            raise ValueError("delta_f_binned missing or invalid")
        if N_time <= 0:
            raise ValueError("N_time missing or invalid")

    return X, f_baseband, dt, df, N_time, used


def fd_to_timeseries_irfft(X_fd, dt, N_time):
    """
    Convert rFFT style FrequencySeries array to a real TimeSeries via irfft.
    Assumes X_fd corresponds to a real time series (DC..Nyquist).
    """
    x = np.fft.irfft(X_fd, n=int(N_time))
    return TimeSeries(np.asarray(x, dtype=np.float64), delta_t=float(dt))


def compute_psd_from_timeseries(
    ts: TimeSeries,
    seg_len_sec=1.0,
    seg_stride_sec=0.5,
    avg_method="median",
):
    """
    Welch PSD on a real TimeSeries.
    """
    dt = float(ts.delta_t)
    seg_len = int(round(seg_len_sec / dt))
    seg_stride = int(round(seg_stride_sec / dt))

    if seg_len < 8:
        raise ValueError("seg_len too small, increase seg_len_sec")
    if seg_stride < 1:
        raise ValueError("seg_stride too small, increase seg_stride_sec")
    if seg_len > len(ts):
        raise ValueError("seg_len bigger than time series length, decrease seg_len_sec")

    return welch(ts, seg_len=seg_len, seg_stride=seg_stride, avg_method=avg_method)


def compute_stacked_psd_welch_from_h5(
    binned_dir="./hr_data/binned_hr_data",
    suffix="_binned.h5",
    prefer_flat=True,
    seg_len_sec=1.0,
    seg_stride_sec=0.5,
    avg_method="median",
    stack_method="median",
    psd_floor=None,
    f_low=None,
    f_high=None,
    verbose=True,
):
    """
    Fixes the large SNR problem by building the PSD from the time series using Welch,
    instead of using a periodogram on the stored FFT arrays.

    This avoids:
      1) incorrect PSD normalization from averaged bin groups
      2) mismatches caused by receiver flattening and polynomial artifacts
      3) fragile bin_factor corrections

    Returns:
      psd_final : FrequencySeries (real, positive)
      freqs     : numpy array of frequency values [Hz]
      used_files, skipped
    """
    files = list_binned_files(binned_dir, suffix=suffix)
    if len(files) == 0:
        raise RuntimeError(f"No files found in {binned_dir} with suffix {suffix}")

    psd_rows = []
    used_files = []
    skipped = []

    ref_df = None
    ref_len = None

    for path in files:
        try:
            X_fd, f_baseband, dt, df, N_time, used = load_fft_fd_and_meta(path, prefer_flat=prefer_flat)

            ts = fd_to_timeseries_irfft(X_fd, dt=dt, N_time=N_time)

            psd = compute_psd_from_timeseries(
                ts,
                seg_len_sec=seg_len_sec,
                seg_stride_sec=seg_stride_sec,
                avg_method=avg_method,
            )

            # force onto the exact (df, length) implied by your stored FFT grid
            # your stored FFT grid spacing is df and length is len(X_fd)
            psd = interpolate(psd, float(df), int(len(X_fd)))

            if ref_df is None:
                ref_df = float(psd.delta_f)
                ref_len = len(psd)
            else:
                if float(psd.delta_f) != ref_df:
                    raise ValueError(f"delta_f mismatch after interpolate: {psd.delta_f} vs ref {ref_df}")
                if len(psd) != ref_len:
                    raise ValueError(f"PSD length mismatch after interpolate: {len(psd)} vs ref {ref_len}")

            psd_arr = np.asarray(psd.numpy(), dtype=np.float64)

            # optional band limiting for stability
            if f_low is not None or f_high is not None:
                freqs = np.arange(len(psd_arr)) * ref_df
                mask = np.ones_like(freqs, dtype=bool)
                if f_low is not None:
                    mask &= (freqs >= float(f_low))
                if f_high is not None:
                    mask &= (freqs <= float(f_high))
                # outside the band, set PSD large so those bins do not contribute
                big = np.nanmax(psd_arr[np.isfinite(psd_arr)]) if np.any(np.isfinite(psd_arr)) else 1.0
                psd_arr = np.where(mask, psd_arr, big)

            # floor if you want one, but keep it sane
            if psd_floor is not None:
                psd_arr = np.maximum(psd_arr, float(psd_floor))

            # clean nonfinite
            finite = np.isfinite(psd_arr)
            if not np.all(finite):
                if np.any(finite):
                    fill = np.median(psd_arr[finite])
                else:
                    fill = 1.0
                psd_arr = np.where(finite, psd_arr, fill)

            psd_rows.append(psd_arr)
            used_files.append(path)

        except Exception as e:
            skipped.append((path, str(e)))

    if len(psd_rows) == 0:
        msg = "No PSDs computed. First few errors:\n" + "\n".join(
            f"{os.path.basename(p)} -> {r}" for p, r in skipped[:10]
        )
        raise RuntimeError(msg)

    arr = np.vstack(psd_rows)

    if stack_method == "median":
        psd_stack = np.median(arr, axis=0)
    elif stack_method == "mean":
        psd_stack = np.mean(arr, axis=0)
    else:
        raise ValueError("stack_method must be 'median' or 'mean'")

    psd_final = FrequencySeries(psd_stack.astype(np.float64), delta_f=float(ref_df))
    freqs = np.arange(len(psd_final)) * float(psd_final.delta_f)

    if verbose:
        p = psd_final.numpy()
        finite = np.isfinite(p)
        pmin = float(np.min(p[finite]))
        pmed = float(np.median(p[finite]))
        pmax = float(np.max(p[finite]))
        print("PSD stats:")
        print("  min   =", pmin)
        print("  median=", pmed)
        print("  max   =", pmax)
        print("Used files:", len(used_files), "Skipped:", len(skipped))
        # show smallest bins to catch blowups
        idx = np.argsort(p[finite])[:10]
        print("10 smallest PSD values:", p[finite][idx])

    return psd_final, freqs, used_files, skipped


# Example usage:
# psd_fd, freqs, used, skipped = compute_stacked_psd_welch_from_h5(
#     binned_dir="./hr_data/binned_hr_data",
#     prefer_flat=True,        # match whatever you matched filtered against
#     seg_len_sec=1.0,
#     seg_stride_sec=0.5,
#     avg_method="median",
#     stack_method="median",
#     psd_floor=None,          # keep None first, only add if needed
#     f_low=10.0,              # set if your matched filter uses low cut
#     f_high=None,
#     verbose=True,
# )