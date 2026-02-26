import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from pycbc.types import FrequencySeries

# ----------------------------
# user settings
# ----------------------------
binned_dir = "./hr_data/binned_hr_data"
suffix = "_binned.h5"
prefer_flat = True          # use FFT_BinnedFlat_* if present
stack_method = "median"     # "median" or "mean"
skip_dc = True              # skip DC bin in plot


# ----------------------------
# helpers
# ----------------------------
def list_binned_files(folder, suffix="_binned.h5"):
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(suffix)
    )

def pick_group(f):
    # if FFT datasets are at root, return root
    root_keys = set(f.keys())
    if any(k in root_keys for k in ("FFT_Binned_Re", "FFT_BinnedFlat_Re", "FFT_Binned_Flat_Re")):
        return f

    # if exactly one group exists, use it
    keys = list(f.keys())
    if len(keys) == 1 and isinstance(f[keys[0]], h5py.Group):
        return f[keys[0]]

    # otherwise, fail loud so you notice the file layout
    raise KeyError(f"Could not auto-detect group. Keys: {keys}")

def load_complex_fft_from_h5(h5_path, prefer_flat=True):
    with h5py.File(h5_path, "r") as f:
        g = pick_group(f)
        gkeys = set(g.keys())

        # support both naming styles
        flat_re_names = ["FFT_BinnedFlat_Re", "FFT_Binned_Flat_Re"]
        flat_im_names = ["FFT_BinnedFlat_Im", "FFT_Binned_Flat_Im"]

        if prefer_flat and any(n in gkeys for n in flat_re_names) and any(n in gkeys for n in flat_im_names):
            re_name = next(n for n in flat_re_names if n in gkeys)
            im_name = next(n for n in flat_im_names if n in gkeys)
            re = g[re_name][:]
            im = g[im_name][:]
            used = "flat"
            ds = g[re_name]
        else:
            if ("FFT_Binned_Re" not in gkeys) or ("FFT_Binned_Im" not in gkeys):
                raise KeyError(f"Missing FFT datasets. Keys present: {sorted(gkeys)}")
            re = g["FFT_Binned_Re"][:]
            im = g["FFT_Binned_Im"][:]
            used = "raw"
            ds = g["FFT_Binned_Re"]

        # frequency axis: prefer baseband (same across different RF tunings)
        f_baseband = g["f_baseband"][:] if "f_baseband" in g else None

        # metadata (try file attrs first, then dataset attrs)
        # your files (per your save snippet) store these in file attrs:
        dt = float(g.file.attrs.get("delta_t", ds.attrs.get("delta_t", np.nan)))
        df = float(g.file.attrs.get("delta_f_binned", ds.attrs.get("delta_f", np.nan)))
        N_time = int(g.file.attrs.get("N_time", ds.attrs.get("N_time", -1)))

        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("delta_t missing/invalid (file attrs or dataset attrs)")
        if not np.isfinite(df) or df <= 0:
            raise ValueError("delta_f missing/invalid (file attrs or dataset attrs)")
        if N_time <= 0:
            raise ValueError("N_time missing/invalid (file attrs or dataset attrs)")

        X = re.astype(np.float64) + 1j * im.astype(np.float64)

    return X, f_baseband, dt, df, N_time, used

def onesided_periodogram_from_rfft(X, dt, N_time):
    """
    X is a one-sided rFFT-like spectrum (length K = N/2+1, or binned version of it).
    We treat it as coming from an rFFT of a real time series sampled at dt.

    Scaling (one-sided):
      T = N_time * dt
      Xc = dt * X   (continuous-FT approx)
      PSD = (2/T) * |Xc|^2 = (2*dt/N_time) * |X|^2   for interior bins
      DC and Nyquist (if present) are not doubled.
    """
    K = len(X)
    psd = (2.0 * dt / N_time) * (np.abs(X) ** 2)

    # DC not doubled
    psd[0] *= 0.5

    # Nyquist bin exists only if original N_time even; we assume it does and your saved rfft kept it.
    # If your binning dropped/changed it, this is approximate; harmless for weighting.
    if K > 1:
        psd[-1] *= 0.5

    # guard
    psd = np.maximum(psd, 0.0)
    return psd

def stack_psds(psd_list, method="median"):
    arr = np.vstack(psd_list)
    if method == "median":
        out = np.median(arr, axis=0)
    elif method == "mean":
        out = np.mean(arr, axis=0)
    else:
        raise ValueError("method must be 'median' or 'mean'")
    return out


# ----------------------------
# compute stacked PSD from directory
# ----------------------------
files = list_binned_files(binned_dir, suffix=suffix)
if len(files) == 0:
    raise RuntimeError(f"No files found in {binned_dir} with suffix {suffix}")

psd_rows = []
used_files = []
skipped = []

ref_len = None
ref_df = None
ref_fbase = None

for path in files:
    try:
        X, f_baseband, dt, df, N_time, used = load_complex_fft_from_h5(
            path, prefer_flat=prefer_flat
        )

        psd = onesided_periodogram_from_rfft(X, dt=dt, N_time=N_time)

        if ref_len is None:
            ref_len = len(psd)
            ref_df = df
            ref_fbase = f_baseband
        else:
            if len(psd) != ref_len:
                raise ValueError(f"length mismatch: {len(psd)} vs ref {ref_len}")
            if df != ref_df:
                raise ValueError(f"delta_f mismatch: {df} vs ref {ref_df}")
            # baseband axis can be None in some files; if present, check consistency
            if (ref_fbase is not None) and (f_baseband is not None):
                if len(f_baseband) != len(ref_fbase):
                    raise ValueError("f_baseband length mismatch")
                # loose check (binning float noise)
                if not np.allclose(f_baseband, ref_fbase, rtol=0, atol=1e-9):
                    raise ValueError("f_baseband values mismatch")

        psd_rows.append(psd)
        used_files.append(path)

    except Exception as e:
        skipped.append((path, str(e)))

if len(psd_rows) == 0:
    print("All files skipped. First few errors:")
    for p, r in skipped[:10]:
        print(p, "->", r)
    raise RuntimeError("No PSDs computed.")

psd_stack = stack_psds(psd_rows, method=stack_method)
psd_final = FrequencySeries(psd_stack, delta_f=ref_df)

print("done")
print("n_used:", len(used_files))
print("n_skipped:", len(skipped))
if len(skipped):
    print("first few skipped:")
    for p, r in skipped[:5]:
        print("  ", os.path.basename(p), "->", r)

# frequency axis for plotting (baseband preferred)
if ref_fbase is not None:
    freqs = np.asarray(ref_fbase)
else:
    freqs = np.arange(len(psd_final)) * psd_final.delta_f

# ----------------------------
# plot
# ----------------------------
i0 = 1 if skip_dc else 0

plt.figure(figsize=(9, 5))
plt.loglog(freqs[i0:], psd_final.numpy()[i0:])
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [ (ADC units)^2 / Hz ]")
plt.title(f"Stacked one-FFT-per-segment PSD ({stack_method}, prefer_flat={prefer_flat})")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.show()

# ----------------------------
# optional: save PSD to h5
# ----------------------------
out_psd = "stacked_psd_periodogram.h5"
with h5py.File(out_psd, "w") as f:
    f.create_dataset("PSD", data=psd_final.numpy().astype(np.float64))
    f["PSD"].attrs["delta_f"] = float(psd_final.delta_f)
    f["PSD"].attrs["n_freq"] = int(len(psd_final))
    f.attrs["source_dir"] = binned_dir
    f.attrs["suffix"] = suffix
    f.attrs["prefer_flat"] = int(bool(prefer_flat))
    f.attrs["stack_method"] = stack_method
    f.attrs["n_used"] = len(used_files)
    f.attrs["n_skipped"] = len(skipped)
    if ref_fbase is not None:
        f.create_dataset("f_baseband", data=np.asarray(ref_fbase, dtype=np.float64))

print("saved:", out_psd)