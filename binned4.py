"""
binned4.py (receiver-corrected refactor)
"""

import os
import struct
import json
import numpy as np
import h5py
import array
import yaml
import pyfftw
import argparse
import pycbc
import pandas as pd
from config_file_handling import get_intermediate_data_file_name


# ------------------------------------------------------------
# Helper: phase-preserving complex binning
# ------------------------------------------------------------
def complex_bin_fft(fft_vals, freqs, num_bins):
    K = len(fft_vals)
    bin_factor = K // num_bins
    K_use = bin_factor * num_bins

    fft_trim = fft_vals[:K_use]
    freqs_trim = freqs[:K_use]

    fft_binned = fft_trim.reshape(num_bins, bin_factor).mean(axis=1)
    freqs_binned = freqs_trim.reshape(num_bins, bin_factor).mean(axis=1)

    delta_f_binned = bin_factor * (freqs[1] - freqs[0])
    return fft_binned, freqs_binned, delta_f_binned, bin_factor


# ------------------------------------------------------------
# Build receiver template (POWER → sqrt → VOLTAGE)
# ------------------------------------------------------------
def build_receiver_voltage_template_from_hr(
    folder_path,
    df,
    num_bins,
    output_h5,
    poly_deg=3,
    crop_seconds=2.0,
):
    power_complex = []
    power_real = []

    f_baseband_ref = None
    delta_t_ref = None
    delta_f_ref = None
    bin_factor_ref = None

    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.endswith(".dat"):
            continue

        match_key = file_name.replace("admx", "").replace(".dat", "")
        if not df["Filename_Tag"].astype(str).str.contains(match_key).any():
            continue

        with open(os.path.join(folder_path, file_name), "rb") as f:
            f.read(struct.unpack("q", f.read(8))[0])
            h1 = json.loads(f.read(struct.unpack("q", f.read(8))[0]))
            delta_t = float(h1["x_spacing"]) * 1e-6
            npts = struct.unpack("Q", f.read(8))[0]
            data = array.array("f")
            data.frombytes(f.read(npts * 4))

        ts = pycbc.types.TimeSeries(np.asarray(data, dtype=np.float64), delta_t)
        if crop_seconds:
            ts = ts.crop(crop_seconds, crop_seconds)

        fft_vals = pyfftw.interfaces.numpy_fft.rfft(ts.numpy())
        freqs = np.fft.rfftfreq(len(ts), delta_t)

        fft_binned, f_baseband, df_binned, bin_factor = complex_bin_fft(
            fft_vals, freqs, num_bins
        )

        power_complex.append(np.abs(fft_binned) ** 2)
        power_real.append(np.real(fft_binned) ** 2)

        if f_baseband_ref is None:
            f_baseband_ref = f_baseband
            delta_t_ref = delta_t
            delta_f_ref = df_binned
            bin_factor_ref = bin_factor

    power_complex = np.mean(power_complex, axis=0)
    power_real = np.mean(power_real, axis=0)

    xs = np.linspace(0, 1, len(power_complex))
    power_complex_fit = np.poly1d(np.polyfit(xs, power_complex, poly_deg))(xs)
    power_real_fit = np.poly1d(np.polyfit(xs, power_real, poly_deg))(xs)

    eps = 1e-30
    power_complex_fit = np.maximum(power_complex_fit, eps)
    power_real_fit = np.maximum(power_real_fit, eps)

    # VOLTAGE dividers
    voltage_complex = np.sqrt(power_complex_fit)
    voltage_real = np.sqrt(power_real_fit)

    with h5py.File(output_h5, "w") as f:
        f.create_dataset("f_baseband", data=f_baseband_ref)
        f.create_dataset("receiver_voltage_complex", data=voltage_complex)
        f.create_dataset("receiver_voltage_real", data=voltage_real)

        f.attrs["delta_t"] = delta_t_ref
        f.attrs["delta_f_binned"] = delta_f_ref
        f.attrs["bin_factor"] = bin_factor_ref
        f.attrs["notes"] = (
            "Receiver built in POWER space separately for |FFT|^2 and |Re(FFT)|^2; "
            "flattening uses sqrt(power)."
        )

    return f_baseband_ref, voltage_complex, voltage_real, delta_t_ref


# ------------------------------------------------------------
# Load receiver dividers
# ------------------------------------------------------------
def load_receiver_voltage_dividers(h5):
    with h5py.File(h5, "r") as f:
        return (
            f["f_baseband"][:],
            f["receiver_voltage_complex"][:],
            f["receiver_voltage_real"][:],
        )


# ------------------------------------------------------------
# Remove receiver (real + complex independently)
# ------------------------------------------------------------
def remove_receiver(fft, freqs, f_baseband, Gc, Gr):
    Gc_i = np.interp(freqs, f_baseband, Gc)
    Gr_i = np.interp(freqs, f_baseband, Gr)

    Gc_i = np.maximum(Gc_i, np.min(Gc_i[Gc_i > 0]))
    Gr_i = np.maximum(Gr_i, np.min(Gr_i[Gr_i > 0]))

    re = np.real(fft) / Gr_i
    im = np.imag(fft) / Gc_i

    return re + 1j * im, Gc_i


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def run_binned4(
    run_definition_path="run1b_definitions.yaml",
    nibble_name="nibble5",
    num_bins=2**15,
    crop_seconds=2.0,
    receiver_poly_deg=3,
    receiver_use_fit=True,
    HR_folder="hr_data",
    data_folder="2018_05_19",
    output_subfolder="binned_hr_data",
    do_binning=True,  # if False: no binning, still builds receiver + flattens (Re/Im separately)
):
    import os, struct, json, array
    import numpy as np
    import h5py, yaml, pyfftw, pycbc, pandas as pd
    from config_file_handling import get_intermediate_data_file_name

    # ----------------------------
    # helpers
    # ----------------------------
    def complex_bin_fft(fft_vals, freqs, num_bins_local):
        K = len(fft_vals)
        if num_bins_local > K:
            raise ValueError(f"num_bins={num_bins_local} > available FFT bins={K}. Increase N_time or lower num_bins.")

        bin_factor = K // num_bins_local
        K_use = bin_factor * num_bins_local

        fft_trim = fft_vals[:K_use]
        freqs_trim = freqs[:K_use]

        fft_binned = fft_trim.reshape(num_bins_local, bin_factor).mean(axis=1)
        freqs_binned = freqs_trim.reshape(num_bins_local, bin_factor).mean(axis=1)

        delta_f_native = freqs[1] - freqs[0]
        delta_f_binned = bin_factor * delta_f_native

        return fft_binned, freqs_binned, float(delta_f_binned), int(bin_factor)

    def _safe_positive(arr, eps=1e-30):
        return np.maximum(arr, eps)

    def build_receiver_templates_from_hr(folder_path, df, output_h5):
        """
        Builds TWO voltage dividers on the frequency grid used by the pipeline:

          - complex divider (for imag part): sqrt( < |FFT|^2 >_polyfit )
          - real divider    (for real part): sqrt( < (Re FFT)^2 >_polyfit )

        If do_binning=True  -> receiver built on the binned grid (num_bins)
        If do_binning=False -> receiver built on the native rFFT grid

        Writes receiver templates to output_h5 (this is NOT the per-file *_binned.h5 output,
        so downstream consumers of *_binned.h5 remain unaffected).
        """
        power_complex_list = []
        power_real_list = []

        f_ref = None
        dt_ref = None
        df_ref = None
        bin_factor_ref = None
        n_freq_ref = None

        for file_name in sorted(os.listdir(folder_path)):
            if not file_name.endswith(".dat"):
                continue
            file_path = os.path.join(folder_path, file_name)
            if not os.path.isfile(file_path):
                continue

            match_key = file_name.replace("admx", "").replace(".dat", "")
            if not df["Filename_Tag"].astype(str).str.contains(match_key).any():
                continue

            with open(file_path, "rb") as f:
                _ = struct.unpack("q", f.read(8))[0]
                f.read(_)  # header bytes
                _ = struct.unpack("q", f.read(8))[0]
                h1 = json.loads(f.read(_))
                delta_t = float(h1["x_spacing"]) * 1e-6

                npts = struct.unpack("Q", f.read(8))[0]
                data = array.array("f")
                data.frombytes(f.read(npts * 4))

            ts = pycbc.types.TimeSeries(np.asarray(data, dtype=np.float32), delta_t)

            if crop_seconds and crop_seconds > 0:
                try:
                    ts = ts.crop(crop_seconds, crop_seconds)
                except Exception:
                    pass

            x = np.asarray(ts, dtype=np.float64)

            fft_vals = pyfftw.interfaces.numpy_fft.rfft(x)
            freqs_native = np.fft.rfftfreq(len(x), d=delta_t)

            if do_binning:
                fft_used, freqs_used, df_used, bin_factor_used = complex_bin_fft(
                    fft_vals, freqs_native, int(num_bins)
                )
            else:
                fft_used = fft_vals
                freqs_used = freqs_native
                df_used = float(freqs_native[1] - freqs_native[0]) if len(freqs_native) > 1 else np.nan
                bin_factor_used = 1

            # build in POWER
            p_complex = np.abs(fft_used) ** 2            # |FFT|^2
            p_real    = (np.real(fft_used)) ** 2         # (Re FFT)^2

            if f_ref is None:
                f_ref = freqs_used.copy()
                dt_ref = float(delta_t)
                df_ref = float(df_used)
                bin_factor_ref = int(bin_factor_used)
                n_freq_ref = int(len(freqs_used))
            else:
                if len(freqs_used) != n_freq_ref:
                    raise ValueError("Receiver build: inconsistent frequency axis length across files.")
                if abs(delta_t - dt_ref) > 1e-12:
                    raise ValueError("Receiver build: delta_t differs across files; do not average across sample rates.")
                if not np.allclose(freqs_used, f_ref, rtol=0, atol=1e-9):
                    raise ValueError("Receiver build: frequency axis differs across files; ensure consistent N_time/crop.")
                if abs(df_used - df_ref) > 0:
                    # very strict; if you want loosen, change this to np.isclose
                    raise ValueError(f"Receiver build: delta_f differs across files ({df_used} vs {df_ref}).")

            power_complex_list.append(p_complex)
            power_real_list.append(p_real)

        if len(power_complex_list) == 0:
            raise RuntimeError("Receiver build: no usable .dat files matched df Filename_Tag.")

        pC = np.mean(np.vstack(power_complex_list), axis=0)
        pR = np.mean(np.vstack(power_real_list), axis=0)

        # polynomial smoothing in POWER space (same style as your old pipeline)
        xs = np.linspace(0.0, 1.0, len(pC))
        poly_deg = int(receiver_poly_deg)

        pC_fit = np.poly1d(np.polyfit(xs, pC, poly_deg))(xs)
        pR_fit = np.poly1d(np.polyfit(xs, pR, poly_deg))(xs)

        pC_fit = _safe_positive(pC_fit)
        pR_fit = _safe_positive(pR_fit)

        # VOLTAGE dividers are sqrt(POWER)
        Gc = np.sqrt(pC_fit)   # complex divider (use for imag)
        Gr = np.sqrt(pR_fit)   # real divider    (use for real)

        os.makedirs(os.path.dirname(output_h5), exist_ok=True)
        with h5py.File(output_h5, "w") as out:
            out.create_dataset("f_baseband", data=f_ref.astype(np.float64))
            out.create_dataset("receiver_voltage_complex", data=Gc.astype(np.float64))
            out.create_dataset("receiver_voltage_real", data=Gr.astype(np.float64))

            out.attrs["num_bins"] = int(num_bins) if do_binning else int(len(f_ref))
            out.attrs["poly_deg"] = int(poly_deg)
            out.attrs["delta_t"] = float(dt_ref)
            out.attrs["delta_f_binned"] = float(df_ref)
            out.attrs["bin_factor"] = int(bin_factor_ref)
            out.attrs["notes"] = (
                "Receiver built in POWER space on the pipeline frequency grid. "
                "Two dividers: sqrt(<|FFT|^2>) for imag, sqrt(<(Re FFT)^2>) for real."
            )

        return f_ref, Gc, Gr, dt_ref, df_ref

    def load_receiver_dividers(receiver_h5_path):
        with h5py.File(receiver_h5_path, "r") as f:
            f_base = f["f_baseband"][:]
            keys = set(f.keys())
            if "receiver_voltage_complex" not in keys or "receiver_voltage_real" not in keys:
                raise KeyError(f"Receiver template missing required datasets. Keys: {sorted(keys)}")
            Gc = f["receiver_voltage_complex"][:]
            Gr = f["receiver_voltage_real"][:]
        return f_base, _safe_positive(Gc), _safe_positive(Gr)

    def flatten_fft_separate_re_im(fft_vals_used, freqs_used, f_base, Gc_base, Gr_base):
        # interpolate onto this file's freq grid
        Gc_i = np.interp(freqs_used, f_base, Gc_base)
        Gr_i = np.interp(freqs_used, f_base, Gr_base)

        # guard against zeros
        Gc_i = _safe_positive(Gc_i)
        Gr_i = _safe_positive(Gr_i)

        # flatten with separate dividers
        re_flat = np.real(fft_vals_used) / Gr_i
        im_flat = np.imag(fft_vals_used) / Gc_i
        fft_flat = re_flat + 1j * im_flat

        # keep output H5 format: dataset "Receiver Shape" stays a single array.
        # store the COMPLEX divider used (Gc_i) like before (single array).
        return fft_flat, Gc_i

    # ----------------------------
    # load run definition / parameters
    # ----------------------------
    with open(run_definition_path, "r") as f:
        run_definition = yaml.load(f, Loader=yaml.Loader)

    parameter_file = get_intermediate_data_file_name(
        run_definition["nibbles"][nibble_name],
        "2018_05_19.txt"
    )

    useful_parameters = [
        "Start_Frequency", "Stop_Frequency", "Digitizer_Log_ID", "Integration_Time",
        "Filename_Tag", "Quality_Factor", "Cavity_Resonant_Frequency", "JPA_SNR",
        "Thfet", "Attenuation", "Reflection", "Transmission?"
    ]
    df = pd.read_csv(parameter_file, delimiter="\t", names=useful_parameters, header=None)

    if "Filename_Tag" not in df.columns:
        rename_map = {i: useful_parameters[i] for i in range(min(len(df.columns), len(useful_parameters)))}
        df = df.rename(columns=rename_map)
    if "Filename_Tag" not in df.columns:
        raise KeyError(f"Filename_Tag not found in df.columns={list(df.columns)}")

    folder_path = os.path.join(HR_folder, data_folder)
    output_folder = os.path.join(HR_folder, output_subfolder)
    os.makedirs(output_folder, exist_ok=True)

    file_prefix = run_definition["nibbles"][nibble_name]["file_prefix"]
    receiver_h5 = os.path.join(HR_folder, f"{file_prefix}_receiver_shape_hr_voltage.h5")

    print(f"do_binning = {bool(do_binning)}")
    if do_binning:
        print(f"num_bins   = {int(num_bins)}")
    else:
        print("num_bins   = (unused; using native rFFT grid)")

    # ----------------------------
    # build + load receiver dividers on the correct grid
    # ----------------------------
    print("Building receiver templates (POWER→sqrt) with separate Re/Im dividers ...")
    f_baseband_ref, _, _, delta_t_ref, _ = build_receiver_templates_from_hr(
        folder_path=folder_path,
        df=df,
        output_h5=receiver_h5,
    )
    print(f"Saved receiver template: {receiver_h5}")

    f_baseband, Gc_base, Gr_base = load_receiver_dividers(receiver_h5)
    print("Loaded receiver dividers")

    outputs = []

    # ----------------------------
    # process each file
    # ----------------------------
    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.endswith(".dat"):
            continue

        file_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(file_path):
            continue

        match_key = file_name.replace("admx", "").replace(".dat", "")
        if not df["Filename_Tag"].astype(str).str.contains(match_key).any():
            continue

        fstart_mhz = float(df.loc[df["Filename_Tag"].astype(str).str.contains(match_key), "Start_Frequency"].iloc[0])
        fstop_mhz  = float(df.loc[df["Filename_Tag"].astype(str).str.contains(match_key), "Stop_Frequency"].iloc[0])
        fstart_abs_hz = fstart_mhz * 1e6
        fstop_abs_hz  = fstop_mhz * 1e6

        match_name = file_name.replace(".dat", "")
        output_file_name = os.path.join(output_folder, f"{match_name}_binned.h5")

        with open(file_path, "rb") as f:
            header_size = struct.unpack("q", f.read(8))[0]
            f.read(header_size)
            h1_size = struct.unpack("q", f.read(8))[0]
            h1_json = json.loads(f.read(h1_size))
            delta_t = float(h1_json["x_spacing"]) * 1e-6

            npts = struct.unpack("Q", f.read(8))[0]
            data = array.array("f")
            data.frombytes(f.read(npts * 4))

        ts = pycbc.types.TimeSeries(np.asarray(data, dtype=np.float32), delta_t)

        if crop_seconds and crop_seconds > 0:
            try:
                ts = ts.crop(crop_seconds, crop_seconds)
            except Exception:
                pass

        if abs(delta_t - float(delta_t_ref)) > 1e-12:
            raise ValueError(
                f"delta_t mismatch for {file_name}: got {delta_t}, receiver template used {delta_t_ref}. "
                "Either rebuild receiver per-sample-rate or restrict files."
            )

        x = np.asarray(ts, dtype=np.float64)

        fft_native = pyfftw.interfaces.numpy_fft.rfft(x)
        freqs_native = np.fft.rfftfreq(len(x), d=delta_t)
        df_native = float(freqs_native[1] - freqs_native[0]) if len(freqs_native) > 1 else np.nan

        if do_binning:
            fft_used, freqs_used, df_used, bin_factor_used = complex_bin_fft(
                fft_native, freqs_native, int(num_bins)
            )
            num_bins_to_save = int(num_bins)
        else:
            fft_used = fft_native
            freqs_used = freqs_native
            df_used = df_native
            bin_factor_used = 1
            num_bins_to_save = int(len(freqs_used))

        # flatten using separate dividers
        fft_flat, G_used_to_store = flatten_fft_separate_re_im(
            fft_vals_used=fft_used,
            freqs_used=freqs_used,
            f_base=f_baseband,
            Gc_base=Gc_base,
            Gr_base=Gr_base,
        )

        freqs_abs = freqs_used + fstart_abs_hz

        # ----------------------------
        # DO NOT CHANGE OUTPUT H5 FORMAT
        # ----------------------------
        with h5py.File(output_file_name, "w") as out:
            out.create_dataset("FFT_Binned_Re", data=np.real(fft_used).astype(np.float32))
            out.create_dataset("FFT_Binned_Im", data=np.imag(fft_used).astype(np.float32))
            out.create_dataset("FFT_BinnedFlat_Re", data=np.real(fft_flat).astype(np.float32))
            out.create_dataset("FFT_BinnedFlat_Im", data=np.imag(fft_flat).astype(np.float32))
            out.create_dataset("Receiver Shape", data=G_used_to_store.astype(np.float32))
            out.create_dataset("f_baseband", data=freqs_used.astype(np.float64))
            out.create_dataset("f_abs", data=freqs_abs.astype(np.float64))

            out.attrs["delta_t"] = float(delta_t)
            out.attrs["delta_f_binned"] = float(df_used)
            out.attrs["num_bins"] = int(num_bins_to_save)
            out.attrs["bin_factor"] = int(bin_factor_used)
            out.attrs["N_time"] = int(len(x))
            out.attrs["fstart_abs_hz"] = float(fstart_abs_hz)
            out.attrs["fstop_abs_hz"] = float(fstop_abs_hz)

            out.attrs["notes"] = (
                "Saved complex rFFT (optionally binned) plus receiver-flattened version. "
                "Flattening divides Re and Im by separate voltage dividers derived from "
                "sqrt(<(Re FFT)^2>) and sqrt(<|FFT|^2>), respectively. "
                "Receiver Shape dataset stores the complex (imag) voltage divider interpolated to this grid."
            )

            # preserve your run_params group/attrs exactly
            dat_name = match_name
            if dat_name.startswith("admx"):
                dat_name = dat_name[len("admx"):]
            dat_name = dat_name + ".dat"
            row = df.loc[df["Filename_Tag"].astype(str).str.contains(dat_name)].iloc[0]

            run_params = {
                "Start_Frequency": float(row["Start_Frequency"]),
                "Stop_Frequency": float(row["Stop_Frequency"]),
                "Digitizer_Log_ID": str(row["Digitizer_Log_ID"]),
                "Integration_Time": float(row["Integration_Time"]),
                "Filename_Tag": str(row["Filename_Tag"]),
                "Quality_Factor": float(row["Quality_Factor"]),
                "Cavity_Resonant_Frequency": float(row["Cavity_Resonant_Frequency"]),
                "JPA_SNR": float(row["JPA_SNR"]),
                "Thfet": float(row["Thfet"]),
                "Attenuation": float(row["Attenuation"]),
                "Reflection": float(row["Reflection"]) if not pd.isna(row["Reflection"]) else np.nan,
                "Transmission": float(row["Transmission?"]) if not pd.isna(row["Transmission?"]) else np.nan,
            }

            g = out.require_group("run_params")
            for k, v in run_params.items():
                if isinstance(v, str):
                    g.attrs[k] = np.bytes_(v)
                else:
                    g.attrs[k] = v

        outputs.append(output_file_name)

    print("Done. All matched files processed.")
    return outputs, receiver_h5, f_baseband, Gc_base
# ------------------------------------------------------------
# TERMINAL EXECUTABLE ENTRYPOINT
# ------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-r", "--run_definition", default="run1b_definitions.yaml")
    p.add_argument("-n", "--nibble_name", default="nibble5")
    p.add_argument("--num_bins", type=int, default=2**15)
    p.add_argument("--crop_seconds", type=float, default=2.0)
    p.add_argument("--receiver_poly_deg", type=int, default=3)
    p.add_argument("--receiver_use_fit", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    run_binned4(
        run_definition_path=args.run_definition,
        nibble_name=args.nibble_name,
        num_bins=args.num_bins,
        crop_seconds=args.crop_seconds,
        receiver_poly_deg=args.receiver_poly_deg,
        receiver_use_fit=args.receiver_use_fit,
    )


if __name__ == "__main__":
    main()

    # Run this line from terminal to run program
    # python binned4.py -r run1b_definitions.yaml -n nibble5 --num_bins 32768 --crop_seconds 2.0 --receiver_poly_deg 3 --receiver_use_fit