"""
waveform_template.py
Notebook-callable:
  from waveform_template import build_voltage_template_on_binned_grid
Terminal executable:
  python waveform_template.py --help

This script:
  - reads f_baseband, f_abs, delta_f_binned from a binned ADMX H5
  - builds a PyCBC FD waveform on native df
  - interpolates (amp + unwrapped phase) onto f_baseband
  - loads Q + cavity params from /run_params attrs in the binned H5
  - loads B from run_definition YAML
  - converts strain -> voltage
  - optionally applies cavity Lorentzian transfer
  - optionally plots and/or saves outputs to an H5
"""

import argparse
import h5py
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from pycbc.waveform import get_fd_waveform


# ----------------------------
# Helpers
# ----------------------------
def _as_py(v):
    if isinstance(v, (np.bytes_, bytes)):
        return v.decode("utf-8")
    if isinstance(v, np.generic):
        return v.item()
    return v


def interp_complex_amp_phase(wf_fd, f_native, f_target, amp_floor=0.0, fill_value=0.0):
    y = wf_fd.numpy() if hasattr(wf_fd, "numpy") else np.asarray(wf_fd)
    y = np.asarray(y, dtype=np.complex128)

    f_native = np.asarray(f_native, dtype=np.float64)
    f_target = np.asarray(f_target, dtype=np.float64)

    if y.ndim != 1 or f_native.ndim != 1:
        raise ValueError("wf_fd and f_native must be 1D")
    if len(y) != len(f_native):
        raise ValueError("len(wf_fd) must match len(f_native)")

    s = np.argsort(f_native)
    f = f_native[s]
    y = y[s]

    amp = np.abs(y)
    phase = np.unwrap(np.angle(y))

    if amp_floor is None:
        mask = np.isfinite(amp) & np.isfinite(phase)
    else:
        mask = (amp > amp_floor) & np.isfinite(amp) & np.isfinite(phase)

    if np.count_nonzero(mask) < 4:
        re = np.interp(f_target, f, y.real, left=0.0, right=0.0)
        im = np.interp(f_target, f, y.imag, left=0.0, right=0.0)
        return re + 1j * im

    f_m = f[mask]
    amp_m = amp[mask]
    ph_m = phase[mask]

    amp_itp = PchipInterpolator(f_m, amp_m, extrapolate=False)
    A = amp_itp(f_target)
    PH = np.interp(f_target, f_m, ph_m, left=np.nan, right=np.nan)

    out = np.full_like(f_target, fill_value, dtype=np.complex128)
    ok = np.isfinite(A) & np.isfinite(PH)
    out[ok] = A[ok] * np.exp(1j * PH[ok])
    return out


def load_run_params_for_lorentzian_from_binned_h5(binned_h5):
    needed = ["Start_Frequency", "Stop_Frequency", "Cavity_Resonant_Frequency", "Quality_Factor"]
    rp = {}

    with h5py.File(binned_h5, "r") as f:
        # Preferred: /run_params group attrs
        if "run_params" in f and isinstance(f["run_params"], h5py.Group):
            g = f["run_params"]
            for k in needed:
                if k in g.attrs:
                    rp[k] = _as_py(g.attrs[k])

        # Fallback: root attrs "run_params/<key>"
        for k in needed:
            if k not in rp:
                key2 = f"run_params/{k}"
                if key2 in f.attrs:
                    rp[k] = _as_py(f.attrs[key2])

        # Fallback: absolute start/stop in Hz
        if "Start_Frequency" not in rp and "fstart_abs_hz" in f.attrs:
            rp["Start_Frequency"] = float(_as_py(f.attrs["fstart_abs_hz"])) / 1e6
        if "Stop_Frequency" not in rp and "fstop_abs_hz" in f.attrs:
            rp["Stop_Frequency"] = float(_as_py(f.attrs["fstop_abs_hz"])) / 1e6

    # Cast numerics
    for k in needed:
        if k in rp:
            rp[k] = float(rp[k])

    missing = [k for k in needed if k not in rp]
    if missing:
        raise KeyError(
            f"Missing {missing} in {binned_h5}. Expected in group '/run_params' as attrs."
        )
    return rp


def load_B_from_run_definition(run_definition_path, target_nibble):
    with open(run_definition_path, "r") as f:
        run_definition = yaml.load(f, Loader=yaml.Loader)
    return float(run_definition["nibbles"][target_nibble]["Bfield"])


def cavity_lorentzian_transfer_on_baseband(f_baseband_hz, fstart_mhz, f0_mhz, Q, complex_response=True):
    fstart_abs_hz = float(fstart_mhz) * 1e6
    f0_abs_hz = float(f0_mhz) * 1e6

    f0_baseband_hz = f0_abs_hz - fstart_abs_hz

    x = (f_baseband_hz - f0_baseband_hz) / f0_abs_hz
    H_complex = 1.0 / (1.0 + 2j * float(Q) * x)

    if complex_response:
        return H_complex, f0_baseband_hz
    return np.abs(H_complex), f0_baseband_hz


def strain_fd_to_voltage_fd(H_strain_fd, f_abs_hz, Q, B_tesla, V_liters=136.0, eta=0.1):
    mu0 = 4 * np.pi * 1e-7      # vacuum permeability [N/A^2 = kg m / (A^2 s^2)]
    c = 299_792_458.0           # speed of light [m/s]
    V_liters *=  1e-3   # liters -> m^3
    f_abs_hz = np.asarray(f_abs_hz, dtype=np.float64)
    H = np.asarray(H_strain_fd, dtype=np.complex128)
    if H.shape != f_abs_hz.shape:
        raise ValueError("H_strain_fd must have same shape as f_abs_hz")

    omega = f_abs_hz
    omega_safe = omega.copy()
    if len(omega_safe) > 1:
        omega_safe[0] = omega_safe[1]
    else:
        omega_safe[0] = 1.0
    # Comparing to Asher Berlin's equation, we need to multiply by 1/(mu c^2) to recover a meaningful expression for power
    scale = np.sqrt(1/(mu0*c**2)*0.5 * float(Q) * (omega_safe ** 3) * (float(V_liters) ** (5.0 / 3.0)))
    scale *= float(eta) * float(B_tesla)

    V_fd = scale * H
    V_fd[0] = 0.0 + 0.0j
    return V_fd, scale


# ----------------------------
# Main notebook-callable function
# ----------------------------
def build_voltage_template_on_binned_grid(
    binned_h5,
    run_definition_path,
    target_nibble,
    approximant="TaylorF2",
    mass1=1.4,
    mass2=1.4,
    V_liters=136.0,
    eta=0.1,
    amp_floor=0.0,
    apply_cavity=True,
    complex_cavity=True,
):
    with h5py.File(binned_h5, "r") as f:
        f_baseband = f["f_baseband"][:].astype(np.float64)
        f_abs = f["f_abs"][:].astype(np.float64)

        if "delta_f_binned" in f.attrs:
            df_target = float(_as_py(f.attrs["delta_f_binned"]))
        elif "delta_f" in f.attrs:
            df_target = float(_as_py(f.attrs["delta_f"]))
        else:
            raise KeyError("Missing delta_f_binned (or delta_f) in binned H5 attrs.")

    # build strain waveform at native df
    f_lower = max(1.0, float(f_baseband[1]))
    hp, hc = get_fd_waveform(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        delta_f=df_target,
        f_lower=f_lower,
    )
    f_hp = np.arange(len(hp), dtype=np.float64) * float(hp.delta_f)

    # interpolate strain onto saved grid
    H_baseband = interp_complex_amp_phase(hp, f_hp, f_baseband, amp_floor=amp_floor, fill_value=0.0)

    # load B and per-scan run_params (includes Q, start/stop/f0)
    B = load_B_from_run_definition(run_definition_path, target_nibble)
    rp = load_run_params_for_lorentzian_from_binned_h5(binned_h5)
    Q = float(rp["Quality_Factor"])

    # strain -> voltage scaling (use abs RF frequency)
    V_fd_scaled, scale = strain_fd_to_voltage_fd(
        H_strain_fd=H_baseband,
        f_abs_hz=f_abs,
        Q=Q,
        B_tesla=B,
        V_liters=V_liters,
        eta=eta,
    )

    H_cav = None
    f0_baseband_hz = None
    V_fd_final = V_fd_scaled

    if apply_cavity:
        H_cav, f0_baseband_hz = cavity_lorentzian_transfer_on_baseband(
            f_baseband_hz=f_baseband,
            fstart_mhz=rp["Start_Frequency"],
            f0_mhz=rp["Cavity_Resonant_Frequency"],
            Q=Q,
            complex_response=complex_cavity,
        )
        V_fd_final = V_fd_scaled * H_cav

    return {
        "f_baseband_hz": f_baseband,
        "f_abs_hz": f_abs,
        "df_target": df_target,
        "hp_native": hp,
        "f_hp_native": f_hp,
        "H_strain_baseband": np.asarray(H_baseband, dtype=np.complex128),
        "V_fd_scaled": np.asarray(V_fd_scaled, dtype=np.complex128),
        "V_fd_final": np.asarray(V_fd_final, dtype=np.complex128),
        "scale": np.asarray(scale, dtype=np.float64),
        "H_cav": None if H_cav is None else np.asarray(H_cav),
        "f0_baseband_hz": f0_baseband_hz,
        "Q": Q,
        "B": B,
        "run_params_for_lorentzian": rp,
    }


# ----------------------------
# plotting + saving helpers (optional)
# ----------------------------
def plot_voltage_template(out, xscale_log=True):
    fbb = out["f_baseband_hz"]
    Vscaled = out["V_fd_scaled"]
    Vfinal = out["V_fd_final"]
    f0bb = out["f0_baseband_hz"]

    plt.figure(figsize=(10, 5))
    plt.plot(fbb[1:], np.abs(Vscaled)[1:], label="scaled voltage (no cavity)")
    plt.plot(fbb[1:], np.abs(Vfinal)[1:], "--", label="after cavity")
    if f0bb is not None:
        plt.axvline(f0bb, linestyle=":", alpha=0.6, label="f0 (baseband)")
    plt.xlabel("Baseband frequency [Hz]")
    plt.ylabel("|V(f)| [arb volts]")
    plt.title("Voltage template on saved binned frequency grid")
    plt.grid(True, alpha=0.3)
    if xscale_log:
        plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_voltage_template_h5(out_path, out_dict):
    with h5py.File(out_path, "w") as f:
        f.create_dataset("f_baseband_hz", data=out_dict["f_baseband_hz"].astype(np.float64))
        f.create_dataset("f_abs_hz", data=out_dict["f_abs_hz"].astype(np.float64))
        f.create_dataset("H_strain_baseband", data=out_dict["H_strain_baseband"].astype(np.complex128))
        f.create_dataset("V_fd_scaled", data=out_dict["V_fd_scaled"].astype(np.complex128))
        f.create_dataset("V_fd_final", data=out_dict["V_fd_final"].astype(np.complex128))
        f.create_dataset("scale", data=out_dict["scale"].astype(np.float64))

        if out_dict["H_cav"] is not None:
            f.create_dataset("H_cav", data=out_dict["H_cav"].astype(np.complex128))

        for k in ("df_target", "Q", "B"):
            f.attrs[k] = float(out_dict[k])
        if out_dict["f0_baseband_hz"] is not None:
            f.attrs["f0_baseband_hz"] = float(out_dict["f0_baseband_hz"])

        rp = out_dict.get("run_params_for_lorentzian", {})
        g = f.require_group("run_params")
        for k, v in rp.items():
            g.attrs[k] = float(v)


# ----------------------------
# terminal entrypoint
# ----------------------------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--binned_h5", required=True)
    p.add_argument("--run_definition", required=True)
    p.add_argument("--target_nibble", required=True)

    p.add_argument("--approximant", default="TaylorF2")
    p.add_argument("--mass1", type=float, default=1.4)
    p.add_argument("--mass2", type=float, default=1.4)
    p.add_argument("--V_liters", type=float, default=136.0)
    p.add_argument("--eta", type=float, default=0.1)

    p.add_argument("--amp_floor", type=float, default=0.0)
    p.add_argument("--no_cavity", action="store_true")
    p.add_argument("--mag_cavity", action="store_true", help="apply |H_cav| instead of complex H_cav")

    p.add_argument("--plot", action="store_true")
    p.add_argument("--linear_x", action="store_true")
    p.add_argument("--save_h5", default="", help="path to save outputs H5 (optional)")
    return p.parse_args()


def main():
    args = _parse_args()

    out = build_voltage_template_on_binned_grid(
        binned_h5=args.binned_h5,
        run_definition_path=args.run_definition,
        target_nibble=args.target_nibble,
        approximant=args.approximant,
        mass1=args.mass1,
        mass2=args.mass2,
        V_liters=args.V_liters,
        eta=args.eta,
        amp_floor=args.amp_floor,
        apply_cavity=not bool(args.no_cavity),
        complex_cavity=not bool(args.mag_cavity),
    )

    print("Q =", out["Q"], "B =", out["B"])
    rp = out["run_params_for_lorentzian"]
    print("Start/Stop/f0 (MHz) =", rp["Start_Frequency"], rp["Stop_Frequency"], rp["Cavity_Resonant_Frequency"])
    if out["f0_baseband_hz"] is not None:
        print("f0 baseband (Hz) =", out["f0_baseband_hz"])

    if args.plot:
        plot_voltage_template(out, xscale_log=not bool(args.linear_x))

    if args.save_h5:
        save_voltage_template_h5(args.save_h5, out)
        print("saved:", args.save_h5)


if __name__ == "__main__":
    main()

# Run from terminal with line below:
#python waveform_template.py \
#   --binned_h5 ./hr_data/binned_hr_data/admx_data_2018_05_19_23_20_24_channel_1_binned.h5 \
#   --run_definition ./run1b_definitions.yaml \
#   --target_nibble nibble5 \
#   --plot \
#   --save_h5 voltage_template_out.h5