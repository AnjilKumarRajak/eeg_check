# COFETT external check

Code for the external check on COFETT (OpenNeuro ds006317, CC0): Section 4.5 and the appendix
"External Check on COFETT" of the paper.

The measurement uses the **same framework code** as ZuCo (`../cprd`, `../experiments`); only two
things change: the data adapter (this folder) and the frozen prior (Qwen2.5-0.5B, float32, because
COFETT is Chinese). Estimator, nulls, gates, encoder and training settings are identical.

| file | what it does |
|---|---|
| `download_cofett.sh` | downloads the 40 runs used (session 1: both participants, para1+para2; sessions 2-4: para2) |
| `extract_features.py` | EDF -> per-character 840-d features: 50 Hz notch, 250 Hz resampling, 105 channels, 8 ZuCo bands, squared Hilbert envelope averaged over each character's 0.4 s window, log |
| `e0_build_cofett.py` | features -> the HDF5 schema of `cprd/data.py`; builds every split (unseen sentences, session-1 repeated, 16-repetition scheme; pooled or one participant; injected controls) |
| `make_fp32_prior.py` | writes the float32 copy of Qwen2.5-0.5B used as the frozen prior |
| `run_cofett.sh` | the full pipeline: features, datasets, all COFETT measurements |
| `smoke_test.sh` | quick end-to-end check on synthetic features (no recordings needed) |
| `stim/` | the released sentence lists used to match trials to sentences (not part of the OpenNeuro dataset) |

## Running

```bash
pip install -r requirements.txt            # adds mne and openpyxl for the COFETT adapter
bash cofett/smoke_test.sh                  # quick check
bash cofett/download_cofett.sh <RAW>       # 40 runs from OpenNeuro ds006317
bash run_v2.sh                             # (once) ZuCo E1 calibration -> runs_v2/gate_estimator.json
bash cofett/run_cofett.sh <RAW> <WORK>     # everything else; idempotent, resumable
```

`run_cofett.sh` writes one run folder per condition to `<WORK>`:

| paper item | run folder |
|---|---|
| unseen sentences, pooled / S1 / S2 | `runs_{reading,recall}`, `runs_{reading,recall}_sub01`, `..._sub02` |
| COFETT scheme (16 repetitions), pooled / S1 / S2 | `runs_repeat16_{reading,recall}`, `..._sub01`, `..._sub02` |
| session-1 repeated split | `runs_repeat_{reading,recall}` |
| injected text-dependent controls | `runs_spike_a{0.1,0.3,1.0}` |
| evidence window / tilt capacity | `runs_reading_w{0,3}`, `runs_reading_{lineartilt,r4,r8,r32,bigtilt}` |
| selection, one-shot test, LOSO, scaling, evaluation gain | `runs_reading` and `runs_recall` (`gate_selection`, `gate_system`, `gate_loso`, `gate_scaling`, `gate_sensitivity`) |

Main instrument rows use 1,000 permutation draws; controls and robustness rows use 300.
