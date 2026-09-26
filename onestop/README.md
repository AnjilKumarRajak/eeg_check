# OneStop external check (gaze)

Code for the external check on OneStop Eye Movements (Berzak et al., 2025, Scientific Data;
data at https://osf.io/2prdq/, CC BY 4.0): the appendix "External Check on OneStop" of the paper.

The measurement uses the **same framework code and the same gaze instrument** as ZuCo
(`../cprd`, `../experiments`, GPT-2 Large prior); only the data adapter is new.

| file | what it does |
|---|---|
| `e0_build_onestop.py` | OneStop interest-area report (ordinary-reading regime) -> the HDF5 schema of `cprd/data.py`; drops practice and repeated-reading trials, treats the original (Adv) and simplified (Ele) version of a paragraph as separate texts, maps the report columns to the 6-d ZuCo gaze vector, splits by text |
| `run_onestop.sh` | the full pipeline: build, gaze bound (E2), word-level view (E13), selection (E3), structure-only control |
| `smoke_test.sh` | quick end-to-end check on a synthetic interest-area report (no data download needed) |

Column mapping (same six quantities as ZuCo):

| gaze quantity | OneStop column |
|---|---|
| fixated | `IA_FIXATION_COUNT > 0` |
| log(1 + fixation count) | `IA_FIXATION_COUNT` |
| log(1 + first-fixation duration) | `IA_FIRST_FIXATION_DURATION` |
| log(1 + gaze duration) | `IA_FIRST_RUN_DWELL_TIME` |
| log(1 + total reading time) | `IA_DWELL_TIME` |
| log(1 + go-past time) | `IA_REGRESSION_PATH_DURATION` |

## Running

```bash
pip install -r requirements.txt
bash onestop/smoke_test.sh                        # quick check
bash run_v2.sh                                    # (once) ZuCo E1 calibration -> runs_v2/gate_estimator.json
bash onestop/run_onestop.sh <IA_REPORT_CSV> <WORK>
```

The main gaze row uses 1,000 permutation draws; the structure-only control uses 300.
