# Ball-fault detection: a negative-result write-up

> **Status:** intentional negative result; preserved as a research artifact.
> **Last run:** 2026-05-28, on 30 ball-fault windows across CWRU files 118.mat (0.007"), 185.mat (0.014"), 222.mat (0.021").
> **Reproduction:** `python scripts/ball_detection_experiment.py`

## What we tried

Roadmap §3 asked: can we lift ball-fault detection above the 0/10 baseline reported in `eval/error_analysis.md` for the deployed diagnostic?

Four envelope-family extensions were tested on the full 30-window ball pool (3 defect diameters × 10 windows). All four use the same downstream scoring shape — peaks at fault frequency + FTF sidebands, normalised by background median — they differ only in the band selection and the envelope flavour.

| Method | Band | Envelope | Harmonics × sidebands | Ball detection |
|---|---|---|---|---|
| A. Baseline | 2-4.5 kHz fixed (`diagnose v2`) | Hilbert envelope | 2 × 2 | **0/30** |
| B. More family | 2-4.5 kHz fixed | Hilbert envelope | 3 × 3 | **0/30** |
| C. SES at fixed band | 2-4.5 kHz fixed | Squared envelope (SES) | 2 × 2 | **0/30** |
| D. SK-selected band + SES | sweep 8 bands in 500-5500 Hz, max envelope-kurtosis | Squared envelope | 2 × 2 | **1/30** |

Method D's lone hit is on a window of the 0.007" file and is within noise of zero given the small sample.

## Why none of them work on this data

Two intertwined reasons:

**1. Bearing geometry alignment.** SKF 6205-2RS gives BPFI ≈ 5.4152 × fr and 2·BSF ≈ 9.4270 × fr. At 1797 RPM these land at 162 Hz and 282 Hz. The 2·BSF harmonic family is structurally smaller than the BPFI/BPFO harmonic family because BSF coefficients are not integers — its harmonics fall between race-related peaks rather than aligning with them. On envelope spectra of real CWRU ball faults, the race-related lines we are *not* looking for typically have higher amplitude than the BSF-family lines we are.

**2. Ball impacts are slip-sensitive and not strictly periodic.** Unlike a race-related defect (which produces a clean periodic impulse train as the rotating element passes a stationary defect), a ball-defect impact occurs *only* when the defective ball is in the load zone — and the ball-spin frequency is sensitive to slip. The envelope spectrum's strength is detecting periodic impulse trains; that strength becomes a weakness when the underlying mechanism is approximately periodic with slip-induced jitter. Smith & Randall (2015) explicitly flag several CWRU ball records as not diagnosable with established methods; the Polito 2021 envelope-demodulation study tags the B021_0 ball case as having non-periodic impulses.

This is *not* a bug. It is the known operating regime of envelope analysis. The deployed `diagnose v2` is correct on the two race classes (10/10 each); ball is the corner where the method's assumptions stop holding.

## What would plausibly work

In order of expected effort × payoff:

1. **AR-residual / cepstral pre-whitening** before SES. Removes the strong race-related "discrete tones" that mask BSF energy. Codex's literature search flagged this as the next experiment worth running. ~1 day.
2. **Cyclic Spectral Coherence** (Antoni 2007). The 2nd-order cyclostationary signature of a ball fault is more robust than the envelope. ~3-5 days; needs a careful implementation to handle the cyclic frequency grid.
3. **Supervised classifier** on top of envelope features. Probably the most pragmatic path if labelled data is available. Out of MVP scope.

## Why this write-up is in the repo

A portfolio that ships only the experiments that worked tells a hiring manager "I can pick winning experiments." A portfolio that ships an honest negative result, with a reproducible script and the literature citations that explain why it failed, tells them something more useful: *this person knows the boundary of their method, and reports it the same way they'd report a positive finding.*

The `order_tracking.py` module remains in the repo because the operators (angle-domain resampling, envelope cepstrum, FTF periodicity score) are textbook-correct implementations and tested at 7/7. They are the right shape to feed into the methods listed above; we just didn't get to them inside this MVP.

## References

- Smith, W.A. & Randall, R.B. (2015) — *Rolling element bearing diagnostics using the Case Western Reserve University data: a benchmark study.* Mechanical Systems and Signal Processing 64-65.
  https://www.sciencedirect.com/science/article/pii/S0888327015002034
- Polito 2021 — *Bearing fault diagnosis with envelope analysis (CWRU dataset).*
  https://iris.polito.it/retrieve/e384c433-b515-d4b2-e053-9f05fe0a1d67/applsci-11-06262-v2.pdf
- IEEE Access 2023 — *Bearing fault diagnosis with envelope analysis and machine-learning approaches using the CWRU dataset.*
  https://digibuo.uniovi.es/dspace/bitstream/handle/10651/69892/Bearing_Fault_Diagnosis_With_Envelope_Analysis_and_Machine_Learning_Approaches_Using_CWRU_Dataset.pdf
- Antoni, J. (2007) — *Cyclic spectral analysis of rolling-element bearing signals.* Journal of Sound and Vibration 304.
