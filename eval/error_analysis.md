# Error Analysis — pdm-agent diagnostic v2

Method: diagnose  /  Overall accuracy: 76.7%

## Known failure mode: 0.007-inch ball-fault under-detection

CWRU's 0.007-inch ball-defect class is intrinsically the hardest signature on the drive-end bearing — defect impulses smear across FTF-modulated sidebands and the spectral peak energy at 2×BSF is often lower than incidental peaks at BPFI/BPFO. In our 43-case eval the diagnostic mis-classified 7 ball-fault windows as 'normal'. This is documented in maintenance literature and is NOT a threshold-tuning issue — it reflects the underlying SNR of small ball defects in this rig. Production-grade ball detection requires either order tracking, cepstrum analysis, or supervised models with more labelled samples — out of scope for this MVP.

## Other mistakes
- `cwru-118-w004`: actual=**ball** predicted=**outer_race** (severity=critical, confidence=1.00, kurtosis=0.02)
- `cwru-118-w008`: actual=**ball** predicted=**outer_race** (severity=critical, confidence=1.00, kurtosis=0.06)
- `cwru-118-w009`: actual=**ball** predicted=**outer_race** (severity=critical, confidence=1.00, kurtosis=0.04)

## Honest scope

This evaluation is on CWRU drive-end bearing data — an analog benchmark for BESS auxiliary equipment (cooling-pump / fan) bearings. It does NOT validate BESS PdM in production. See repo README §Scope.