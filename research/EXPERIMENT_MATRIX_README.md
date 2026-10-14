# Component Falsification Study

The Astra prompt requires a controlled contribution study over the major TG-Detect modifications:

- F = improved/engineered features
- G = graph construction
- T = temporal encoding
- M = persistent memory
- R = GRU/recurrent update
- L = loss change
- N = normalization change
- H = rehearsal/replay

Required first-pass matrix:
- baseline B
- all 8 single-factor variants
- all 28 pairwise variants
- full stack
- all 8 full-stack single-removal variants
- negative/falsification controls

All heavy training runs are to execute on Modal after local smoke tests and environment verification. Every run records seed, code/config hash, dataset hash, GPU type, resource profile, metrics, and artifact paths.
