# Follow-up validation

13 tests pass in the local CPU environment. New checks verify a disjoint benign
reference, zero synthetic event-type rarity, bounded features, preserved marginals
under field shuffling, uncapped score scaling, identical update counts across all
six arms, and notebook code compilation. The full original tests also pass.

All six arms completed a real KDD smoke run with 128 preparation-training records,
32 reference records, 32 validation, 128 calibration, 128 test, one seed and one
epoch. This validates execution only; it is too small to interpret gains. The
default 60-run GPU study has not been executed here. There is no local CUDA GPU.

New split seeds change the sample, but may overlap the first study's records.
They measure same-dataset sensitivity, not independent replication. Production
count lookup, ingestion, drift, and full-model transfer remain separate tests.
