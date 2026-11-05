# Data card: NASA C-MAPSS turbofan engine degradation

**Source.** Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) run-to-failure data,
published by the NASA Prognostics Center of Excellence. Reference: A. Saxena, K. Goebel,
D. Simon, N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure
simulation", International Conference on Prognostics and Health Management (PHM08), 2008.

**How this repo gets it.** `rul download` fetches the 12 text files from a GitHub mirror pinned
to a commit, falls back to a second mirror, and checks each file against the SHA-256 hash in
`src/rul/data.py`. Both mirrors were compared: identical bytes. The data is not committed here.

| Subset | Train engines | Test engines | Conditions | Fault modes |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1 (sea level) | HPC degradation |
| FD002 | 260 | 259 | 6 | HPC degradation |
| FD003 | 100 | 100 | 1 (sea level) | HPC and fan degradation |
| FD004 | 249 | 248 | 6 | HPC and fan degradation |

**Format.** One row per engine and cycle: unit id, cycle, 3 operating settings (altitude, Mach
number, throttle resolver angle), 21 sensors (names in `src/rul/config.py`). Training engines
run until failure. Test engines stop at some point before failure; `RUL_FD00x.txt` gives the
true remaining cycles at that point.

**Known properties that shape the pipeline.**

- Six to seven sensors are constant or nearly so within a condition (dropped automatically).
- In FD002/FD004 the operating condition changes from cycle to cycle and dominates the raw
  sensor values, hence per-condition normalisation.
- Each engine starts with a different, unknown amount of initial wear and manufacturing
  variation; noise is added to the sensors.

**Limitations.** Simulated, not field data: one engine model, a small number of fault modes, no
maintenance history, no missing values or sensor dropouts. Results do not transfer to real
fleets without revalidation.
