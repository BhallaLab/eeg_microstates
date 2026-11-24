This code is adapted from von Wegner et al 2018 to grab 120s segments from the EO and EC conditions from CALM-Brain RS-EEG. 
The original code worked on a single state, and thus has to be modified for this two-state paradigm. Changes include grabbing 
two 120s clean segments from both conditions for each subject, pooling them and applying k-means to the pooled data, and then 
back-fitting each state. Run in batch on SCZ, DEM and HC. 

Retained steps:
  1. data  filtered with their 6th-order Butterworth band-pass (BAND)
  2. clustering on the GFP peaks of pooled EO and EC pooled sorted by GEV
  3. back-fitting of EO and of EC with these maps so a map label means the same topography in both
  4. per condition tutorial statistics: coverage, duration and occurrence per map, GEV, GFP peaks per second,
     relaxation time, entropy and entropy rate, Markov/geometric/stationarity/symmetry tests, AIF with Markov
     surrogate band and first peak (aif_peak1)
  5. per subject, how much map use changes from EO to EC (0 = identical, 1 = completely different)

Changed:
1. channels
2. average reference and montage
3. windows and artefact blocks
