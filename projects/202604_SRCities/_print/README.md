# FACTS dashboard 
Visualize the output from a FACTS experiment. This tool is used for QC. The current file,`facts_dashboard.py` is a version moded from Tarun's [facts.plotting.dashboard](https://github.com/Ttheegela/facts.plotting.dashboard). The version in Tarun's repo [facts.plotting.dashboard](https://github.com/Ttheegela/facts.plotting.dashboard) cannot handle non-standard scenarios, so users are asked to run this version directly in Python. 

```
python3 facts_dashboard.py \
  --ssp-dir /scratch/USR/facts_Dev/202603_SRC/facts/exp.src/src.H.ssp370/output \
  --ssp-dir /scratch/USR/facts_Dev/202603_SRC/facts/exp.src/src.HL.ssp585/output \
  --ssp-dir /scratch/USR/facts_Dev/202603_SRC/facts/exp.src/src.L.ssp245/output \
  --ssp-dir /scratch/USR/facts_Dev/202603_SRC/facts/exp.src/src.LN.ssp245/output \
  --ssp-dir /scratch/USR/facts_Dev/202603_SRC/facts/exp.src/src.M.ssp245/output \
  --ssp-dir /scratch/USR/facts_Dev/202603_SRC/facts/exp.src/src.ML.ssp245/output \
  --ssp-dir /scratch/USR/facts_Dev/202603_SRC/facts/exp.src/src.VL.ssp126/output \
  --output /scratch/USR/facts_Dev/202603_SRC/facts.postprocess.pub/projects/202604_SRCities/_print/SRCities.html \
  --title "SRCities dashboard"
```