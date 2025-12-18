# 3D-WheatSpikeMorphologyExtraction
High-resolution 3D pipeline for quantifying wheat spike morphology. Includes python codes for 3D spike mesh analysis, trait extraction, thickness profiling and statistical modeling to capture spike main segments (zone of aborted spikelets, base and apex),  length, volume, curvature, and branching for advanced phenotyping.

## 1-Align the spike

Use folder WheatSpikeAlignement and install the rigth python packages
In the anaconda powershell prompt specify the path to "Align_spikes_main.py", input the wheat spike spike to be aligned "WATDE0323_spike1.stl", and give the output path to save the aligned spike "WATDE0323_spike1.ply" ; see bellow:
              python "...\Align_spikes_main.py" -i "...\WATDE0323_spike1.stl" -o "...\WATDE0323_spike1.ply"  --show
