# 3D-WheatSpikeMorphologyExtraction
High-resolution 3D pipeline for quantifying wheat spike morphology. Includes python codes for 3D spike mesh analysis, trait extraction, thickness profiling and statistical modeling to capture spike main segments (zone of aborted spikelets, base and apex),  length, volume, curvature, and branching for advanced phenotyping.

### Requirements

  - Python ≥ 3.9 
  - Numpy
  - Pandas 
  - Scipy
  - Scikit-learn
  - Openpyxl
  - Vedo
  - pyvista

### 1-Align the spike

Use folder WheatSpikeAlignement and install the rigth python packages
In the anaconda powershell prompt specify the path to "Align_spikes_main.py", input the wheat spike spike to be aligned "WATDE0323_spike1.stl", and give the output path to save the aligned spike "WATDE0323_spike1.ply" ; see bellow:
    
      ```md
      python "...\Align_spikes_main.py" -i "...\WATDE0323_spike1.stl" -o "...\WATDE0323_spike1.ply"  

Or you can visulaze the spike before and after alignement using:

      ```md
      python "...\Align_spikes_main.py" -i "...\WATDE0323_spike1.stl" -o "...\WATDE0323_spike1.ply"  --show


### 2-Thickness profiling, length and volume extraction
Use folder "ThicknessProfileExtraction" and in the anaconda powershell prompt specify the path to "ThicknessExtraction_main.py", input the aligned wheat spike, and give the output path to save the results; see bellow:

              md
              python "path to...\slicing_spike_main.py" -i "path to...\WATDE0323_spike1.ply" -o "path to...\result.xlsx"

  Then, run the "summary_spike_metrics.py" to get the z-length, skeleton length , AUC volume, and the voxelisation volume.
  Finally, run "ThicknessProfile_over_rachis&slices.py" if you want to visualize the thickness profile over the spike length
