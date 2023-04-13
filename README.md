# Elevation Mapping Project

## Data

Data obtained from [here](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html), specifically the living_room2 RGB-D images in TUM format (without noise), poses also in the TUM format.

## Code

`psdf/sdf.py` contains all the helper sdf methods, some of which aren't being used, looking through the code should make it pretty obvious what's being done. `psdf/helpers.py` contains one method that compares tsdf and sdf because I was confused and wanted to see if there was a difference, because I couldn't see it visually, turns out there is, I did not write that function, that was GPT-4. `psdf/visualizers.py` contains visualizer methods for the things we generate, again some methods are no longer in use, it's kind of messed up. `psdf/data_loader.py` is how I load data, yes the `preprocess` method does nothing, but that's because the current data I'm using is clean as a whistle, 0 noise to mess with. `main.py` is where everything is combined. 

## To-Do
- [x] Process Data
- [x] Pointcloud from Depth
- [x] SDF
- [x] TSDF
- [ ] ESDF (half done?)
- [ ] Elevation Map (should be easy enough)
- [ ] Bayesian Updating
- [ ] Do it live
- [ ] Better scene reconstruction
