## Done
- [x] ideally we would be saving the outputs into a 'run specific' dir along with the config settings, maybe use a timestamp for the dir name?
- [x] extract `## Repository Structure` and other data like this from the `README.md` into an `AGENTS.md` file - this is where this information belongs
- [x] seperate the 'populous' cities into different config and increase the number to 11 to match our pilot site numbers
- [x] check the sites against our ACTUAL pilot sites


## Critical
- [ ] there are too many output files, we need to consolodate where we can
- [ ] extract all formula, configs, settings and critical input/output processing steps so they are isolated and easy to test/debug
- [ ] it seems that no diagnostics were saved when running
- [ ] we need a base level test suite to identify if everything is working as expected
  - maybe there are some online benchmarks we can use?
- [ ] Udmal is in the wrong location on the map
- [ ] seperate site/city logic so we can run them independently
- [ ] the annual % change in ALAN image seems to have an inverted color scheme, green is used for bad and red for good...

## Good
- [ ] structure outputs/ dir to have `site|city|district` folders with all csvs, plots, images, and pdfs containd in their respective folders
- [ ] seperate data processing, analysis, and plotting into different scripts that can be run in isolation or with one script



## Nice to have

- [ ] `processing_history` in the `data_manifest.json` is currently pretty useless - think about how this can be made less verbose and useful or delete it
- [ ] create a bash script to do all setup and run pipeline; the goal should be one cmd to initialise everything - this may need flags for different options
- [ ] sub-district level analysis - probably decide this for a few districts strategically based on where we will likely be conducting our work
- [ ] most of the charts seem pretty useless to me, we need to think of better ways to store the data