## Done
- [x] ideally we would be saving the outputs into a 'run specific' dir along with the config settings, maybe use a timestamp for the dir name?
  - Done: `outputs/runs/{timestamp}/` with `config_snapshot.json` and `pipeline_run.json`; `outputs/latest` symlink
- [x] extract `## Repository Structure` and other data like this from the `README.md` into an `AGENTS.md` file - this is where this information belongs
  - Done: AGENTS.md created with full repository structure, CLI reference, pipeline architecture, testing docs
- [x] seperate the 'populous' cities into different config and increase the number to 11 to match our pilot site numbers
- [x] it seems that no diagnostics were saved when running
  - Fixed: diagnostics now route to `{entity}/diagnostics/` subdirectories (quality heatmap, stability scatter, breakpoint timeline, trend panels)
- [x] we need a base level test suite to identify if everything is working as expected
  - Done: 260 tests across 17 test files; research-backed threshold validation with full citations
- [x] extract all formula, configs, settings and critical input/output processing steps so they are isolated and easy to test/debug
- [x] seperate site/city logic so we can run them independently
- [x] there are too many output files, we need to consolodate where we can
  - Done: entity-based output structure (`district/`, `city/`, `site/` each with `csv/`, `maps/`, `reports/`, `diagnostics/`)
- [x] structure outputs/ dir to have `site|city|district` folders with all csvs, plots, images, and pdfs containd in their respective folders
- [x] seperate data processing, analysis, and plotting into different scripts that can be run in isolation or with one script
- [x] create a bash script to do all setup and run pipeline; the goal should be one cmd to initialise everything - this may need flags for different options
  - Done: `scripts/run_all.sh`, `run_district.sh`, `run_city.sh`, `run_site.sh`, `run_preprocess.sh` — all accept passthrough args


## Critical
- [ ] check the sites against our ACTUAL pilot sites
- [ ] Udmal is in the wrong location on the map
- [ ] the annual % change in ALAN image seems to have an inverted color scheme, green is used for bad and red for good...


## Nice to have
- [ ] `processing_history` in the `data_manifest.json` is currently pretty useless - think about how this can be made less verbose and useful or delete it
- [ ] sub-district level analysis - probably decide this for a few districts strategically based on where we will likely be conducting our work
- [ ] most of the charts seem pretty useless to me, we need to think of better ways to store the data
