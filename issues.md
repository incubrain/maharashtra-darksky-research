- [x] it seems that no diagnostics were saved when running
  - Fixed: diagnostics now route to `{entity}/diagnostics/` subdirectories (quality heatmap, stability scatter, breakpoint timeline, trend panels)
- [x] we need a base level test suite to identify if everything is working as expected
  - maybe there are some online benchmarks we can use?
  - Done: 231 tests across 14 test files; research-backed threshold validation with full citations
- [x] ideally we would be saving the outputs into a 'run specific' dir along with the config settings, maybe use a timestamp for the dir name?
  - Done: `outputs/runs/{timestamp}/` with `config_snapshot.json` and `pipeline_run.json`; `outputs/latest` symlink
- [ ] Udmal is in the wrong location on the map
- [x] extract `## Repository Structure` and other data like this from the `README.md` into an `AGENTS.md` file - this is where this information belongs
  - Done: AGENTS.md created with full repository structure, CLI reference, pipeline architecture, testing docs
- [ ] check the sites against our ACTUAL pilot sites
- [ ] seperate the 'populous' cities into different config and increase the number to 11 to match our pilot site numbers
- [x] there are too many output files, we need to consolodate where we can
  - Done: entity-based output structure (`district/`, `city/`, `site/` each with `csv/`, `maps/`, `reports/`, `diagnostics/`)


## Nice to have

- [x] create a bash script to do all setup and run pipeline; the goal should be one cmd to initialise everything - this may need flags for different options
  - Done: `scripts/run_all.sh`, `run_district.sh`, `run_city.sh`, `run_site.sh`, `run_preprocess.sh` — all accept passthrough args
- [ ] sub-district level analysis - probably decide this for a few districts strategically based on where we will likely be conducting our work
- [ ] most of the charts seem pretty useless to me, we need to think of better ways to store the data
