# CausalNNCAM

## ***Run Tigramite (PCMCI) for SPCAM data with specified settings:***
### Fixed:
- PC-stable (i.e., MCI component not run)
- tau_min/tau_max = -1
- Significance: analytics
- experiments: '002_train_1_year'
- links: parents (state fields) -> children (parameterizations)
### Options:
- analysis: 'single': gridpoints individually
            'concat': gridpoints contatenated into a
                      single time-series
- children (parameterizations)
- region: lat/lon limits (gridpoints to be used)
- levels: children's levels to be explored
- pc_alphas: list of value(s)
