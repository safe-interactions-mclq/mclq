## MCLQ

This repository showcases *MCLQ* from [here](https://collab.me.vt.edu/pdfs/ben_ral2025.pdf). 

This code is not meant to be used in final production: it sacrifices
performance for legibility. For simulations and user studies, a mix of
`lru_caching` and parallelization were used to improve performance, those
methods are not detailed here. 

Run `python.main.py --help` to see command line parameters:

```
usage: main.py [-h] [--method {npg,ilq,mclq}] [--env {driving,pointmass}] [--live-plot] [--horizon HORIZON] [--timesteps TIMESTEPS]
               [--trials TRIALS]

options:
  -h, --help            show this help message and exit
  --method {npg,ilq,mclq}
  --env {driving,pointmass}
  --live-plot
  --horizon HORIZON
  --timesteps TIMESTEPS
  --trials TRIALS
```
