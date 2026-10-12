"""Run the TSA analysis, original or this branch, on the demo data (no plots).

    <python of the original's environment> run_tsa.py orig CONFIG ORIG_CODE_DIR   (run from ORIG_CODE_DIR)
    <python with this branch installed> run_tsa.py new CONFIG                     (run from the repository)

The steps are those of dwi-tracts' compute_tract_metrics.py up to the mean TSA images.
"""

import json
import sys
import time

which, config_file = sys.argv[1], sys.argv[2]
if which == "orig":
    sys.path.insert(0, sys.argv[3])
    from dwitracts.main import DwiTracts
else:
    from dwitracts.main import DwiTracts

params = json.load(open(config_file))
clobber, debug, verbose = params["general"]["clobber"], params["general"]["debug"], params["general"]["verbose"]
tracts = DwiTracts(params)
steps = [
    ("initialize", lambda: tracts.initialize()),
    ("averages", lambda: tracts.compute_bidirectional_averages(verbose=verbose, clobber=clobber, debug=debug)),
    ("distances", lambda: tracts.compute_tract_distances(verbose=verbose, clobber=clobber, debug=debug)),
    ("unidirectional", lambda: tracts.estimate_unidirectional_tracts(verbose=verbose, clobber=clobber, debug=debug)),
    ("bidirectional", lambda: tracts.estimate_bidirectional_tracts(verbose=verbose, clobber=clobber)),
    ("pajek", lambda: tracts.tracts_to_pajek(verbose=verbose, clobber=clobber)),
    ("orientations", lambda: tracts.compute_average_orientations(verbose=verbose, clobber=clobber)),
    ("reinitialize", lambda: tracts.initialize()),  # picks up the final tracts for the regressions
    ("tsa", lambda: tracts.compute_tsa(verbose=verbose, clobber=clobber)),
    ("mean_tsa", lambda: tracts.generate_mean_tsa_images(verbose=verbose)),
]
for name, step in steps:
    start = time.time()
    ok = step()
    print(f"{which} {name}: {'ok' if ok else 'FAILED'} ({time.time() - start:.0f} s)", flush=True)
    if not ok:
        sys.exit(1)
