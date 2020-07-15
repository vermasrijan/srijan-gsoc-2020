## Environment 1
1. `conda activate openmined`
2. Initialize a Gateway - cd into PyGridNetwork repo, `python -m gridnetwork --port=5000 --host=localhost --start_local_db`
3. Initialize nodes - cd into PyGridNode repo, then run the following :
- `python -m gridnode --id=h1 --port=3000 --host=localhost --gateway_url=http://localhost:5000`
- `python -m gridnode --id=h2 --port=3001 --host=localhost --gateway_url=http://localhost:5000`

## Environment 2
1. `conda activate pysyft`
2. Run `data-owner` & `model-owner` notebooks
