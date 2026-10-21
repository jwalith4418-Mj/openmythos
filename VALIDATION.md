# Publication validation — 2026-10-21

Selected source: `project/OPEN MTHYOS/OpenMythos_v4/OpenMythos_v4`.

## Version selection

Compared all available OpenMythos folders and ZIP archives. V4 includes inference profiling, speculative depth decoding, batching and thesis documentation beyond V3. The extracted V4 folder has 42 Python files, all syntactically valid; its ZIP has 41. The folder adds `run_application_and_agents.py` and updates `open_mythos/serving/server.py`. This existing local folder was selected without combining divergent releases.

## Tests

The existing suite passed with a local test API key configured:

```text
OPENMYTHOS_API_KEYS=dev
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
python -m pytest tests -m "not network" -q --disable-warnings --maxfail=3

104 passed, 9 deselected in 9.82s
```

The nine deselected tokenizer tests require Hugging Face access and were not run. An initial run without the API key received expected HTTP 401 responses in serving tests; after setting the fixture's test key, the entire offline suite passed. No source changes were needed.

These CPU checks do not establish GPU performance, trained model quality, distributed scaling or production readiness. The existing reports in `research/` and the source documentation are preserved as supplied; their historical benchmark, commercial and audit claims were not independently verified by this publication check.

## Packaging

Preserved the original MIT license, source and associated supplied research reports. Excluded Python/pytest caches and duplicate archives. Checked staged content for credential/private-key patterns and sensitive filenames; no actual credentials were identified. Compared against 21 accessible GitHub repositories; no existing OpenMythos project was found.
