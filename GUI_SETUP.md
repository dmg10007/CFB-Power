# CFB Power local dashboard

Apply these files to the checkout containing PR #6 (feature/upcoming-predictions or main after merge). Existing files are not replaced. No GitHub changes or merges have been made by this bundle.

From the repository root in your virtual environment:

```bash
python -m pip install -e '.[dev]'
python -m pip install -r requirements-gui.txt
python -m pytest
python -m cfb_power.dashboard
```

Open the localhost address printed by Streamlit. Save your CFBD API key in Settings, test it, choose the season and weeks, then click Update & forecast. Refresh missing past weeks and the current week; defaults 6 and 7 are examples, not automatic week detection.

The dashboard reads .env explicitly and passes the key directly to the API client; it does not change CLI configuration. Existing CLI commands still require CFBD_API_KEY in their environment. Saved .env values take precedence in the dashboard. The .env is plaintext (not encrypted), written atomically with owner-only permissions. Existing .gitignore excludes it. Never paste a real key into GitHub or chat.

Before committing generated data, add /data/datasets/ and /data/runs/ to .gitignore. Add the same exclusions to any backup/sync policy as appropriate.

Jobs run in a background thread. The page polls progress; repeated clicks in that browser session are disabled. Use only one browser session: cross-session/process job locking is not implemented. Do not expose this unauthenticated dashboard over a network. Start it using the module command above, which binds to 127.0.0.1 and disables usage telemetry.

Failure stops subsequent steps. Existing snapshots are retained, and assembly may still reject conflicts in historical snapshots; this UI does not loosen validation. Ingestion currently hides some underlying error messages, so its raw audit path remains relevant. API costs: one schedule request per week plus one box request per completed game; test-key uses one teams request.

Downloads verify the forecast manifest's artifact hashes first. These hashes detect accidental changes, not malicious edits of both files and manifest.

Validation: service tests cover key handling and orchestration with mocked API operations. Browser behavior and real API capture require local smoke testing. No claim of a passing full repository suite is made for this bundle.
