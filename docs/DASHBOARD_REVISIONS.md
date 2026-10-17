# Local dashboard with reviewed revisions

Install and launch from the virtual environment:

```bash
python -m pip install -e '.[dev]'
python -m pip install -r requirements-gui.txt
python -m pytest
python -m streamlit run src/cfb_power/dashboard.py --server.address=127.0.0.1 --server.port=8501 --browser.gatherUsageStats=false
```

Keep the server local. Use one browser session for jobs; cross-session locking is not implemented. The key is saved as plaintext in .env with owner-only permissions; existing CLI key loading is unchanged. Never commit .env. Ignore /data/datasets/ and /data/runs/ before adding generated data.

Update & forecast pauses when verified snapshots conflict. Review existing snapshots (no API requests) also opens review without recapture. Select one complete version for every conflicting game, inspect differences and complete records, check approval and build. No version is preselected. Resume makes no API requests. New or modified inputs invalidate approval. Decisions apply only to this build.

CLI assembly stays strict unless explicit revision decisions and the matching review token are supplied. The dataset manifest records decisions, reviewed versions, source manifest hashes and approval time. Originals stay unchanged. Validation rejects inconsistent scores, different team identities/season and unrelated conflicts. Artifact hashes are verified; raw archives are not re-normalized. Newer does not guarantee official correctness.

Thirteen reconciliation tests passed in an isolated sandbox using the uploaded assembler, synthetic canonical data, reconstructed validation rules, and stubs for unrelated training/forecast imports. Full repository and browser validation are still required locally. The five dashboard-service tests previously passed in your local 121-test run. The HTTP error-message fixes in ingestion remain local and are not part of this commit.
