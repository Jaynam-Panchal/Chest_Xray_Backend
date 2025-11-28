Render deployment notes for `backend`

Environment / Secrets
- `HUGGINGFACE_REPO` (optional) — defaults to `jainam1510/Chest_Xray_model` in code.
- `HUGGINGFACE_TOKEN` (required for private repo) — add as a secret in Render.
- `PORT` — Render provides a port; the `render.yaml` sets a default but Render will override at runtime.

Start command (used in `render.yaml`):
```
gunicorn app:app --chdir backend --bind 0.0.0.0:$PORT --workers 2 --threads 4
```

Notes
- The server will attempt to predownload model files from the Hugging Face repo into `best_models/` at startup.
- Ensure `HUGGINGFACE_TOKEN` is set as a Render secret for access to a private HF repo.
- The container will fetch models at startup (and copy them into `best_models/`). On redeploy, models are re-fetched.
