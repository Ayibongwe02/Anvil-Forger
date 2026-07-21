# Anvil — train ML models as a team, export them anywhere

A self-hosted internal tool: your team logs in to a shared workspace,
uploads a CSV or a folder of labeled images, and Anvil runs a real
AutoML sweep across several algorithms, picks the winner by
cross-validation, and shows you the leaderboard, confusion matrix, and
feature importance.

Then — the actual point — you get the model back out, three ways:

1. **Pickle bundle** — a `.zip` with the fitted scikit-learn pipeline +
   a loader script, for teams staying inside Python.
2. **Universal export** — `model.json` + a ~150-line inference engine in
   both pure Python and pure JavaScript, **zero dependencies**. No
   scikit-learn, no numpy, no ONNX runtime needed on the receiving end.
   Verified bit-for-bit identical to scikit-learn's own predictions,
   including the float32 precision scikit-learn's own tree models use
   internally.
3. **Hosted API** — every model gets a live `POST /api/v1/predict/<id>`
   endpoint, gated by a per-team API key, served directly by this app.

## Honest scope

- **Tabular models**: real AutoML — logistic/linear regression, random
  forest, gradient boosting, SVM, k-NN — picked by cross-validation.
- **Image classification**: uses classical computer-vision features
  (HOG + color histograms via scikit-image) feeding a scikit-learn
  classifier, not a deep CNN. This is genuinely useful for small,
  visually-distinct classification tasks (defect tagging, simple
  sorting) but won't match a deep-learning model on complex natural
  images. The app tells you this up front rather than pretending
  otherwise.
- **Universal export** (dependency-free) is available for logistic
  regression, linear regression, and random forest — the algorithms
  with a clean closed-form/tree-structure serialization. Gradient
  boosting, SVM, and k-NN still export via pickle bundle or the hosted
  API, just not as a dependency-free file.

## Project structure

```
anvil/
├── app.py                     # Flask entry point, all routes
├── src/
│   ├── db.py                    # SQLite schema + CRUD
│   ├── auth.py                   # team/user auth, session, login_required
│   ├── constants.py                # algorithms, task kinds, export support
│   ├── tabular_training.py           # AutoML engine for CSV data
│   ├── image_training.py               # classical-CV image classifier engine
│   ├── export_universal.py               # dependency-free JSON + Python/JS export
│   ├── export_bundle.py                    # zip builders for both file exports
│   └── api_serving.py                        # hosted prediction API blueprint
├── templates/                  # Jinja2 (dashboard, project, model detail, etc.)
├── static/css/style.css        # dark "forge" theme
├── data/                       # sqlite db + uploaded datasets + model artifacts
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000, create a team, and you're in. Share the
invite code shown after signup so teammates can join the same
workspace.

## Running with Docker

```bash
docker compose up --build
```

## Data model

SQLite (`data/anvil.db`): `teams`, `users`, `projects`, `datasets`,
`models` (metrics + leaderboard + artifact path), `api_keys`,
`prediction_log`. Training runs synchronously in the request — fine for
small/medium datasets on an internal tool; for large datasets or heavy
traffic you'd want to move `train_model()` in `app.py` to a background
worker (Celery/RQ), which the module boundaries here are already set up
to support cleanly.

## Security notes for a real deployment

- Set `ANVIL_SECRET_KEY` to a real random value (the Docker Compose file
  has a placeholder).
- This uses Flask's signed-cookie sessions — fine behind your own
  VPN/HTTPS for an internal tool, not a substitute for SSO if you need
  it.
- API keys are full-access per-team, not per-model scoped. Add scoping
  if different teammates should only reach specific models.
