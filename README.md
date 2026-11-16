# Heart Health Risk Demo

An original, repository-ready Streamlit demonstration of binary heart-disease classification. It uses the Cleveland subset of the UCI Heart Disease dataset (UCI dataset ID 45), a scikit-learn preprocessing pipeline, and logistic regression. The original multi-level outcome is grouped as `0 = no disease` and `1 = any positive value` for the demonstration.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit input form and result display |
| `heart_model.py` | Fetches the public UCI data, preprocesses it, and fits the model |
| `requirements.txt` | Pinned Python dependencies |
| `.gitignore` | Excludes local environments and generated files |

No model or patient dataset is committed. The app fetches the dataset and fits its pipeline at first use; Streamlit caches the fitted model for the process lifetime. This requires outbound internet access on first load.

## Run locally

Use Python 3.10 or 3.11.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install packages and start the app:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local address, usually `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Create an empty GitHub repository under your account.
2. Copy these files into its root and push them to GitHub.
3. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
4. Select your repository and branch, set the app file to `app.py`, then deploy.
5. The first app load downloads the UCI dataset, so allow a little extra startup time.

No secrets or environment variables are required. Keep `requirements.txt` in the same directory as `app.py`.

## Data and limitations

The source is the [UCI Machine Learning Repository Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). It contains historical clinical records and a target with values indicating disease severity. This demo collapses those values to a binary label and is intended only to show a deployable machine-learning workflow. It is not validated for clinical use, individual risk estimation, diagnosis, treatment, or triage. Never enter identifying patient information.

## License

The implementation in this repository is provided under the MIT License; see `LICENSE`. The dataset remains subject to the terms provided by its source.
