# Ticket Triage Lab

[![Tests](https://github.com/yasaswireddy7032/ticket-triage-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/yasaswireddy7032/ticket-triage-lab/actions)

A TF-IDF support-ticket classifier with stratified evaluation, confidence-based abstention, and an out-of-vocabulary gate.

**Status:** runnable portfolio reference implementation. Uses synthetic or hand-authored demonstration data. No employer code, customer data, or production performance claims.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
python -m unittest -v
```

The default command prints a reproducible JSON demonstration. See `app.py` for the reusable Python API. `example-output.json` records the locally executed demo; it is not an external benchmark.

## Architecture

Hand-authored labeled tickets → stratified split → train-only TF-IDF → logistic regression → confidence / vocabulary checks → route or human review.

## Validation

Tests exercise failure cases and behavioral invariants, in addition to the executable demo. GitHub Actions runs the tests and demo on pushes and pull requests. See the Actions tab for current results.

## Limitations and next steps

Tiny dataset and uncalibrated probabilities. Threshold is illustrative and is not selected from a validation set. Add real de-identified tickets, temporal validation, calibration, multilingual testing, label drift monitoring, and escalation cost analysis.

## Container

```bash
docker build -t ticket-triage-lab .
docker run --rm ticket-triage-lab
```

The Docker recipe is provided; local validation uses Python, not Docker.

## Responsible use

Review the code and understand its assumptions before deploying. Keep credentials in environment variables and exclude private data from commits. This project is an educational engineering baseline, not a production-certified service.
