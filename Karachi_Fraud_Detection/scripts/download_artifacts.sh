#!/usr/bin/env bash
# Karachi Real Estate Fraud Detection — Artifact Setup
#
# .gitignore excludes data/, models/, and *.npy because they are too large
# for a normal GitHub repository (the full data/ + models/ directories are
# ~280MB combined, with several files between 16-48MB).
#
# If you are cloning this repo fresh, the large files below are NOT included
# in git history. Use this script as a placeholder for wherever you actually
# host them (Google Drive, Hugging Face Hub, S3, etc.) and document the real
# download URLs here once you upload them.
#
# Required files for running the API:
#   models/fraud_detector_v1.pkl          (~48MB)
#   models/artifacts/*.pkl                (small, can stay in git)
#
# Required files for re-running the notebooks from scratch:
#   data/raw/raw_listings.csv             (~45MB)
#
# Files NOT needed for inference (safe to exclude permanently):
#   models/lof.pkl, models/lof_pca.pkl    (~35MB combined, LOF is training-only)
#   models/artifacts/X_train_robust.npy, X_test_robust.npy,
#     X_train_standard.npy, X_test_standard.npy   (training arrays only)

set -e

echo "=================================================="
echo " Artifact Download — Karachi Fraud Detection"
echo "=================================================="
echo ""
echo "TODO: Replace the placeholders below with real URLs once you"
echo "      upload the model/data files to cloud storage."
echo ""

mkdir -p models/artifacts data/raw

# Example pattern (uncomment and fill in once you have real hosting):
# echo "Downloading trained model..."
# curl -L -o models/fraud_detector_v1.pkl "<YOUR_HOSTED_URL_HERE>"
#
# echo "Downloading raw dataset..."
# curl -L -o data/raw/raw_listings.csv "<YOUR_HOSTED_URL_HERE>"

echo "No download URLs configured yet — see comments in this script."
echo "For now, copy fraud_detector_v1.pkl and models/artifacts/*.pkl"
echo "manually into place before running the API."
