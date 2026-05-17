#!/usr/bin/env bash
# Launch the MoSL OpenPose Streamlit application.
#
# Usage:
#   ./run_app.sh              # default port 8501
#   ./run_app.sh --port 8080  # custom port
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Install dependencies if needed
python3 -m pip install --quiet --break-system-packages \
    streamlit opencv-python-headless numpy 2>/dev/null || true

echo ""
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║   MoSL · OpenPose Sign Language Demo                ║"
echo "  ║   Multimodal Moroccan Sign Language Generation      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Open your browser at: http://localhost:8501"
echo "  Press Ctrl+C to stop."
echo ""

streamlit run app.py \
    --server.headless true \
    --server.port "${1:-8501}" \
    --browser.gatherUsageStats false \
    "$@"
