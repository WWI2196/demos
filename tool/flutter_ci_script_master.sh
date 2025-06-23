#!/bin/bash

set -e

DIR="${BASH_SOURCE%/*}"
source "$DIR/flutter_ci_script_shared.sh"

flutter doctor -v

declare -ar PROJECT_NAMES=(
    "agentic_app_manager"
    "genkit_flutter_agentic_app/flutter_frontend"
    "green_thumb_cloud_next_25/client"
    "ios_platform_views_io_2025"
    "vertex_ai_firebase_flutter_app"
)

ci_projects "master" "${PROJECT_NAMES[@]}"

echo "-- Success --"
