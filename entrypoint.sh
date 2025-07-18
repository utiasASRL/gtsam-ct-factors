#!/bin/bash

# Execute the command passed to the container (or default to an interactive bash shell)
echo "Ready to connect..."
exec "$@"