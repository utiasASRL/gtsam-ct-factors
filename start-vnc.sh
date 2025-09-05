#!/bin/bash
# start-vnc.sh (non-root version)

# Pick a display number that isn’t in use
export DISPLAY=:1

# Start a virtual framebuffer for that display
Xvfb $DISPLAY -screen 0 1920x1080x16 &

# Start desktop environment (optional)
startxfce4 &

# Start VNC server
# -rfbport lets you pick a port if 5900 is busy
x11vnc -display $DISPLAY -nopw -forever -listen 0.0.0.0 -xkb -rfbport 5900