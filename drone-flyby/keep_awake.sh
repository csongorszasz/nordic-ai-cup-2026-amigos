#!/bin/bash
# Keep the screen on overnight even under the power-saver profile, which resets the
# GNOME idle settings behind us. Nudges the X idle timer; moves the pointer by 0 px,
# so it changes nothing the user can see.
#   setsid nohup ~/Desktop/nordic-ai-cup-2026-amigos/drone-flyby/keep_awake.sh >/dev/null 2>&1 &
export DISPLAY=${DISPLAY:-:0}
while true; do
  xset s off -dpms 2>/dev/null
  xset s reset 2>/dev/null
  xdotool mousemove_relative -- 0 0 2>/dev/null
  sleep 120
done
