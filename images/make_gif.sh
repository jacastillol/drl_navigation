#!/bin/bash
echo ' ******** Getting reduced frames ...'
rm -f frame/*
ffmpeg -ss 00:00:33 -t 5 -i "$1" -vf 'scale=iw*.5:ih*.5' -r 5 frame/%03d.png
echo ' ******** Making the animation gif ...'
rm -f output.gif
ffmpeg -i frame/%03d.png "$2"
echo ' ******** Slowing down the animation ...'
convert -delay 40x100 "$2" "$2"
