for file in *.svg; do
    inkscape "$file" --export-area-drawing --export-width=1000 --export-filename="../${file%.svg}.png"
done
