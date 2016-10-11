# remove the # at line 104 of chunk.py
for i in {0..9}
do
    python perc.py -m model$i > output$i
    python score-chunks.py -t output$i
done
