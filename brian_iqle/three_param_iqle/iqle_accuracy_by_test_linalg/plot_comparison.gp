set datafile separator "," 
set terminal png
set output "three_parameter_iqle.png"
set ylabel "Parameter Value"
set xlabel "Test Number"
set yrange [0:1]
set xrange [1:100]
set key outside
plot \
 "iqle_3_params.csv" using 1:2 title "Parameter 1" lc 1 lt 5 w l , \
 "iqle_3_params.csv" using 1:5 pt 7 lc 1 title "Linalg - param 1" ,  \
 "iqle_3_params.csv" using 1:8 pt 4 lc 1 title "Custom - param 1" ,  \
 "iqle_3_params.csv" using 1:3 title "Parameter 2"  lc 2 w l, \
 "iqle_3_params.csv" using 1:6 pt 7 lc 2 lt 5 title "Linalg - param 2" ,  \
 "iqle_3_params.csv" using 1:9 pt 4 lc 2 title "Custom - param 2" ,  \
 "iqle_3_params.csv" using 1:4 title "Parameter 3" lc 3 lt 5 w l,  \
 "iqle_3_params.csv" using 1:7 pt 7 lc 3 title "Linalg - param 3" ,  \
 "iqle_3_params.csv" using 1:10 pt 4 lc 3 title "Custom - param 3",  \
 
