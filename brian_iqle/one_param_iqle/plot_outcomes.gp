set datafile separator "," 
set terminal png
set output "one_parameter_iqle.png"
set ylabel "Parameter Value"
set xlabel "Test Number"
set yrange [0:1]
set xrange [1:100]
set key outside
plot \
 "one_param_estimates.csv" using 1:2 title "Parameter 1" lc 7 lt 5 w l , \
 "one_param_estimates.csv" using 1:3 pt 7 lc 2 title "Linalg - param 1" ,  \
 "one_param_estimates.csv" using 1:4 pt 4 lc 1 title "Custom - param 1" ,  \
 
