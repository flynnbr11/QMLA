set datafile separator "," 
set terminal png
set output "one_parameter_iqle.png"
input= "1_param_estimates.csv" 
set ylabel "Parameter Value"
set xlabel "Test Number"
set yrange [0:1]
set xrange [1:100]
set key outside
plot \
 input using 1:2 title "Parameter 1" lc 1 lt 5 w l , \
 input using 1:3 pt 7 lc 2 title "Custom - param 1" ,  \
 input using 1:4 pt 4 lc 7 title "Linalg - param 1" ,  \

