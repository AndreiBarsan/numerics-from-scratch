## A few stats on Ladybug 49 dataset, using Intel TBB threading.
 * 1 thread =  97.25s, 117.73s
 * 2 threads = 59.71s
 * 4 threads = 50.80s (*)
 * 6 threads = 51.75s, 49.75s
 * 8 threads = 71.53s
 
## With the -O3 flag on my program (same Ladybug 49 dataset, and Intel TBB)
 * 1 thread  = 3.53s
 * 2 threads = 3.00s
 * 4 threads = 2.63s
 * 6 threads = 1.75s, 4.10s, 2.05s, 1.88s
 * 8 threads = 2.54s
