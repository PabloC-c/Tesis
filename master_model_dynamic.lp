\ SCIP STATISTICS
\   Problem name     : model
\   Variables        : 4 (0 binary, 0 integer, 0 implicit integer, 4 continuous)
\   Constraints      : 5
Minimize
 Obj: +0.219000006973744 theta_0 +0.252000006467104 theta_1 +0.186000007480383 theta_2 -0.0339999995231628 theta_3
Subject to
 convex_comb: +1 theta_0 +1 theta_1 +1 theta_2 +1 theta_3 = +1
 relaxed_cons_0,0,1: = +0
 relaxed_cons_0,1,0: = +0
 relaxed_cons_1,0,1: +0.539999999105931 theta_3 = +0
 relaxed_cons_1,1,0: -0.320000016689301 theta_3 = +0
Bounds
 0 <= theta_0 <= 1
 0 <= theta_1 <= 1
 0 <= theta_2 <= 1
 0 <= theta_3 <= 1
End
