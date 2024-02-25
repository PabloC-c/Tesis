\ SCIP STATISTICS
\   Problem name     : model
\   Variables        : 9 (2 binary, 0 integer, 0 implicit integer, 7 continuous)
\   Constraints      : 10
Minimize
 Obj: +1 x^1_0
Subject to
 duplicate_0,0,0: +1 y^-1,0_0 -1 x^-1_0 = +0
 bigm_upper1_0,0: +1 x^0_0 >= +0
 bigm_upper2_0,0: +1 x^0_0 -0.801 z^0_0 <= +0
 bigm_lower1_0,0: +1 x^0_0 +0.899999976158142 y^-1,0_0 -0.300000011920929 y^-1,0_1 >= +0.5
 bigm_lower2_0,0: +1 x^0_0 +0.899999976158142 y^-1,0_0 -0.300000011920929 y^-1,0_1 +0.401 z^0_0 <= +0.901
 duplicate_1,0,0: +1 y^0,0_0 -1 x^0_0 = +0
 bigm_upper1_1,0: +1 x^1_0 >= +0
 bigm_upper2_1,0: +1 x^1_0 -0.361 z^1_0 <= +0
 bigm_lower1_1,0: +1 x^1_0 -0.200000002980232 y^0,0_0 +0.300000011920929 y^0,0_1 >= +0.200000002980232
 bigm_lower2_1,0: +1 x^1_0 -0.200000002980232 y^0,0_0 +0.300000011920929 y^0,0_1 +0.101 z^1_0 <= +0.301000002980232
Bounds
 0 <= z^0_0 <= 1
 0 <= z^1_0 <= 1
 y^-1,0_0 free
 0.6 <= y^-1,0_1 <= 0.8
 0.2 <= x^-1_0 <= 0.4
 x^1_0 free
 y^0,0_0 free
 0 <= y^0,0_1 <= 1.001
 x^0_0 free
Binaries
 z^0_0 z^1_0
End
