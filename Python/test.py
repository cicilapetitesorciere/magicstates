from onelevel15to1 import cost_of_one_level_15to1
from twolevel15to1 import cost_of_two_level_15to1
from mpmath import mp

mp.prec = 4096

# When running this line, pphys=.001 and pphys=.0001 give the same result
cost_of_two_level_15to1(0.0001, 17, 7, 7, 41, 17, 17, 6)
