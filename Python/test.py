from mpmath import mp

mp.prec = 4096

from onelevel15to1 import cost_of_one_level_15to1
from twolevel15to1 import cost_of_two_level_15to1

# both of these no longer match the paper
cost_of_one_level_15to1(0.0001, 11, 5, 5)
# cost_of_two_level_15to1(0.001, 13, 5, 5, 27, 13, 15, 4)
