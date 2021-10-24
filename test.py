#
#  @author: zac-j-harris 2021
#


import random


t1 = [i for i in range(10)]
t2 = t1[::-1]
print(t1)
print(t2)
t3 = list(zip(t1, t2))
random.shuffle(t3)
t1, t2 = zip(*t3)
t1 = list(t1)
t2 = list(t2)
print(t1)
print(t2)