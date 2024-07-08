import math

def softmax(x):
    f_x = math.exp(x) / sum(math.exp(x))
    return f_x

def unique_check(list):
    if len(list) != len(set(list)):
        return False
    else:
        return True