def add1(a,b):
    c=a+b
    return c

def add2(a,b=4):
    c=a+b
    return c

def add3(a,b,c):
    d=a+b+c
    return d

print(add1(3,4))
print(add2(3))
t=(1,2,3)
print(add3(*t))