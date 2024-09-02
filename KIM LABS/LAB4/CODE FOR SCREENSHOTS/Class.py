class Parent:
    def __init__(self,a):
        self.a=a

    def add(self, b):
        print(self.a+b)

    def multiply(self,b):
        print(self.a*b)

p=Parent(3)
p.add(4)
p.multiply(4)