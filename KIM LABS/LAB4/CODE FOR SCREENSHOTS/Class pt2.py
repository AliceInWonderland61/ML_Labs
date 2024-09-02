class Parent:
    def __init__(self,a):
        self.a=a
    def add(self,b):
        print(self.a+b)
    def multiply(self,b):
        print(self.a*b)

class Child(Parent):
    def subtract(self,b):
        print(self.a-b)
    def divide(self,b):
        print(self.a/b)

c=Child(4)
c.add(3)
c.multiply(3)
c.subtract(3)
c.divide(3)