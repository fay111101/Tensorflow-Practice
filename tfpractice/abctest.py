#!/usr/bin/env python
# encoding: utf-8
"""
@author: fay
@contact: fayfeixiuhong@didiglobal.com
@time: 2020-12-29 11:21
@desc:
"""
import abc

class A(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasscheck__(cls, subclass):
        # 存在greet()返回True，不存在返回False
        if hasattr(subclass, "greet"):
            return True
        return False

class B(object):
    def greet(self):  # 定义了greet()方法
        pass

class C(object):  # 没有greet()方法
    pass

class D(B):  # 继承自B类，因此继承了greet()方法
    pass

if __name__ == "__main__":
    b = B()
    c = C()
    d = D()

    print(isinstance(b, A))  # True
    print(isinstance(c, A))  # False
    print(isinstance(d, A))  # True
