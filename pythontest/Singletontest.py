#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-8-14 上午10:28
@Author  : fay
@Email   : fay625@sina.cn
@File    : Singletontest.py
@Software: PyCharm
"""
'''
多线程不安全！！
'''
class Singleton(object):
    def __init__(self):
        import time
        time.sleep(1)
        pass

    @classmethod
    def instance(cls,*args,**kwargs):
        if not hasattr(Singleton,"_instance"):
            Singleton._instance=Singleton(*args,**kwargs)
        return Singleton._instance

import threading

def task(arg):
    obj=Singleton.instance()
    print(obj)

for i in range(10):
    t=threading.Thread(target=task,args=[i,])
    t.start()