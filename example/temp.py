def fun1(a=1, b=2, **kwargs):
    print('a={},b={}'.format(a, b))


def fun2(**kwargs):
    fun1(**kwargs)


def fun3(a=1, b=2, **kwargs):
    fun2(a=5, b=9)


fun3()
