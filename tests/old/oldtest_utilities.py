#
#

def test_pi2pi():
    x = 6.7
    xl = [-32.1, -6.7, -1.2, 1.2, 6.7, 32.1]
    xa = 30*(np.random.rand(1,5)-0.5)
    print("x = ", x, "\np = ", pi2pi(x),\
          "\nxa = ", xa, "\npa = ", pi2pi(xa), "\nda = ", xa-pi2pi(xa), "\n")

@static_vars(a=0, b='hello')
def foo(c):
    foo.a += c
    print('a: {0}, b: {1}, c: {2}'.format(foo.a, foo.b, c))


if __name__ == "__main__":
    foo(3)
    foo(7)
    test_pi2pi()
    