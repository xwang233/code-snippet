import cupy

cupy.show_config()

s = cupy.load('syevj-bug.npy')

sa = s.tolist()
print(sa)

a = cupy.cusolver._syevj_batched(s, 'L', True)
print('*' * 24)
print(a)

b = cupy.cusolver.syevj(s[0], 'L', True)
print('*' * 24)
print(b)

c = cupy.cusolver.syevj(s[1], 'L', True)
print('*' * 24)
print(c)