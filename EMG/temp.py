import time

s = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
print(s, type(s))
