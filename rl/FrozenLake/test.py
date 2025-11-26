import random

def draw5balls():
    cnt = 0
    num_list = []
    while cnt<5:
        a = random.randint(1, 10)
        if not (a in num_list):
            num_list.append(a)
            cnt += 1
    red = 0
    green = 0
    for num in num_list:
        if num < 6: red += 1
        else: green += 1
    if (red == 5) or (green == 5): return True
    else: return False

def try_n_times(n):
    cnt = 0
    for i in range(n):
        if draw5balls: cnt += 1
    print(cnt)
    print(f'try {n} times, probability: {cnt / n}')

n = 1
random.seed(1)
for i in range(6):
    try_n_times(n)
    n *= 10