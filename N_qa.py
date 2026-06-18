def id(l:int, n:int, digit_num:int):        # 十進数である L を N 桁の2進数に変換して、その中の bit1 番目の数字（0か1）を返す
    return (l >> (digit_num - n - 1)) & 1

n = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

N = len(n)

N2 = N ** 2

