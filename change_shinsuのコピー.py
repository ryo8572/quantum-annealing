def id(l:int, n:int, digit_num:int):
    """10進数を2進数に変換し、指示された桁を返す関数

        Args:
            l (int):        変換の対象となる10進数の値
            n (int):        返す桁番号を示す値
            digit_num(int): 2進数を何桁まで作成するかを示す値

        Returns:
            int: 作成した2進数の,n桁目
    """
    # 引数digit_num分の要素をもった配列の作成
    binary_num = [0]*digit_num

    for i in range(l):
        list_operation(binary_num, 0)
   
    binary_num.reverse()

    # 作成した2進数のn桁目を返す
    # print(binary_num)
    return binary_num[n]
       

def list_operation(target:list, index:int):
    """2進数の桁の繰り上げを操作する関数

        Args:
            target (list): 操作の対象となる配列
            index  (int) : 操作の対象となる桁の値
    """
    # 操作する桁の値が0だったときは、1にする
    if target[index] == 0:
        target[index] = 1

    # 操作する桁の値が1だったときは、0にし、繰り上げを行う
    elif target[index] == 1:
        target[index] = 0
        list_operation(target, index+1)


if __name__ == "__main__":
    print(id(1, 0, 3))