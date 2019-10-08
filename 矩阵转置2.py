import numpy as np

element = input("请输入想转置的矩阵（逗号分隔一行中的不同值，分号分隔不同的行：）")

arr = []
arr_add = []
for ele in element:
    if ele != ';' and ele != ',':
        arr_add.append(int(ele))
    elif ele == ',':
        pass
    else:
        arr.append(arr_add)
        arr_add = []
arr.append(arr_add)

#打印原矩阵
def PrintArr(arr):
    for ele in arr:
        for e in ele:
            print("%2d" % e, end = ' ')
        print('')

print("原矩阵：")
PrintArr(arr)

def transform_arr(arr):
    
    return np.transpose(arr).tolist()

print("-"*10)
print("转置矩阵：")
PrintArr(transform_arr(arr))
