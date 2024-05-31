index_1 = [43, 20, 46, 44, 26, 19, 29, 14, 62, 0, 12, 36]

index_2 = [43, 20, 46, 44, 19, 26, 29, 14, 12, 62, 0, 4]

for i in range(len(index_2)):
    if index_2[i] not in index_1:
        print(index_2[i])
