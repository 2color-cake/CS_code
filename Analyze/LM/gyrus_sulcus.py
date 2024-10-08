# G:1  S:2   G&S:3 none:4
dic = {1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 2, 40: 2, 41: 2, 42: 4, 43: 4, 44: 2, 45: 2, 46: 2, 47: 2, 48: 2, 49: 2, 50: 2, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2, 61: 2, 62: 2, 63: 2, 64: 2, 65: 2, 66: 2, 67: 2, 68: 2, 69: 2, 70: 2, 71: 2, 72: 2, 73: 2, 74: 2, 75: 3, 76: 3, 77: 3, 78: 3, 79: 3, 80: 3, 81: 3, 82: 3, 83: 1, 84: 1, 85: 1, 86: 1, 87: 1, 88: 1, 89: 1, 90: 1, 91: 1, 92: 1, 93: 1, 94: 1, 95: 1, 96: 1, 97: 1, 98: 1, 99: 1, 100: 1, 101: 1, 102: 1, 103: 1, 104: 1, 105: 1, 106: 1, 107: 1, 108: 1, 109: 1, 110: 1, 111: 1, 112: 1, 113: 2, 114: 2, 115: 2, 116: 4, 117: 4, 118: 2, 119: 2, 120: 2, 121: 2, 122: 2, 123: 2, 124: 2, 125: 2, 126: 2, 127: 2, 128: 2, 129: 2, 130: 2, 131: 2, 132: 2, 133: 2, 134: 2, 135: 2, 136: 2, 137: 2, 138: 2, 139: 2, 140: 2, 141: 2, 142: 2, 143: 2, 144: 2, 145: 2, 146: 2, 147: 2, 148: 2}
# 60个G，16个G&S，68个S,4nONE
def get_dic():
    # G:1  S:2   G&S:3 none:4
    dic = {}
    # G&S
    for i in range(1, 9):
        dic[i] = 3
        dic[i + 74] = 3
    # G
    for i in range(9, 39):
        dic[i] = 1
        dic[i + 74] = 1

    # None
    for i in range(39, 44):
        dic[i] = 4
        dic[i + 74] = 4

    # 勉强
    for i in range(39, 42):
        dic[i] = 2
        dic[i + 74] = 2

    for i in range(44, 75):
        dic[i] = 2
        dic[i + 74] = 2

    dic = sorted([(key, value) for key, value in dic.items()])
    print(dict(dic))

# get_dic()
node_scale1_56 = [56, 130, 2, 116, 21, 68, 100, 58, 142, 19, 132, 27, 95, 139, 65, 93, 89, 101, 96, 99, 85, 7, 104, 69, 67,
               81, 128, 26, 44, 22, 147, 127, 11, 52, 25, 30, 103, 118, 54, 73, 15, 102, 141, 20, 29, 53, 80, 126, 94,
               28, 4, 90, 45, 6, 119, 16]

select_56 = [90, 2, 131, 130, 57, 28, 67, 132, 20, 116, 89, 56, 100, 127, 42, 7, 81, 76, 53, 82, 129, 85, 86, 147, 11, 134, 15, 21, 26, 69, 4, 94, 51, 19, 16, 88, 93, 58, 102, 27, 141, 1, 22, 49, 34, 59, 45, 46, 103, 75, 73, 142, 136, 114, 120, 77]

intersection_56 = [2, 130, 4, 132, 7, 11, 141, 142, 15, 16, 147, 20, 21, 19, 22, 26, 27, 28, 45, 53, 56, 58, 67, 69, 73, 81, 85, 89, 90, 93, 94, 100, 102, 103, 116, 127]

def g_s(node):
    d = {1: 0, 2: 0, 3: 0, 4: 0}
    for i in node:
        d[dic[i]] += 1
    print("gyrus: ", d[1])
    print("sulcus: ", d[2])
    print("G&S ", d[3])
    print("None: ", d[4])

g_s(select_56)
print()
g_s(intersection_56)
