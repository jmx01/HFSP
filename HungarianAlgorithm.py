import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

path = 'D:/desk/industry_synthesis/'
dc1 = pd.read_csv(path + 'case1_time.csv')  # case1_change
# cost = dc1.values
# for i in range(cost.shape[0]):
#     cost[i][i] = 10000
#
# row_ind, col_ind = linear_sum_assignment(cost)
# circle_list = []
# used_element = [0]*len(row_ind)
# while 0 in used_element:
#     element = used_element.index(0)  # 可用的元素值
#     used_element[element] = 1
#     now = [element]
#     ind = np.where(row_ind == now[-1])[0][0]
#     while col_ind[ind] not in now:
#         now.append(col_ind[ind])
#         used_element[col_ind[ind]] = 1
#         ind = np.where(row_ind == now[-1])[0][0]
#     circle_list.append(now)


# print(circle_list)
# print(row_ind)
# print(col_ind)
# print(cost[row_ind, col_ind].sum())
