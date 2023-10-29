import copy
import operator
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def Hungarian_Algorithm():
    """
    匈牙利算法，求解最优循环
    :return:最优循环列表
    """
    cost = dc1.values
    for i in range(cost.shape[0]):
        cost[i][i] = 10000
    row_ind, col_ind = linear_sum_assignment(cost)
    use_circle = []
    used_element = [0] * len(row_ind)
    while 0 in used_element:
        element = used_element.index(0)  # 可用的元素值
        used_element[element] = 1
        now = [element]
        ind = np.where(row_ind == now[-1])[0][0]
        while col_ind[ind] not in now:
            now.append(col_ind[ind])
            used_element[col_ind[ind]] = 1
            ind = np.where(row_ind == now[-1])[0][0]
        use_circle.append(now)
    return use_circle


def check_unique(p, part):
    """
    判断是否已经在种群中
    :param p: 子个体第一段
    :param part: 种群
    """
    if np.any(np.all(p != part, axis=1)):  # 如果新产生的子个体不在种群中则添加
        part = np.r_[part, [p]]
        fitness(p)
    return part


def add_initial_element(pop_num):
    """
    向种群添加不重复的精英解和随机解
    pop_num:总个体数
    return:空
    """
    part = np.array([copy.deepcopy(num)])  # 添加最初序列
    # 添加精英解
    out_layer = np.arange(len(circle_list))
    while len(part) <= population_elite_num:
        random.shuffle(out_layer)
        for k in range(len(circle_list)):
            start = random.randint(0, len(circle_list[k]) - 1)
            circle_list[k] = circle_list[k][start:] + circle_list[k][:start]
        part = check_unique(np.concatenate(circle_list), part)

    # 添加随机解
    while len(part) < pop_num:  # 当小于子个体数时
        random.shuffle(num)
        part = check_unique(num, part)


def fitness(p):
    """
    计算适应度
    :param p: 子个体，被解码的
    """
    delay_five = process(p, delay_initial, dp1, dc1, range(5))  # 前五个操作台结束时时间
    pop_six = buffer_now(delay_five, p, dp1, dc1)  # 某个体在5-6
    process_new(pop_six[0], pop_six[1], pop_six[2], dp1, dc1)  # 添加与计算


def process(now_list, delay_time, dp, dc, arr=range(6, 10), complete=False) -> list:
    """
    通过现在的加工序列，每个加工序列的延迟时间，生成对应操作台后的延迟时间
    :param now_list:加工序列
    :param delay_time:上一个部分的延迟时间
    :param dp:操作时间
    :param dc:换模时间
    :param arr:操作的工作台序号
    :param complete:判断是否完整显示
    :return:经过此部分操作后的延迟时间
    """
    end_time = []  # 每个操作台的结束时间

    for i in range(len(now_list)):  # 第几个元素
        epoch = [delay_time[i]]
        for j in arr:  # 每个元素在每台机器处的到达时间
            if i == 0:  # 当是子个体的第一个加工工件
                epoch.append(dp.iat[j, now_list[i]] + epoch[-1])  # 此工作台的延迟时间为操作时间+上一个工作台的延迟时间

            else:  # 当不是子个体的第一个加工工件
                start_time = max(end_time[-1][j - arr[0] + 1], epoch[-1]) + dc.iat[
                    now_list[i - 1], now_list[i]]  # 起始操作时间为max(上一个工件此操作台的延迟时间，此工件上一个工作台的操作时间)+换模时间
                epoch.append(start_time + dp.iat[j, now_list[i]])  # 此工作台的延迟时间为起始操作时间+操作时间
        end_time.append(epoch)  # 每个工件在每个操作台的结束时间

    if not complete:
        return [era[-1] for era in end_time]
    else:
        return [era[-1] for era in end_time], end_time


def six_merge(delay1, delay2, index1, index2):
    """
    在六处加工结束后，生成出工序六的顺序
    :param delay1: 机器一延迟时间
    :param delay2: 机器二延迟时间
    :param index1: 顺序一
    :param index2: 顺序二
    :return:排好的顺序，对应的延迟时间
    """
    delay1, delay2 = np.array(delay1), np.array(delay2)
    index_buffer = np.hstack((index1, index2))
    delay_time = np.hstack((delay1, delay2))
    delay_arg = np.argsort(delay_time)
    index_sort = index_buffer[delay_arg]
    delay_sort = np.sort(delay_time)
    return index_sort, delay_sort


def buffer_now(delay: list, index: list, dp, dc):
    """
    计算进入buffer时的状态
    :param delay:出工序5时的延迟
    :param index:出工序5的顺序
    :param dp:操作时间
    :param dc:换模时间
    :return:进入buffer的顺序，延迟时间，1-buffer的顺序
    """
    first, second = [], []
    delay_time1, delay_time2 = [], []

    first.append(index[0])
    delay_time1.append(delay[0] + dp.iat[5, index[0]])
    deed = 1  # 已加工个数
    while deed < len(index):  # 当还有工件未完成时
        if not first:  # 当first为空时
            first.append(index[deed])
            delay_time1.append(delay[deed] + dp.iat[5, index[deed]])
        elif not second:  # first不为空,second为空
            second.append(index[deed])
            delay_time2.append(delay[deed] + dp.iat[5, index[deed]] * alpha)
        else:  # 均不为空
            if delay_time1[-1] <= delay_time2[-1]:
                first.append(index[deed])
                delay_time1.append(
                    max(delay[deed], delay_time1[-1]) + dc.iat[first[-2], first[-1]] + dp.iat[
                        5, index[deed]])
            else:
                second.append(index[deed])
                delay_time2.append(
                    max(delay[deed], delay_time2[-1]) + dc.iat[second[-2], second[-1]] + dp.iat[
                        5, index[deed]] * alpha)
        deed += 1

    first, second = np.array(first), np.array(second)
    index_sort, delay_sort = six_merge(delay_time1, delay_time2, first, second)
    index_1t6 = [index, first, second]
    return index_sort, delay_sort, index_1t6


def update_inhibit_dict(index_all, delay, dp, dc):
    """
    更新禁忌表
    :param index_all:新的顺序
    :param delay: 进入7时的延迟时间
    :param dp: 操作时间
    :param dc: 换模时间
    :return: 空
    """
    ind = index_all[-1]
    name = [*index_all[0]]
    for k in range(len(index_all) - 1):
        name.append(-1)
        name.extend(index_all[k + 1])
    name = tuple(name)
    if name not in inhibit_dict.keys():
        inhibit_dict[name] = 1e5 / process(ind, delay, dp, dc)[-1]


def neighborhood_search(index, delay, dp, dc, key_words="cross"):
    index_copy = copy.deepcopy(index)
    index1, index2 = np.random.choice(component_num, size=2, replace=False)
    if key_words == "cross":
        index_copy[-1][index2], index_copy[-1][index1] = index_copy[-1][index1], index_copy[-1][index2]
    elif key_words == "insert":
        value = index_copy[-1][index1]
        if index1 < index2:
            index_copy[-1] = np.insert(index_copy[-1], index2, value)
            index_copy[-1] = np.delete(index[-1], index1)
        else:
            index_copy[-1] = np.delete(index[-1], index1)
            index_copy[-1] = np.insert(index_copy[-1], index2, value)
    elif key_words == "reverse":
        left, right = min(index1, index2), max(index1, index2)
        index_copy[-1][left:right] = index_copy[-1][left:right][::-1]
    else:
        pass
    update_inhibit_dict(index_copy, delay, dp, dc)


def process_new(index, delay, index_16, dp, dc):
    """
    在7处的邻域搜索,求解
    :param index:从buffer进入7的顺序
    :param delay:从buffer进入7的延迟时间
    :param index_16: 1-6的编码
    :param dp:操作时间
    :param dc:换模时间
    :return:此子个体较优的解
    """
    # first in first out
    index_origin = copy.deepcopy(index_16)
    index_origin.append(index)
    update_inhibit_dict(index_origin, delay, dp, dc)
    # 交换一些顺序
    neighborhood_search(index_origin, delay, dp, dc, action[0])
    neighborhood_search(index_origin, delay, dp, dc, action[1])
    neighborhood_search(index_origin, delay, dp, dc, action[2])


def get_element_index(old_array, new_array):
    """
    用来寻找新数组在旧数组的索引
    :param old_array:旧数组
    :param new_array:新数组
    :return:索引数组
    """
    index_new = []
    for era in new_array:
        index_new.append(np.where(old_array == era)[0][0])
    return np.array(index_new)


def six_decode(now_list, delay_time, dp, dc, choose):
    """
    在六处的解码
    :param now_list:输入数组6-1或者6-2
    :param delay_time:输入对应延迟时间
    :param dp:操作时间
    :param dc:换模时间
    :param choose:选择机器1还是机器2
    :return:解码离开6时的延迟时间
    """
    delay_sort = []
    if choose == 1:
        for k in range(len(now_list)):
            if k == 0:
                delay_sort.append(delay_time[0] + dp.iat[5, now_list[0]])
            else:
                delay_sort.append(
                    max(delay_time[k], delay_sort[-1]) + dp.iat[5, now_list[k]] + dc.iat[now_list[k - 1], now_list[k]])
    else:
        for k in range(len(now_list)):
            if k == 0:
                delay_sort.append(delay_time[0] + dp.iat[5, now_list[0]] * alpha)
            else:
                delay_sort.append(max(delay_time[k], delay_sort[-1]) + dp.iat[5, now_list[k]] * alpha + dc.iat[
                    now_list[k - 1], now_list[k]])
    return delay_sort


def undo(p):
    """
    解开禁忌表key的元组，返回加工元组
    :param p:禁忌key，tuple
    :return:被解开的key，加工路径
    """
    p = np.array(p)
    indices = np.where(p == -1)[0]
    p = [p[:component_num],  # 1-5
         p[component_num + 1:indices[1]],  # 6-1
         p[indices[1] + 1:-component_num - 1],  # 6-2
         p[-component_num:]]  # 7-10
    return p


def decode_6(delay, end_time, po, pn):
    """
    返回查找的延迟时间和全部时间
    :param delay:延迟时间
    :param end_time:全部加工过程
    :param po: 原数组
    :param pn: 新数组
    :return:延迟，全部加工时间
    """
    index = get_element_index(po, pn)
    delay_6 = six_decode(pn, np.array(delay)[index], dp1, dc1, 1)  # 6-1 延迟时间
    for k in range(len(index)):
        end_time[index[k]].append(delay_6[k])
    return delay_6, end_time


def decode(p):
    """
    解码子个体
    :param p:某个子个体，格式为【1-5编码，6-1编码，6-2编码，7-10编码，适应度】
    :return:返回供甘特图使用的数据
    """
    delay_15, end_time_15 = process(p[0], delay_initial, dp1, dc1, range(5), complete=True)  # 1-5 延迟时间
    delay_61, end_time_15 = decode_6(delay_15, end_time_15, p[0], p[1])
    delay_62, end_time_15 = decode_6(delay_15, end_time_15, p[0], p[2])

    # 进入buffer的顺序和时间
    index_buffer, delay_buffer = six_merge(delay_61, delay_62, p[1], p[2])
    delay_buffer = delay_buffer[get_element_index(index_buffer, p[3])]

    # 7-10 延迟时间
    delay_710, end_time_710 = process(p[3], delay_buffer, dp1, dc1, complete=True)
    print(delay_710)


# 零件号,机器号均减了1，记得最后要加1
path = 'D:/desk/industry_synthesis/数据/'
dp1 = pd.read_csv(path + 'case1_process.csv')  # case1_process
dc1 = pd.read_csv(path + 'case1_time.csv')  # case1_change
# dp1 = pd.read_csv(path + 'case2_process.csv')  # case2_process
# dc1 = pd.read_csv(path + 'case2_time.csv')  # case2_change

component_num = dp1.shape[1]  # 工件数
num = np.arange(component_num)  # 生成初始数据
circle_list = Hungarian_Algorithm()  # 最优循环
delay_initial = [0] * component_num  # 初始延迟时间，为工件数序列
population_elite_num = component_num  # 精英子个体数
action = ["cross", "insert", "reverse"]  # 邻域搜索的操作

out_circle = 1000
inside_circle = 50
population_random_num = 64  # 随机子个体数
population_num = population_elite_num + population_random_num  # 种群总个体数
alpha = 1.2  # 六号工位二号机的加工时间系数

best_time_series = []
inhibit_dict = {}  # 空禁忌表
add_initial_element(population_num)

# for kk in tqdm(range(out_circle), ncols=80, position=0, leave=True):

# 种群迭代
inhibit_dict = sorted(inhibit_dict.items(), key=operator.itemgetter(1), reverse=True)

best_time = 1e5 / inhibit_dict[0][1]
best_path = undo(inhibit_dict[0][0])
decode(best_path)

inhibit_dict = dict(inhibit_dict)
print(best_time)
print(best_path)
