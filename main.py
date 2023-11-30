import copy
import operator
import random
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit
from plotly.figure_factory import create_gantt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def Hungarian_Algorithm():
    """
    匈牙利算法，求解最优循环
    :return:最优循环列表
    """
    row, col = np.diag_indices_from(dc_new)
    dc_new[row, col] = 10000
    row_ind, col_ind = linear_sum_assignment(dc_new)
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


def check_unique(ppp, part):
    """
    判断是否已经在种群中
    :param ppp: 子个体第一段
    :param part: 种群
    """
    if np.any(np.all(ppp != part, axis=1)):  # 如果新产生的子个体不在种群中则添加
        part = np.r_[part, [ppp]]
        delay_five = process(ppp, delay_initial, np.arange(5))[0]  # 前五个操作台结束时时间
        pop_six = buffer_now(delay_five, ppp)  # 某个体在5-6
        process_new(pop_six[0], pop_six[1], pop_six[2])  # 添加与计算
    return part


def add_initial_element(pop_num):
    """
    向种群添加不重复的精英解和随机解
    pop_num:总个体数
    return:空
    """
    num = np.arange(component_num)  # 生成初始数据
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


@jit(nopython=True)
def process(now_list, delay_time, arr=np.arange(6, 10)) -> list:
    """
    通过现在的加工序列，每个加工序列的延迟时间，生成对应操作台后的延迟时间
    :param now_list:加工序列
    :param delay_time:上一个部分的延迟时间
    :param arr:操作的工作台序号
    :return:经过此部分操作后的延迟时间
    """
    end_time = np.zeros((component_num, len(arr) + 1))  # 每个操作台的结束时间
    out_time = np.zeros(component_num)
    for xx in np.arange(component_num):  # 第几个元素
        epoch = np.zeros(len(arr) + 1)
        epoch[0] = delay_time[xx]
        for j in arr:  # 每个元素在每台机器处的到达时间
            s = j - arr[0]
            if xx == 0:  # 当是子个体的第一个加工工件
                epoch[s + 1] = dp_new[j, now_list[xx]] + epoch[s]
            else:  # 当不是子个体的第一个加工工件
                start_time = max(end_time[xx - 1][s + 1], epoch[s]) + dc_new[now_list[xx - 1], now_list[xx]]
                epoch[s + 1] = start_time + dp_new[j, now_list[xx]]
        end_time[xx] = epoch
        out_time[xx] = epoch[-1]
    return out_time, end_time


@jit(nopython=True)
def six_merge(delay1, delay2, index1, index2):
    """
    在六处加工结束后，生成出工序六的顺序
    :param delay1: 机器一延迟时间
    :param delay2: 机器二延迟时间
    :param index1: 顺序一
    :param index2: 顺序二
    :return:排好的顺序，对应的延迟时间
    """
    index_buffer = np.hstack((index1, index2))
    delay_time = np.hstack((delay1, delay2))
    delay_arg = np.argsort(delay_time)
    index_sort = index_buffer[delay_arg]
    delay_sort = np.sort(delay_time)
    return index_sort, delay_sort


@jit(nopython=True)
def dynamic_buffer(index, delay, machine, deed, delay_time, way):
    machine = np.append(machine, index[deed])
    if way == 1:
        if len(machine) != 1:
            delay_time = np.append(delay_time,
                                   max(delay[deed], delay_time[-1]) + dc_new[machine[-2], machine[-1]] + dp_new[
                                       5, index[deed]])
        else:  # 当first为空
            delay_time = np.append(delay_time, delay[deed] + dp_new[5, index[deed]])
    else:
        if len(machine) == 1:  # 当first为空
            delay_time = np.append(delay_time, delay[deed] + dp_new[5, index[deed]])
        else:
            delay_time = np.append(delay_time,
                                   max(delay[deed], delay_time[-1]) + dc_new[machine[-2], machine[-1]] + dp_new[
                                       5, index[deed]] * alpha)
    idle = delay_time[-1]
    deed += 1
    return machine, deed, delay_time, idle


def buffer_now(delay, index):
    """
    计算进入buffer时的状态
    :param delay:出工序5时的延迟
    :param index:出工序5的顺序
    :return:进入buffer的顺序，延迟时间，1-buffer的顺序
    """
    first, second = np.array([], dtype=int), np.array([], dtype=int)
    delay_time1, delay_time2 = np.array([]), np.array([])
    idle1, idle2 = 0, 0  # 当前机器的空闲时间
    deed = 0

    while deed < len(index):
        if delay[deed] >= idle1 and delay[deed] >= idle2:  # 两机器都有空闲
            if np.random.random() < buffer_p:
                first, deed, delay_time1, idle1 = dynamic_buffer(index, delay, first, deed, delay_time1, 1)
            else:
                second, deed, delay_time2, idle2 = dynamic_buffer(index, delay, second, deed, delay_time2, 2)
            continue

        if idle1 < idle2:  # 机器1先空闲
            first, deed, delay_time1, idle1 = dynamic_buffer(index, delay, first, deed, delay_time1, 1)
        else:  # 机器2先空闲
            second, deed, delay_time2, idle2 = dynamic_buffer(index, delay, second, deed, delay_time2, 2)

    index_sort, delay_sort = six_merge(delay_time1, delay_time2, first, second)
    index_1t6 = [index, first, second]
    return index_sort, delay_sort, index_1t6


def update_inhibit_dict(index_all, delay):
    """
    更新禁忌表
    :param index_all:新的顺序
    :param delay: 进入7时的延迟时间
    :return: 空
    """
    i0, i3 = np.array(index_all[0]), np.array(index_all[3])
    if not (i0 == i3).all():
        name = tuple(np.concatenate((i0, [-1], index_all[1], [-1], index_all[2], [-1], i3, [-1])))
        if name not in inhibit_dict.keys():
            inhibit_dict[name] = 1e5 / process(index_all[-1], delay)[0][-1]
    else:
        pass


def cross(index, delay, key_words="cross", times=1):
    index_c = []
    index_copy = copy.deepcopy(index)
    for _ in np.arange(times):
        index1, index2 = np.random.choice(component_num, size=2, replace=False)
        if key_words == "cross":
            index_copy[-1][index2], index_copy[-1][index1] = index_copy[-1][index1], index_copy[-1][index2]
        elif key_words == "insert":
            value = index_copy[-1][index1]
            if index1 > index2:
                index_copy[-1] = np.delete(index[-1], index1)
                index_copy[-1] = np.insert(index_copy[-1], index2, value)
        elif key_words == "reverse":
            left, right = min(index1, index2), max(index1, index2)
            index_copy[-1][left:right] = index_copy[-1][left:right][::-1]
        else:
            pass
        index_c.append(copy.deepcopy(index_copy))
    delay_c = np.zeros((times, component_num))
    for _ in np.arange(times):
        delay_c[_] = delay[get_element_index(index[-1], index_c[_][-1])]
    return index_c, delay_c
    # index_c = []
    # index_copy = copy.deepcopy(index)
    # for _ in np.arange(times):
    #     index1, index2 = np.random.choice(component_num, size=2, replace=False)
    #     if key_words == "cross":
    #         index_copy[-1][index2], index_copy[-1][index1] = index_copy[-1][index1], index_copy[-1][index2]
    #     elif key_words == "insert":
    #         value = index_copy[-1][index1]
    #         if index1 > index2:
    #             index_copy[-1] = np.delete(index[-1], index1)
    #             index_copy[-1] = np.insert(index_copy[-1], index2, value)
    #     elif key_words == "reverse":
    #         left, right = min(index1, index2), max(index1, index2)
    #         index_copy[-1][left:right] = index_copy[-1][left:right][::-1]
    #     else:
    #         pass
    #     index_c.append(copy.deepcopy(index_copy))
    # delay_c = np.zeros((times, component_num))
    # for _ in np.arange(times):
    #     delay_c[_] = delay[get_element_index(index[-1], index_c[_][-1])]
    # return index_c, delay_c


def neighborhood_search(index, delay, key_words="cross"):
    index_copy, delay_copy = cross(index, delay, key_words, 10)
    for u in range(len(index_copy)):
        update_inhibit_dict(index_copy[u], delay_copy[u])


def process_new(index, delay, index_16):
    """
    在7处的邻域搜索,求解
    :param index:从buffer进入7的顺序
    :param delay:从buffer进入7的延迟时间
    :param index_16: 1-6的编码
    :return:此子个体较优的解
    """
    # first in first out
    index_origin = copy.deepcopy(index_16)
    index_origin.append(index)
    update_inhibit_dict(index_origin, delay)
    # 交换一些顺序
    neighborhood_search(index_origin, delay, action[0])  # 交换
    neighborhood_search(index_origin, delay, action[1])  # 插入
    neighborhood_search(index_origin, delay, action[2])  # 倒置


@jit(nopython=True)
def get_element_index(old_array, new_array):
    """
    用来寻找新数组在旧数组的索引
    :param old_array:旧数组
    :param new_array:新数组
    :return:索引数组
    """
    return np.array([np.argwhere(old_array == era)[0][0] for era in new_array])


@jit(nopython=True)
def six_decode(now_list, delay_time, choose):
    """
    在六处的解码
    :param now_list:输入数组6-1或者6-2
    :param delay_time:输入对应延迟时间
    :param choose:选择机器1还是机器2
    :return:解码离开6时的延迟时间
    """
    nln = len(now_list)
    delay_sort = np.zeros(nln)
    if choose == 1:
        for k in np.arange(nln):
            if k == 0:
                delay_sort[k] = delay_time[0] + dp_new[5, now_list[0]]
            else:
                delay_sort[k] = max(delay_time[k], delay_sort[-1]) + dp_new[5, now_list[k]] + dc_new[
                    now_list[k - 1], now_list[k]]
    else:
        for k in np.arange(nln):
            if k == 0:
                delay_sort[k] = delay_time[0] + dp_new[5, now_list[0]] * alpha
            else:
                delay_sort[k] = max(delay_time[k], delay_sort[-1]) + dp_new[5, now_list[k]] * alpha + dc_new[
                    now_list[k - 1], now_list[k]]
    return delay_sort


def undo(ppp):
    """
    解开禁忌表key的元组，返回加工元组
    :param ppp:禁忌key，tuple
    :return:被解开的key，加工路径
    """
    ppp = np.array(ppp)
    indices = np.where(ppp == -1)[0]
    p0 = ppp[0:component_num]
    new_p = [[*p0]]
    for i in range(len(indices) - 1):
        new_p.append(ppp[indices[i] + 1:indices[i + 1]])
    return new_p, p0


@jit(nopython=True)
def decode_6(delay, end_time, p0, p1, p2):
    index1 = get_element_index(p0, p1)
    index2 = get_element_index(p0, p2)
    delay_61 = six_decode(p1, delay[index1], 1)
    delay_62 = six_decode(p2, delay[index2], 2)
    index_buffer = np.hstack((index1, index2))
    delay_time = np.hstack((delay_61, delay_62))
    end_time = np.column_stack((end_time[index_buffer], delay_time))
    end_time = end_time[get_element_index(index_buffer, np.arange(component_num))]
    return delay_61, delay_62, end_time


def decode(ppp):
    """
    解码子个体
    :param ppp:某个子个体，格式为【1-5编码，6-1编码，6-2编码，7-10编码，适应度】
    :return:返回供甘特图使用的数据
    """
    ppp[0] = np.array(ppp[0])
    delay_15, end_time_15 = process(ppp[0], delay_initial, np.arange(5))  # 1-5 延迟时间
    delay_61, delay_62, end_time_15 = decode_6(delay_15, end_time_15, ppp[0], ppp[1], ppp[2])

    # 进入buffer的顺序和时间
    index_buffer, delay_buffer = six_merge(delay_61, delay_62, ppp[1], ppp[2])
    delay_buffer = delay_buffer[get_element_index(index_buffer, ppp[3])]

    # 7-10 延迟时间
    delay_710, end_time_710 = process(ppp[3], delay_buffer)
    end_index = get_element_index(ppp[3], ppp[0])

    end_time_15 = np.delete(end_time_15, 0, axis=1)
    end_time_710 = np.delete(end_time_710, 0, axis=1)
    end_time_15 = np.c_[end_time_15, end_time_710[end_index]]

    start_time = np.zeros((component_num, 10))
    for out in range(component_num):
        for inside in range(10):
            if ppp[0][out] in ppp[2] and inside == 5:
                start_time[out][inside] = end_time_15[out][inside] - dp_new[inside, ppp[0][out]] * alpha
            else:
                start_time[out][inside] = end_time_15[out][inside] - dp_new[inside, ppp[0][out]]

    # gantt图绘制
    gantt = []
    datetime = pd.Timestamp('20231128 14:00:00')
    for component in range(component_num):
        for machine in range(10):
            gantt.append([ppp[0][component] + 1, datetime + pd.Timedelta(seconds=start_time[component][machine]),
                          datetime + pd.Timedelta(seconds=end_time_15[component][machine]), machine + 1])
    gantt = pd.DataFrame(gantt, columns=["Task", "Start", "Finish", "Resource"])
    gantt['Resource'] = gantt['Resource'].astype(str)
    fig = create_gantt(gantt, index_col='Resource', reverse_colors=True, show_colorbar=True, group_tasks=True)
    fig.show()
    print(ppp[0])
    print(ppp[1])
    print(ppp[2])
    print(ppp[3])
    print(delay_710[-1])


def population_to_batch():
    random.shuffle(population)
    batch_pop = [population[i:i + component_num] for i in range(0, len(population), component_num)]
    return batch_pop


def fetch_parents(bi):
    father_p = undo(max(bi, key=lambda x: x[-1])[0])[1]
    mother_p = np.where(np.random.random() < count_p, np.random.permutation(father_p),
                        undo(bi[random.randint(0, len(bi) - 1)][0])[1])
    return father_p, mother_p


@jit(nopython=True)
def ox(solution1: np.ndarray, solution2: np.ndarray) -> np.ndarray:
    ccc1, ccc2 = solution1, solution2
    index = np.sort(np.random.choice(component_num, 2, replace=False))
    fix1, fix2 = ccc1[index[0]:index[1]], ccc2[index[0]:index[1]]
    res12, res11, res22, res21 = ccc1[index[1]:], ccc1[:index[0]], ccc2[index[1]:], ccc2[:index[0]]
    res1, res2 = np.concatenate((res12, res11, fix1)), np.concatenate((res22, res21, fix2))
    index_c1, index_c2 = get_element_index(res1, fix2), get_element_index(res2, fix1)
    res1, res2 = np.delete(res1, index_c1), np.delete(res2, index_c2)
    ccc1 = np.concatenate((res2[:index[0]], fix1, res2[index[0]:]))
    ccc2 = np.concatenate((res1[:index[0]], fix2, res1[index[0]:]))
    return ccc1, ccc2


def update_c(f, m, cf, cdf):
    c1, c2 = ox(f, m)
    cf = np.vstack((cf, c1, c2))
    cdf = np.vstack((cdf, process(c1, delay_initial, np.arange(5))[0], process(c2, delay_initial, np.arange(5))[0]))
    return cf, cdf


def generate_child(f, m, bf, adorable_times=20, child_num=5):
    ad_num = 1e5 / bf * 0.6
    f = np.array(f)
    m = np.array(m)
    cf = np.stack((f, m), 0)
    cdf = process(f, delay_initial, np.arange(5))[0]
    cdf = np.row_stack((cdf, process(m, delay_initial, np.arange(5))[0]))
    for i in range(child_num):
        cf, cdf = update_c(f, m, cf, cdf)

    while adorable_times >= 0:
        now = np.random.randint(0, len(cf))
        if cdf[now][-1] > ad_num:
            cf = np.delete(cf, now, 0)
            cdf = np.delete(cdf, now, 0)
            new1, new2 = cf[np.random.choice(cf.shape[0], 2, replace=False)]
            cf, cdf = update_c(new1, new2, cf, cdf)
        adorable_times -= 1

    adorable_times = 20
    ad_num = 1e5 / bf * 0.8
    while adorable_times > 0:
        now = np.random.randint(0, len(cf))
        if cdf[now][-1] > ad_num:
            cf = np.delete(cf, now, 0)
            cdf = np.delete(cdf, now, 0)
        adorable_times -= 1

    for cc in np.arange(len(cf)):
        index, delay, index1_6 = buffer_now(cdf[cc], cf[cc])
        process_new(index, delay, index1_6)


def work(bi):
    father, mother = fetch_parents(bi)
    generate_child(father, mother, population_best_one[-1])
    # add_initial_element(population_num)


# 零件号,机器号均减了1，记得最后要加1
path = 'D:/desk/industry_synthesis/数据/'
reed = 2
dp1 = pd.read_csv(path + 'case' + str(reed) + '_process.csv')
dc1 = pd.read_csv(path + 'case' + str(reed) + '_time.csv')
dp_new = np.array(dp1)
dc_new = np.array(dc1)

component_num = dp1.shape[1]  # 工件数
circle_list = Hungarian_Algorithm()  # 最优循环
delay_initial = np.zeros(component_num)  # 初始延迟时间，为工件数序列
population_elite_num = component_num  # 精英子个体数
action = ["cross", "insert", "reverse"]  # 邻域搜索的操作

out_circle = 10
thread_num = 6
all_circle = out_circle * thread_num * 30
population_random_num = 64  # 随机子个体数
population_num = population_elite_num + population_random_num  # 种群总个体数
half = population_num // 2
alpha = 1.2  # 六号工位二号机的加工时间系数
count = 0  # 重复迭代数
count_p = 1 - count / all_circle
buffer_p = 0.6

best_time_series = []
inhibit_dict = {}  # 空禁忌表

add_initial_element(population_num)
for kk in tqdm(range(out_circle), ncols=80, position=0, leave=True):
    inhibit_tuple = sorted(inhibit_dict.items(), key=operator.itemgetter(1), reverse=True)
    best_time_series.append(1e5 / inhibit_dict[next(iter(inhibit_dict))])
    population = inhibit_tuple[0:half]
    population.extend(random.sample(inhibit_tuple[half:200], half))
    inhibit_dict = dict(inhibit_tuple)

    population_best_one = population[0]
    batch = population_to_batch()

    for b in batch:
        tp = []
        for t in np.arange(thread_num):
            tp.append(threading.Thread(target=work, args=(b,)))
        for t in np.arange(thread_num):
            tp[t].start()
            tp[t].join()
            best_time_series.append(1e5 / inhibit_dict[next(iter(inhibit_dict))])
            if best_time_series[-1] == best_time_series[-2]:
                count += 1
                count_p = 1 - count / all_circle
            else:
                count = 0

print(len(inhibit_dict))
inhibit_tuple = sorted(inhibit_dict.items(), key=operator.itemgetter(1), reverse=True)
decode(undo(inhibit_tuple[0][0])[0])
plt.plot(best_time_series)
plt.show()

# 优秀解
# decode([[5, 1, 6, 0, 4, 7, 3, 2], [6, 0, 4, 3], [5, 1, 7, 2], [5, 6, 0, 1, 7, 4, 3, 2]])
# decode([[5, 1, 6, 0, 4, 7, 3, 2], [5, 1, 6, 0, 4, 3], [7, 2], [5, 6, 0, 1, 7, 4, 3, 2]])  # 也行
# decode([[6, 7, 15, 5, 8, 3, 2, 1, 9, 4, 10, 0, 12, 13, 14, 11],
#         [6, 15, 8, 2, 9, 10, 13, 11],
#         [7, 5, 3, 1, 4, 0, 12, 14],
#         [6, 7, 15, 5, 8, 3, 2, 1, 9, 4, 10, 0, 12, 13, 14, 11]])
