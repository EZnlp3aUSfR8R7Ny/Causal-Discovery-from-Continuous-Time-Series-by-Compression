import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from granger_test import granger
from data_generation import generate_continue_data, GMM
from Util import zero_change, change_to_zero_one, get_type_array_with_quantile, bh_procedure
from Disc import calculate_difference, calculate_difference_with_weight_window, calculate_difference_with_weight
from cute import bernoulli, bernoulli2, cbernoulli, cbernoulli2


def test_data(window_length, array_length, is_causal, is_linear, is_GMM, is_test_CUTE, noise):
    counter11 = 0
    counter10 = 0
    counter01 = 0
    counter00 = 0
    counter11_01 = 0
    counter10_01 = 0
    counter01_01 = 0
    counter00_01 = 0
    counter_undecided = 0
    counter_true = 0
    counter_false = 0
    p_array_CUTE1 = []
    p_array_CUTE2 = []
    p_array_improve_CUTE1 = []
    p_array_improve_CUTE2 = []
    p_array1 = []
    p_array2 = []
    for i in range(0, 1000):
        if is_causal:
            lag = random.randint(1, 3)
            cause, effect = generate_continue_data(array_length, lag, noise)
            cause_tmp = list(cause)
            effect_tmp = list(effect)
            cause = zero_change(cause)
            effect = zero_change(effect)
            if not is_linear:
                for i in range(0, len(effect)):
                    effect[i] = math.tanh(effect[i])
        else:
            lag = 5  # give a fixed lag if no causality
            if is_GMM:
                cause = GMM(3, array_length)
                effect = GMM(5, array_length)
            else:
                cause = np.random.standard_normal(array_length)
                effect = np.random.standard_normal(array_length)
            cause_tmp = list(cause)
            effect_tmp = list(effect)
            cause = zero_change(cause)
            effect = zero_change(effect)
        flag1 = False
        ce_p = granger(cause, effect, lag)
        if ce_p < 0.05:
            flag1 = True
        flag2 = False
        ce2_p = granger(effect, cause, lag)
        if ce2_p < 0.05:
            flag2 = True
        if flag1 and flag2:
            counter11 += 1
        elif flag1 and not flag2:
            counter10 += 1
        elif not flag1 and flag2:
            counter01 += 1
        elif not flag1 and not flag2:
            counter00 += 1
        if is_test_CUTE:
            cause2 = change_to_zero_one(cause)
            effect2 = change_to_zero_one(effect)
        else:
            cause2 = get_type_array_with_quantile(cause)
            effect2 = get_type_array_with_quantile(effect)
        flag3 = False
        ce3_p = granger(cause2, effect2, lag)
        if ce3_p < 0.05:
            flag3 = True
        flag4 = False
        ce4_p = granger(effect2, cause2, lag)
        if ce4_p < 0.05:
            flag4 = True
        if flag3 and flag4:
            counter11_01 += 1
        elif flag3 and not flag4:
            counter10_01 += 1
        elif not flag3 and flag4:
            counter01_01 += 1
        elif not flag3 and not flag4:
            counter00_01 += 1
        delta_ce = calculate_difference(cause, effect, window_length)
        delta_ec = calculate_difference(effect, cause, window_length)
        # print 'cause' + ' -> ' + 'effect' + ':' + str(delta_ce)
        # print 'effect' + ' -> ' + 'cause' + ':' + str(delta_ec)
        if delta_ce > delta_ec and delta_ce - delta_ec >= -math.log(0.05, 2):
            counter_true += 1
        elif delta_ec > delta_ce and delta_ec - delta_ce >= -math.log(0.05, 2):
            counter_false += 1
        else:
            counter_undecided += 1
        p = math.pow(2, -(delta_ce - delta_ec))
        p_array1.append(p)
        p_array2.append(math.pow(2, -(delta_ec - delta_ce)))
        cause = change_to_zero_one(cause_tmp)
        effect = change_to_zero_one(effect_tmp)
        cause2effect = bernoulli2(effect, window_length) - cbernoulli2(effect, cause, window_length)
        effect2cause = bernoulli2(cause, window_length) - cbernoulli2(cause, effect, window_length)
        p = math.pow(2, -(cause2effect - effect2cause))
        p_array_improve_CUTE1.append(p)
        p_array_improve_CUTE2.append(math.pow(2, -(effect2cause - cause2effect)))
        cause2effect = bernoulli(effect) - cbernoulli(effect, cause)
        effect2cause = bernoulli(cause) - cbernoulli(cause, effect)
        p_array_CUTE1.append(math.pow(2, -(cause2effect - effect2cause)))
        p_array_CUTE2.append(math.pow(2, -(effect2cause - cause2effect)))
    print "Continuous data, Granger causality test:"
    print "Two-way causality:" + str(counter11)
    print "Correct causality:" + str(counter10)
    print "Wrong causality:" + str(counter01)
    print "No causality:" + str(counter00)
    print "-----------------"
    print "Encoding data, Granger causality test:"
    print "Two-way causality:" + str(counter11_01)
    print "Correct causality:" + str(counter10_01)
    print "Wrong causality:" + str(counter01_01)
    print "No causality:" + str(counter00_01)
    print "-----------------"
    print "Encoding data, Our test:"
    print "Correct cause and effect:" + str(counter_true)
    print "Wrong cause and effect:" + str(counter_false)
    print "Undecided:" + str(counter_undecided)
    print "-----------------"
    if is_causal:
        ourmodel = bh_procedure(p_array1, 0.05)
        cute = bh_procedure(p_array_CUTE1, 0.05)
        improve_cute = bh_procedure(p_array_improve_CUTE1, 0.05)
        print "Origin CUTE Accuracy:" + str(cute)
        print "Improved CUTE Accuracy:" + str(improve_cute)
        print "Our model Accuracy:" + str(ourmodel)
    else:
        ourmodel = (bh_procedure(p_array1, 0.05) + bh_procedure(p_array2, 0.05)) / 1000.0
        cute = (bh_procedure(p_array_CUTE1, 0.05) + bh_procedure(p_array_CUTE2, 0.05)) / 1000.0
        improve_cute = (bh_procedure(p_array_improve_CUTE1, 0.05) + bh_procedure(p_array_improve_CUTE2, 0.05)) / 1000.0
        print "Origin CUTE Accuracy:" + str(1 - cute)
        print "Improved CUTE Accuracy:" + str(1 - improve_cute)
        print "Our model Accuracy:" + str(1 - ourmodel)
    return cute, improve_cute, ourmodel


def test_linear_data():
    noises = [0.0, 0.1, 0.2, 0.3]
    for noise in noises:
        test_data(6, 150, 1, 1, 0, 0, noise)
    for noise in noises:
        test_data(7, 250, 1, 1, 0, 0, noise)
    for noise in noises:
        test_data(8, 350, 1, 1, 0, 0, noise)
    for noise in noises:
        test_data(9, 450, 1, 1, 0, 0, noise)


def test_non_linear():
    noises = [0.0, 0.1, 0.2, 0.3]
    for noise in noises:
        test_data(6, 150, 1, 0, 0, 0, noise)
    for noise in noises:
        test_data(7, 250, 1, 0, 0, 0, noise)
    for noise in noises:
        test_data(8, 350, 1, 0, 0, 0, noise)
    for noise in noises:
        test_data(9, 450, 1, 0, 0, 0, noise)


def test_no_causality_consistency():
    test_data(6, 150, 0, 0, 0, 0, 0)
    test_data(7, 250, 0, 0, 0, 0, 0)
    test_data(8, 350, 0, 0, 0, 0, 0)
    test_data(9, 450, 0, 0, 0, 0, 0)

    test_data(6, 150, 0, 0, 0, 1, 0)
    test_data(7, 250, 0, 0, 0, 1, 0)
    test_data(8, 350, 0, 0, 0, 1, 0)
    test_data(9, 450, 0, 0, 0, 1, 0)


def test_causality_consistency():
    test_data(6, 150, 1, 1, 0, 0, 0)
    test_data(7, 250, 1, 1, 0, 0, 0)
    test_data(8, 350, 1, 1, 0, 0, 0)
    test_data(9, 450, 1, 1, 0, 0, 0)

    test_data(6, 150, 1, 1, 0, 1, 0)
    test_data(7, 250, 1, 1, 0, 1, 0)
    test_data(8, 350, 1, 1, 0, 1, 0)
    test_data(9, 450, 1, 1, 0, 1, 0)


def time_test_window(array_length, window_size):
    for i in range(0, 10):
        cause, effect = generate_continue_data(array_length, 3, 0)
        cause = zero_change(cause)
        effect = zero_change(effect)
        cause2effect = calculate_difference(cause, effect, window_size)
        effect2cause = calculate_difference(effect, cause, window_size)


def time_test_weighted_window(array_length, window_size):
    for i in range(0, 10):
        cause, effect = generate_continue_data(array_length, 3, 0)
        cause = zero_change(cause)
        effect = zero_change(effect)
        cause2effect = calculate_difference_with_weight_window(cause, effect, 0.7, window_size)
        effect2cause = calculate_difference_with_weight_window(effect, cause, 0.7, window_size)


def time_test_weighted(array_length, window_size):
    for i in range(0, 10):
        cause, effect = generate_continue_data(array_length, 3, 0)
        cause = zero_change(cause)
        effect = zero_change(effect)
        cause2effect = calculate_difference_with_weight(cause, effect, window_size)
        effect2cause = calculate_difference_with_weight(effect, cause, window_size)


def time_window():
    times = []
    xs = []
    for i in range(100, 5000, 100):
        xs.append(i / 100)
        start = time.clock()
        time_test_window(i, 6)
        end = time.clock()
        times.append((end - start) / 10)
    plt.plot(xs, times)
    plt.xlabel("Length($\\times10^2$)")
    plt.ylabel("Time Per Series(/s)")
    plt.show()


def time_weighted():
    times = []
    xs = []
    for i in range(100, 2000, 100):
        xs.append(i / 100)
        start = time.clock()
        time_test_weighted(i, 6)
        end = time.clock()
        times.append((end - start) / 10)
    plt.plot(xs, times)
    plt.xlabel("Length($\\times10^2$)")
    plt.ylabel("Time Per Series(/s)")
    plt.show()


def time_weighted_window():
    times = []
    xs = []
    for i in range(100, 5000, 100):
        xs.append(i / 100)
        start = time.clock()
        time_test_weighted(i, 6)
        end = time.clock()
        times.append((end - start) / 10)
    plt.plot(xs, times)
    plt.xlabel("Length($\\times10^2$)")
    plt.ylabel("Time Per Series(/s)")
    plt.show()


if __name__ == '__main__':
    test_causality_consistency()
    test_no_causality_consistency()

    test_linear_data()
    test_non_linear()

    time_window()
    time_weighted()
    time_weighted_window()
