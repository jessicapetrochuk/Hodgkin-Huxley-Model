## For a dendrite infinitely long in either direction
import math

import matplotlib.pyplot as plt


def propogate_signal(V_df, C_m, g_m, lambda_m):
    V_0 = 300
    V_all_1 = []
    V_all_2 = []
    V_all_3 = []

    x_1 = 1
    x_2 = 5
    x_3 = 10

    for v in V_df:
        V_1 = v * math.exp(-abs(x_1) / lambda_m)
        V_2 = v * math.exp(-abs(x_2) / lambda_m)
        V_3 = v * math.exp(-abs(x_3) / lambda_m)

        V_all_1.append(V_1)
        V_all_2.append(V_2)
        V_all_3.append(V_3)

    # for x in range(50):
    #     V = V_0 * math.exp(- abs(x)/lambda_m)
    #     V_all.append(V)

    return V_all_1, V_all_2, V_all_3
