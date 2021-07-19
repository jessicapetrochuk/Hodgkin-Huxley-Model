import numpy as np
import math
import matplotlib.pyplot as plt

class Model():
    # membrane capacitance
    # UNITS: uF/cm^2
    C_m = 0.9

    # max conducatance  
    # UNIT: mS/cm^2
    g_na_max = 120
    g_k_max = 36
    g_l_max = 0.3

    # nernst potential
    # UNITS: mV
    E_na = 55
    E_k = -75
    E_l = -50

    # initial conditions at t=0
    # UNITS: mV, mS
    V = -70
    t_start = 0
    t_final = 50
    delta_t = 1

    # transition rates for activation and inactivation of potassium and sodium channels
    # represented as functions of voltage
    def alpha_n(self, V):
        return (0.01 * (10 - V)) / (math.exp((10 - V) / 10) - 1)
    
    def beta_n(self, V):
        return 0.125 * math.exp(-V / 80)

    def alpha_m(self, V):
        return (0.1 * (25 - V))  / (math.exp((25 - V) / 10) - 1)
    
    def beta_m(self, V):
        return 4 * math.exp(-V / 18)
    
    def alpha_h(self, V):
        return 0.07 * math.exp(-V / 20) 
    
    def beta_h(self, V):
        return 1 / (math.exp((30 - V) / 10) + 1)

    
    # steady state solutions for transition rates 
    def n_inf(self, V):
        return (self.alpha_n(V)) / (self.alpha_n(V) + self.beta_n(V))

    def m_inf(self, V):
        return (self.alpha_m(V)) / (self.alpha_m(V) + self.beta_m(V))
    
    def h_inf(self, V):
        return (self.alpha_h(V)) / (self.alpha_h(V) + self.beta_h(V))

    
    # time constants 
    def tao_n(self, V):
        return 1 / (self.alpha_n(V) + self.beta_n(V))

    def tao_m(self, V):
        return 1 / (self.alpha_m(V) + self.beta_m(V))
    
    def tao_h(self, V):
        return 1 / (self.alpha_h(V) + self.beta_h(V))


    # calculating the next state
    def calculate_next_state(self, previous_state, transition_rate_step):
        return previous_state + transition_rate_step * self.delta_t


    # change in transition rate over change in time
    def dndt(self, V, n):
        return (self.alpha_n(V) * (1.0 - n)) - (self.beta_n(V) * n)

    def dmdt(self, V, m):
        return (self.alpha_m(V) * (1.0 - m)) - (self.beta_m(V) * m)

    def dhdt(self, V, h):
        return (self.alpha_h(V) * (1.0 - h)) - (self.beta_h(V) * h)


    # conductance through each channel
    def g_na(self, m, h):
        return self.g_na_max * m ** 3 * h

    def g_k(self, n):
        return self.g_k_max * n ** 4

    def g_l(self):
        return self.g_l_max


    # current 
    def i_na(self, V, m, h):
        return self.g_na(m, h) * (V - self.E_na)

    def i_k(self, V, n):
        return self.g_k(n) * (V - self.E_k)

    def i_l(self, V):
        return self.g_l() * (V - self.E_l)

    def i_tot(self, i_na, i_k, i_l):
        return i_na + i_k + i_l


    def main(self):
        V = -70
        data = np.zeros((self.t_final + 1))
        data[0] = V

        for t in range(self.t_final):
            # set input current
            # UNITS: uA/cm^2
            if t > 10 and t < 15:
                I_input = 50
            else:
                I_input = 0
            
            n = 0
            m = 0
            h = 0

            n = self.calculate_next_state(n, self.dndt(V, n))
            m = self.calculate_next_state(m, self.dmdt(V, m))
            h = self.calculate_next_state(h, self.dhdt(V, h))

            print(self.dndt(V, n), self.dmdt(V, m), self.dhdt(V, h))
            i_tot = self.i_tot(self.i_na(V, m, h), self.i_k(V, n), self.i_l(V))
            dv_dt = (I_input - i_tot)/ self.C_m

            V = self.calculate_next_state(V, dv_dt)    
            data[t + 1] = V
            plt.scatter(t, V)
            #plt.plot(data)
            plt.pause(0.05)
        plt.show()


if __name__ == '__main__':
    model = Model()
    model.main()