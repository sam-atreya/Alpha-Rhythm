import numpy as np
import matplotlib.pyplot as plt
#constants
g_Na=90
g_K=10
g_L=0.01
g_KL=0.0069
g_h= 0.36
g_AHP = 15
g_TLT =2
g_THT =12

E_Na=50
E_K=-100
E_L=-70
E_h= -40

class cell_biophysics():
    def __init__(self,X):
        self.X=X
        self.V, self.m, self.h, self.n, self.r, self.m_AHP, self.Ca, self.h_TLT, self.h_THT= X[0],X[1],X[2],X[3],X[4],X[5],X[6],X[7],X[8]

    def alpha_m(self):
        V_t = self.V+25
        return 0.32*(13-V_t)/(np.exp((13-V_t) / 4.0)-1.0)

    def beta_m(self):
        V_t = self.V+25
        return 0.28*(V_t-40)/(np.exp((V_t-40) / 5.0)-1)

    def alpha_h(self):
        V_t = self.V+25
        return 0.128*np.exp((17.0-V_t) / 18.0)

    def beta_h(self):
        V_t = self.V+25
        return (4.0/(np.exp((40.0-V_t)/5.0)+1))

    def alpha_n(self):
        V_t = self.V+25
        return 0.032*(15.0-V_t)/( np.exp((15-V_t) / 5.0)-1)

    def beta_n(self):
        V_t = self.V+25
        return 0.5*np.exp((10-V_t) / 40.0)

    def r_inf(self):
        return 1/(1+np.exp((self.V+60)/5.5))

    def t_r(self):
        return (20 + (1000/(np.exp((self.V+56.5)/14.2 + np.exp(-(self.V+74)/11.6)))))

    def m_AHP_inf(self):
        return 48*self.Ca**2/((48*self.Ca**2)+0.09)

    def t_m_AHP(self):
        return 1/((48*self.Ca**2)+0.09)

    def m_TLT_inf(self):
        V_t = self.V+2
        return 1/((np.exp(-(57+V_t)/6.2))+1.0)

    def h_inf_TLT(self):
        V_t = self.V+2
        return 1/(1.0+np.exp((V_t+81.0)/4.0))

    def t_inf_TLT(self):
        V_t = self.V+2
        return (30.8+((211.4+np.exp((V_t+113.2)/5.0))/(1+np.exp(V_t+84.0)/3.2)))/3.74

    def E_Ca(self):
        return 2.303*8.314*np.log(2/self.Ca)/(2*96485)

    def m_THT_inf(self):
        V_t = self.V+2
        return 1/(np.exp(-(40.1+V_t)/3.5)+1)

    def h_inf_THT(self):
        V_t = self.V+2
        return 1/(1+np.exp((V_t+62.2)/5.5))

    def t_inf_THT(self):
        V_t = self.V+2
        return 0.1483*(np.exp(-0.09398*V_t))+ 5.284*(np.exp(0.008855*V_t))
#make class of currents that inherit biophysics wala class and the current class will be inherited 
#by the HTC_cell wala class
class Currents(cell_biophysics):
    def __init__(self,X):
        cell_biophysics.__init__(self,X)
    def I_Na(self):
        return g_Na*(self.m**3)*self.h*(self.V-E_Na)
        
    def I_K(self):
        return g_K*(self.n**4)*(self.V-E_K)

    def I_L(self):
        return g_L*(self.V-E_L) + g_KL*(self.V-E_K)

    def I_H(self):
        return g_h*self.r*(self.V-E_h)

    def I_AHP(self):
        return g_AHP*(self.m_AHP**2)*(self.V-E_K)

    def I_TLT(self):
        return g_TLT*(self.m_TLT_inf()**2)*self.h_TLT*(self.V-self.E_Ca())
        
    def I_THT(self):
        return g_THT*(self.m_THT_inf()**2)*self.h_THT*(self.V-self.E_Ca())

class HTC_cell(Currents):

    #class attributes

    def __init__(self,X):
        #X should be the array containing values for V,m,n,h,r,Ca,h_THT,h_TLT,m_AHP
        Currents.__init__(self,X)
    
    def dxdt(self):
        dVdt = self.I_Na() - self.I_K() - self.I_L() -self.I_H() - self.I_AHP() -self.I_THT() -self.I_TLT()
        dmdt = self.alpha_m()*(1-self.m) - self.beta_m()*self.m
        dhdt = self.alpha_h()*(1-self.h) - self.beta_h()*self.h
        dndt = self.alpha_n()*(1-self.n) - self.beta_n()*self.n
        drdt = (self.r_inf() -self.r)/self.t_r()
        dmAHPdt = (self.m_AHP_inf() - self.m_AHP)/self.t_m_AHP()
        dCadt = (-10*(self.I_TLT()+ self.I_THT() )/(2*96489) + (0.00024 - self.Ca)/3.0)
        dhtltdt = (self.h_inf_TLT() - self.h_TLT)/self.t_inf_TLT()
        dhthtdt = (self.h_inf_THT() - self.h_THT)/self.t_inf_THT()
        return [dVdt, dmdt, dhdt, dndt, drdt, dmAHPdt,dCadt,dhtltdt,dhthtdt]
    
    def integrator(self):
        h1 = 0.01 #ms
        duration = 10000 #ms
        V_tsd = np.zeros(int(duration/h1))
        time = np.zeros(int(duration/h1))
        for t in np.arange(0,duration,h1):
            i = int(t/h1)
            V_tsd[i] = self.V
            time[i]= t
            d_dt = self.dxdt()
            self.V = self.V + d_dt[0]*h1
            self.m = self.m + d_dt[1]*h1
            self.h = self.h + d_dt[2]*h1
            self.n = self.n + d_dt[3]*h1
            self.r = self.r + d_dt[4]*h1
            self.m_AHP = self.m_AHP + d_dt[5]*h1
            self.Ca = self.Ca + d_dt[6]*h1
            self.h_TLT = self.h_TLT + d_dt[7]*h1
            self.h_THT = self.h_THT + d_dt[8]*h1
        return [time,V_tsd]
X0 = [-50,  2.63049028e-05,  9.99996535e-01,  2.09755191e-04,
  7.76347208e-02,  8.67441820e-05,  2.58531488e-04,  4.42457678e-04,
  7.16624141e-02]
one_cell = HTC_cell(X0)
z=one_cell.integrator()
plt.plot(z[0],z[1])
plt.savefig("Fig 1.png")
plt.show()
np.savetxt("Data.csv",z)


