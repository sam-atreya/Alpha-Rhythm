import numpy as np
import matplotlib.pyplot as plt
from  scipy.signal import hilbert
import math

#pbs_index = int(sys.argv[0])
#g_GJ=g_GJ_arr[pbs_index]

def alpha_m(V):
    V_t = V+25
    return 0.32*(13-V_t)/(np.exp((13-V_t) / 4.0)-1.0)

def beta_m(V):
    V_t = V+25
    return 0.28*(V_t-40)/(np.exp((V_t-40) / 5.0)-1)

def alpha_h(V):
    V_t = V+25
    return 0.128*np.exp((17.0-V_t) / 18.0)

def beta_h(V):
    V_t = V+25
    return (4.0/(np.exp((40.0-V_t)/5.0)+1))

def alpha_n(V):
    V_t = V+25
    return 0.032*(15.0-V_t)/( np.exp((15-V_t) / 5.0)-1)

def beta_n(V):
    V_t = V+25
    return 0.5*np.exp((10-V_t) / 40.0)

def r_inf(V):
    return 1/(1+np.exp((V+60)/5.5))

def t_r(V):
    return (20 + (1000/(np.exp((V+56.5)/14.2 + np.exp(-(V+74)/11.6)))))

def m_AHP_inf(Ca):
    return 48*Ca**2/((48*Ca**2)+0.09)

def t_m_AHP(Ca):
    return 1/((48*Ca**2)+0.09)

def m_TLT_inf(V):
    V_t = V+2
    return 1/((np.exp(-(57+V_t)/6.2))+1.0)

def h_inf_TLT(V):
    V_t = V+2
    return 1/(1.0+np.exp((V_t+81.0)/4.0))

def t_inf_TLT(V):
    V_t = V+2
    return (30.8+((211.4+np.exp((V_t+113.2)/5.0))/(1+np.exp(V_t+84.0)/3.2)))/3.74

def E_Ca(Ca):
    return 2.303*8.314*np.log(2/Ca)/(2*96485)

def m_THT_inf(V):
    V_t = V+2
    return 1/(np.exp(-(40.1+V_t)/3.5)+1)

def h_inf_THT(V):
    V_t = V+2
    return 1/(1+np.exp((V_t+62.2)/5.5))

def t_inf_THT(V):
    V_t = V+2
    return 0.1483*(np.exp(-0.09398*V_t))+ 5.284*(np.exp(0.008855*V_t))

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

def I_Na(V,m,h):
    return g_Na*(m**3)*h*(V-E_Na)
    
def I_K(V,n):
    return g_K*(n**4)*(V-E_K)

def I_L(V):
    return g_L*(V-E_L) + g_KL*(V-E_K)

def I_H(V,r):
    return g_h*r*(V-E_h)

def I_AHP(V,m_AHP):
    return g_AHP*(m_AHP**2)*(V-E_K)

def I_TLT(V, h_TLT, Ca):
    return g_TLT*(m_TLT_inf(V)**2)*h_TLT*(V-E_Ca(Ca))
     
def I_THT(V, h_THT, Ca):
    return g_THT*(m_THT_inf(V)**2)*h_THT*(V-E_Ca(Ca))

def I_GJ12(V, V2):
    return g_GJ*(V2 - V)

def f_V(V,m,h,n,I,r,Ca, h_TLT, h_THT, m_AHP):
    dVdt=I-I_Na(V,m,h)-I_K(V,n)-I_L(V)- I_H(V,r) -I_TLT(V, h_TLT, Ca) - I_THT(V, h_THT, Ca) - I_AHP(V,m_AHP)
    return dVdt

def f_m(V,m):
    dmdt=alpha_m(V)*(1-m)-beta_m(V)*m
    return dmdt

def f_h(V,h):
    dhdt=alpha_h(V)*(1-h)-beta_h(V)*h
    return dhdt
   
def f_n(V,n):
    dndt=alpha_n(V)*(1-n)-beta_n(V)*n
    return dndt

def f_r(V,r):
    drdt=(r_inf(V) -r)/t_r(V)
    return drdt

def f_m_AHP(Ca, m_AHP):
    return (m_AHP_inf(Ca) - m_AHP)/t_m_AHP(Ca)

def f_Ca(V, Ca, h_TLT, h_THT):
    #print(Ca)
    return (-10*(I_TLT(V, h_TLT, Ca)+ I_THT(V, h_THT, Ca) )/(2*96489) + (0.00024 - Ca)/3.0)

def f_h_TLT(V,h_TLT):
    return (h_inf_TLT(V) - h_TLT)/t_inf_TLT(V)

def f_h_THT(V,h_THT):
    return (h_inf_THT(V) - h_THT)/t_inf_THT(V)

duration = 100 #ms
h1 = 0.01  #ms
neurons = 2
m_list=np.zeros((neurons,int(duration/h1)))
n_list=np.zeros((neurons,int(duration/h1)))
h_list=np.zeros((neurons,int(duration/h1)))
r_list=np.zeros((neurons,int(duration/h1)))
Ca_list =np.zeros((neurons,int(duration/h1)))
h_THT_list =np.zeros((neurons,int(duration/h1)))
h_TLT_list =np.zeros((neurons,int(duration/h1)))
m_AHP_list =np.zeros((neurons,int(duration/h1)))
V_list =np.zeros((neurons,int(duration/h1)))
time = np.zeros((neurons, int(duration/h1)))
def integrator (h1,V_initial,V2,g_GJ,noise): 
    for i in range(0,neurons):
        m_list[i][0]=2.63049028e-05
        n_list[i][0]=2.09755191e-04
        h_list[i][0]=9.99996535e-01
        r_list[i][0]=7.76347208e-02
        Ca_list[i][0]=2.58531488e-04
        h_TLT_list[i][0]= 4.42457678e-04
        h_THT_list[i][0]=7.16624141e-02
        m_AHP_list[i][0] = 8.67441820e-05
        V_list[i][0]=V_initial
    print(V_list,m_list,n_list,Ca_list)
	#here dt=h1=0.01ms, if the simul is for 10,000ms, its a 10s activity
	#Rn, I intend to run it for (1min) 10s, and discard the first (10s) 1s.
    for t in np.arange(h1,duration,h1):
        I=0
        I2=0
        k = int(t/h1)
        for i in range(0,neurons):
            V0 = V_list[i][k-1]
            m0 = m_list[i][k-1]
            h0 = h_list[i][k-1]
            n0 = n_list[i][k-1]
            r0 = r_list[i][k-1]
            Ca0 = Ca_list[i][k-1]
            h_TLT0 = h_TLT_list[i][k-1]
            h_THT0 = h_THT_list[i][k-1]
            m_AHP0 = m_AHP_list[i][k-1]

            #no gap junction yet
            V0=V0+(f_V(V0,m0,h0,n0,I,r0,Ca0, h_TLT0,h_THT0, m_AHP0) + noise*np.random.normal(0,1,1)*h1**(0.5))*h1 
            m0=m0+f_m(V0,m0)*h1
            h0=h0+f_h(V0,h0)*h1
            n0=n0+f_n(V0,n0)*h1
            r0=r0+f_r(V0,r0)*h1
            m_AHP0= m_AHP0+f_m_AHP(Ca0, m_AHP0)*h1
            print(Ca0)
            Ca0= Ca0+ f_Ca(V0, Ca0, h_TLT0,h_THT0)*h1
            h_TLT0 =h_TLT0 + f_h_TLT(V0,h_TLT0)*h1
            h_THT0 =h_THT0 + f_h_THT(V0,h_THT0)*h1
            V_list[i][k] = V0
            m_list[i][k] = m0
            h_list[i][k] = h0
            n_list[i][k] = n0
            r_list[i][k] = r0
            Ca_list[i][k] =Ca0
            h_TLT_list[i][k] =h_TLT0
            h_THT_list[i][k] = h_THT0
            time[i][k] = t
        
    return [V_list,time,Ca_list]

#pbs_index = int(sys.argv[0])
g_GJ_arr=[0.0, 0.001,0.004,0.008,0.01,0.016,0.02,0.030,0.040,0.06, 0.1]
#g_GJ=g_GJ_arr[pbs_index]

noise_arr = [0,1,4,10]
"""
for noise in noise_arr:
	for g_GJ in g_GJ_arr:
"""
g_GJ = 0.004
noise = 2
h1=0.01 #ms
V_initial=-60	#mV
V2=-60	#mV
z=integrator(h1,V_initial,V2,g_GJ,noise)
print(z[2])
#np.savetxt("Data_"+str(g_GJ)+"_"+str(noise)+".csv",z)
plt.plot(z[1][0],z[0][0])
plt.show()

