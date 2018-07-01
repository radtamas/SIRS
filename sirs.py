%matplotlib notebook
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from ipywidgets import *

pp=0.1
qq=0.2
rr=0.2

ssize=100
eeps=0.001
A0=np.zeros((ssize,ssize), dtype=np.int)
tmax = 100

AA=np.copy(A0)
AA_cont = np.zeros((AA.shape[0], AA.shape[1], tmax+1), dtype=np.int)

def random_general(size,tm):
    randseq = np.zeros((size, size, tm), dtype=np.float)
    for r in range(0, tm):
        randseq[:,:,r]=np.random.random_sample((size,size))
    
    return randseq

def fertozes(Ai,p,q,r,RND,size,eps):
    S=Ai==0
    I=Ai==1
    R=Ai==5
    A=np.copy(Ai)
    for j in range(0,size):
        for k in range(0,size):
            Szomszum=(Ai[(j-1)%size,k]+Ai[(j+1)%size,k]+Ai[j,(k-1)%size]+Ai[j,(k+1)%size])%5
            if ((RND[j,k]<Szomszum*p and Ai[j,k]==0) or (RND[j,k]<eps)):
                A[j,k]=1
    A[(RND<q) & (I==True)]=5
    A[(RND<r) & (R==True)]=0
    return A


def plot_AA(ts):
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(AA_cont[:, :, ts], cmap=cmap, norm=norm)

    
def sirs_calc(A_cont,p,q,r,randomm,ts):
    RND=randomm[:, :, ts]
    A_ts = np.copy(A_cont[:, :, ts])
    
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(A_ts, cmap=cmap, norm=norm)
    A_cont[:, :, ts+1]=fertozes(A_ts,p,q,r,RND,ssize,eeps)

    

def sirs_slider(ts,p,q,r):
    sirs_calc(AA_cont,p,q,r,randomm,ts)

randomm=random_general(ssize,tmax)

cmap = colors.ListedColormap(['white', 'red', 'blue'])
bounds=[0,1,2,5]
norm = colors.BoundaryNorm(bounds, cmap.N)    

play = widgets.Play(
    interval=200,
    value=0,
    min=0,
    max=tmax-1,
    step=1,
    continuous_update=False,
    disabled=False
)
slider = widgets.IntSlider(description='time', min=0, max=tmax-1, step=1)
slider_p = widgets.FloatSlider(description='fertozodes',min=0,max=1,step=0.01,value=pp)
slider_q = widgets.FloatSlider(description='immunitas',min=0,max=1,step=0.01,value=qq)
slider_r = widgets.FloatSlider(description='gyogyulas',min=0,max=1,step=0.01,value=rr)
widgets.jslink((play, 'value'), (slider, 'value'))
ui = widgets.HBox([play, slider])
ui_param = widgets.VBox([slider_p,  slider_q,  slider_r])

VBox(children=(FloatSlider(value=pp), FloatSlider(value=qq), FloatSlider(value=rr)))
HBox(children=(Play(value=0), IntSlider(value=0)))

display(HBox([ui_param, ui]))

fig = plt.figure('SIRS sim')
ax = fig.add_subplot(111)
plt.tight_layout()
ax.set_xticks([])
ax.set_yticks([])
plt.ion()
fig.show()

out = widgets.interactive_output(sirs_slider, {'ts': slider, 'p': slider_p, 'q': slider_q, 'r': slider_r})