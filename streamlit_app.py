import streamlit as st


st.set_page_config(
    layout="wide",
    page_title="Normal Conjugate model",
    initial_sidebar_state="expanded",
)

# Belief updating for mean height, one height at a time
import scipy.stats as stats
from matplotlib import pyplot as plt 
import numpy as np

# Prior Quartiles
q_m = (100,160,220)
q_sd = (2,30,100)
with st.sidebar:
    st.header('Prior Quartiles')
    st.subheader('Mean')
    q1 = st.number_input('q1',value=q_m[0])
    q2 = st.number_input('q2',value=q_m[1])
    q3 = st.number_input('q3',value=q_m[2])
    q_m = (q1,q2,q3)
    st.subheader('SD')
    q1_sd = st.number_input('q1_sd',value=q_sd[0])
    q2_sd = st.number_input('q2_sd',value=q_sd[1])
    q3_sd = st.number_input('q3_sd',value=q_sd[2])
    q_sd = (q1_sd,q2_sd,q3_sd)

    n_plots = st.number_input('Plot columns',value=4,min_value=1)

with st.expander('Data'):
    heights = np.array(list(map(float,st.text_area('Heights',value='193,182').split(','))))



# Convert CI-s to parameters 
import preliz as pz
#dsd = pz.unidimensional.maxent(pz.distributions.inversegamma.InverseGamma(),ci_sd[0]**2,ci_sd[1]**2, ci,plot=False)
#a,b = dsd.alpha, dsd.beta
#print(stats.invgamma.cdf(ci_sd[0]**2,a,scale=b),1-stats.invgamma.cdf(ci_sd[1]**2,a,scale=b))
q_sd2 = tuple(q**2 for q in q_sd)
dsd = pz.unidimensional.quartile(pz.distributions.inversegamma.InverseGamma(),*q_sd2,plot=False)
a,b = dsd.alpha, dsd.beta
print( stats.invgamma.cdf(q_sd2[0],a,scale=b),
        stats.invgamma.cdf(q_sd2[1],a,scale=b),
        stats.invgamma.cdf(q_sd2[2],a,scale=b))
print(a,b)

dm = pz.unidimensional.quartile(pz.distributions.studentt.StudentT(nu=2*a), *q_m, plot=False)
mu, lmbda = dm.mu, b/(a*dm.sigma**2)
prior = (mu,lmbda,a,b)

def posterior_after_first_n(n):
    xm, x2m = (heights[:n].mean(), (heights[:n]**2).mean()) if n>0 else (0,0)
    mu = (prior[1]*prior[0]+n*xm)/(prior[1]+n)
    lmbda = prior[1] + n
    a = prior[2] + n/2
    b = prior[3] + 0.5*n*(x2m-xm**2) + n*prior[1]*(xm-prior[0])**2/(2*(prior[1]+n))
    return mu, lmbda, a, b

st.header('Posterior')
display = st.selectbox('Display',['pp','mean','sd'])

x = np.linspace(120, 220, 1000)
xs2 = np.linspace(0,q_sd[1]*1.5,1000)**2

fig, ax = plt.subplots()
ax.axis('off')
fig.set_size_inches(8.5, 10.5)

hdi_w = 0.89
for i,n in enumerate(range(len(heights)+1)):
    sx = fig.add_subplot(6, n_plots, i+1)
    sx.get_yaxis().set_visible(False)
    
    mu,lmbda,a,b = posterior_after_first_n(n)
    
    if display=='pp': 
        y = stats.t.pdf(x,df=2*a,loc=mu,scale=np.sqrt(b*(lmbda+1)/(a*lmbda)))
        hdi = pz.distributions.studentt.StudentT(nu=2*a,mu=mu,sigma=np.sqrt(b*(lmbda+1)/(a*lmbda))).hdi(hdi_w)
    if display=='mean': 
        y = stats.t.pdf(x,df=2*a,loc=mu,scale=np.sqrt(b/(a*lmbda)))
        hdi = pz.distributions.studentt.StudentT(nu=2*a,mu=mu,sigma=np.sqrt(b/(a*lmbda))).hdi(hdi_w)
    elif display=='sd': 
        x,y = np.sqrt(xs2), stats.invgamma.pdf(xs2,a=a,scale=b)
        hdi = pz.distributions.inversegamma.InverseGamma(alpha=a,beta=b).hdi(hdi_w)
        hdi = np.sqrt(hdi[0]), np.sqrt(hdi[1])

    sx.plot(x, y)#, label="võetud %d kommi,\n %d sinised" % (N, heads))
    sx.fill_between(x, 0, y, color="#348ABD", alpha=0.4)
    if display!='sd': sx.scatter(heights[:n], np.zeros_like(heights[:n]), marker='|', color='red', s=100)
    sx.fill_between([max(hdi[0],x[0]), min(hdi[1],x[-1])], [0, 0], [max(y), max(y)], color='green', alpha=0.2)
    #plt.vlines(0.5, 0, 4, color="k", linestyles="--", lw=1)

    ax.autoscale(tight=True)

#fig.suptitle(f"Bayesian updating: {display}", y=1.02, fontsize=14)

fig.tight_layout()

st.pyplot(fig)
