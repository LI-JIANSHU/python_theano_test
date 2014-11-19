import numpy as np
import scipy.optimize as opt

from scipy.optimize import minimize



n_s=3
n_a=2
phi=range(-90,100,10)
n_phi=len(phi)

r_phi=np.zeros((n_a**2*n_phi,1),dtype=complex)
fh=open('r_phi.csv');
idx=0
for line in fh:
	r_phi[idx]=complex(line.strip().replace('i','j'))
	idx+=1


def cost_function(x):
	Theta=np.zeros((1,n_s));
	P=np.zeros((n_s,1));
	for i in range(n_s):
		Theta[0][i]=x[i]
		P[i]=x[i+n_s]		

	G=np.zeros((n_a**2,2*n_a-1))
	for k in range(n_a):
		G[k*n_a:(k+1)*n_a,(n_a-1-k):(2*n_a-1-k)]=np.eye(n_a)

	projector=np.zeros((n_a**2*n_phi,n_a**2*n_phi),dtype=float)
	part1=np.eye(n_a**2)*(n_phi-1)/n_phi;
	part2=-np.eye(n_a**2)/n_phi;
	for r in range(n_phi):
		for c in range(n_phi):
			r_start=r*n_a**2
			c_start=c*n_a**2
			if r==c:
				projector[r_start:r_start+n_a**2,c_start:c_start+n_a**2]=part1
			else:
				projector[r_start:r_start+n_a**2,c_start:c_start+n_a**2]=part2

	Space_vector=np.zeros((2*n_a-1,1));
	for k in range(len(Space_vector)):
		Space_vector[k]=range(-n_a+1,n_a)[k]


	B_list=[]
	for k in range(n_phi):
		B=np.exp(1j*np.dot(Space_vector*np.pi,np.sin((Theta+phi[k])*np.pi/180.0)))
		B_list.append(np.dot(G,B))	

	B_phi = np.concatenate(B_list,axis=0);
	
	cost=np.dot(np.dot(projector,B_phi),P)-np.dot(projector,r_phi)
	cost_norm=((np.real(cost))**2+(np.imag(cost))**2).sum()
	return cost_norm

x0 = np.array([-1, 0, 2, 1, 1, 1])
res = opt.fmin(cost_function, x0,xtol=1e-10, ftol=1e-10)
#res = opt.fmin_bfgs(cost_function, x0)
print(res)
print('=========================================')
res = minimize(cost_function, x0, method='SLSQP',
               options={'xtol': 1e-10, 'disp': True})
print(res)

