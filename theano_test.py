import numpy as np
import theano.tensor as T
from theano import *
import matplotlib.pyplot as plt
import sys
import random


P=T.dcol('P')
Theta=T.drow('Theta')

#p1=T.dscalar('p1')
#p2=T.dscalar('p2')
#p3=T.dscalar('p3')
#P=p1,p2,p3

#theta1=T.dscalar('theta1')
#theta2=T.dscalar('theta2')
#theta3=T.dscalar('theta3')
#Theta=theta1,theta2,theta3

n_s=3
n_a=2
phi=range(-90,100,10)
n_phi=len(phi)

G=np.zeros((n_a**2,2*n_a-1))
for k in range(n_a):
	G[k*n_a:(k+1)*n_a,(n_a-1-k):(2*n_a-1-k)]=np.eye(n_a)

#B_phi=np.zeros((n_phi*n_a**2,n_s),dtype=complex)
#B_phi=T.cmatrix('B_phi')
B_phi_r=T.dmatrix('B_phi_r')
B_phi_i=T.dmatrix('B_phi_i')

Space_vector=np.zeros((2*n_a-1,1));
for k in range(len(Space_vector)):
	Space_vector[k]=range(-n_a+1,n_a)[k]

B_r_list=[];
B_i_list=[];

for k in range(n_phi):
	#B=np.zeros((2*n_a-1,n_s),dtype=complex)
	#B=T.cmatrix('B')
	B_r=T.dmatrix('B_r')
	B_i=T.dmatrix('B_i')
	B_r=T.cos(T.dot(Space_vector*np.pi,T.sin((Theta+phi[k])*np.pi/180.0)))
	B_i=T.sin(T.dot(Space_vector*np.pi,T.sin((Theta+phi[k])*np.pi/180.0)))
	B_r=T.dot(G,B_r);
	B_i=T.dot(G,B_i);
	B_r_list.append(B_r)
	B_i_list.append(B_i)


B_phi_r = T.concatenate(B_r_list,axis=0);
B_phi_i = T.concatenate(B_i_list,axis=0);

#test_f = theano.function([Theta], B_phi)
#test_f([[-45,0,30]])
#B_phi = T.stack(G*B_list[0],G*B_list[1],B_list[3],B_list[4],B_list[5],B_list[6],B_list[7],B_list[8],B_list[9],B_list[10],B_list[11],B_list[12],B_list[13],B_list[14],B_list[15],B_list[16],B_list[17],B_list[18],B_list[2],)
	#start=k*(n_a**2);
	#B_phi=T.set_subtensor(B_phi[start:start+n_a**2,0:n_s],T.dot(G,B))


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


r_phi=np.zeros((n_a**2*n_phi,1),dtype=complex)
fh=open('r_phi.csv');
idx=0
for line in fh:
	r_phi[idx]=complex(line.strip().replace('i','j'))
	idx+=1

r_phi_r=np.real(r_phi)
r_phi_i=np.imag(r_phi)


#projector=np.zeros((n_a**2*n_phi,n_a**2*n_phi))
cost_r=T.dot(T.dot(projector,B_phi_r),P)-T.dot(projector,r_phi_r)
cost_i=T.dot(T.dot(projector,B_phi_i),P)-T.dot(projector,r_phi_i)
cost=(cost_r**2+cost_i**2).sum()
gT, gP = T.grad(cost,[Theta,P])
#cost_function = theano.function([theta1,theta2,theta3,p1,p2,p3], cost)

cost_function = theano.function([Theta,P], cost)
get_gT = theano.function([Theta,P], gT)
get_gP = theano.function([Theta,P], gP)

#cost_function([[-30,0,45]],[[1],[1],[1]])

Theta0=[[-1,0,2]]
P0=[[0.8],[1],[1.1]]
cost_value=[]
T_w=Theta0
P_w=P0
training_step=50000


lr=0.005
for i in range(training_step):
	T_w=T_w-lr*get_gT(T_w,P_w)
	P_w=P_w-lr*get_gP(T_w,P_w)
	cost_value.append(cost_function(T_w,P_w))
	sys.stdout.write("\r Processing "+str(i)+" over " + str(training_step))
	sys.stdout.flush()

	
print T_w
print P_w

plt.plot(np.log(cost_value))
plt.ylabel('cost')
plt.show()

plt.plot(cost_value[50000:])
plt.ylabel('cost')
plt.show()

dithering_cost=[]


num_dither=10000
for nd in range(num_dither):
	sys.stdout.write("\r Processing "+str(nd)+" over " + str(num_dither))
	sys.stdout.flush()
	for i in range(3):
		#print T_w
		#print T_w[0][i]
		ini_cost=cost_function(T_w,P_w)
		#print "Ini  : "+str(ini_cost)
		alpha=random.gauss(0,0.1)
		T_w2=np.copy(T_w)
		T_w2[0][i]=T_w[0][i]*(1.0+alpha)
		new_cost=cost_function(T_w2,P_w)
		#print "new 1: "+str(new_cost)
		#print T_w[0][i]
		if new_cost<ini_cost:
			T_w=T_w2
			#print "T_W["+str(i)+"] is changed 1"
			#print T_w[0][i]
		else:
			T_w2[0][i]=T_w[0][i]*(1.0-alpha)
			new_cost=cost_function(T_w2,P_w)
			#print T_w[0][i]
			#print "new 2: "+str(new_cost)
			if new_cost<ini_cost:
				T_w=T_w2
				#print "T_w["+str(i)+"] changed 2"
			else:
				new_cost=ini_cost
		dithering_cost.append(new_cost)
	for i in range(3):
		ini_cost=cost_function(T_w,P_w)
		#print "Ini: "+str(ini_cost)
		alpha=random.gauss(0,0.01)
		P_w2=np.copy(P_w)
		P_w2[i]=P_w[i]*(1.0+alpha)
		new_cost=cost_function(T_w,P_w2)
		#print "new 1: "+str(new_cost)
		if new_cost<ini_cost:
			P_w=P_w2
			#print "P_W["+str(i)+"] is changed 1"
		else:
			P_w2[i]=P_w[i]*(1.0-alpha)
			new_cost=cost_function(T_w,P_w2)
			#print "new 2: "+str(new_cost)
			if new_cost<ini_cost:
				P_w=P_w2
				#print "P_W["+str(i)+"] is changed 2"
			else:
				new_cost=ini_cost
		dithering_cost.append(new_cost)
			


plt.plot(dithering_cost)
plt.show()

plt.plot(dithering_cost[-10000:])
plt.show()





















