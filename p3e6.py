from matplotlib.pylab import *
from matplotlib import cm
import numpy as np
import datetime
import time
from scipy.interpolate import interp1d



a = 0.5
b = 0.5
z = 1.

Nx = 20
Ny = 20
Nz = 20 *2

dx = b/Ny
dy = a/Nx
dz = z/Nz

if dx!=dy or dx!=dz:
	print(f'dx = {dx}')
	print(f'dy = {dy}')
	print(f'dz = {dz}')
	print('Error de dominio')
	exit(-1)

coords = lambda i,j:(dx*i,dy*j)

u_k = zeros((Nx+1,Ny+1,Nz+1),dtype=double)
u_km1 = zeros((Nx+1,Ny+1,Nz+1),dtype=double)



K = 0.001495 #kW / m C
c = 1.023 #kJ / kg C
ρ = 2476. #kg /m3

print(f'dx={dx} [m]')
print(f'dy={dy} [m]')
print(f'dy={dz} [m]')
print(f'K={K} [kW / m C]')
print(f'c={c} [kJ / kg C]')
print(f'rho={ρ} [kg /m3]')
#print(f'alpha={α}')


minuto = 60.
hora = 60. * minuto
dia = 24. * hora

dt = 10.#1.* minuto
dnext_t= 0.5*hora

next_t = 0
framenum = 0

Days = 2.* dia

def truncate(n,decimals=0):
	multiplier = 10 ** decimals
	return int(n*multiplier)/multiplier
def imshowbien(u):
	imshow(u.T[Nx::-1,:,int(Nz/2)],cmap=cm.coolwarm)#,interpolation='bilinear')
	cbar=colorbar(extend='both')#,cmap=cm.coolwarm)
	ticks=arange(20,70,5)
	ticks_Text=[f'{deg}°' for deg in ticks]
	cbar.set_ticks(ticks)
	cbar.set_ticklabels(ticks_Text)
	clim(20,65)

	xlabel('x')
	ylabel('y',rotation='horizontal')
	xTicks_N = arange(0,Nx+1,3)
	yTicks_N = arange(0,Ny+1,3)
	xTicks = [coords(i,0)[0] for i in xTicks_N]
	yTicks = [coords(0,j)[1] for j in yTicks_N]
	xTicks_Text=['{0:.2f}'.format(tick) for tick in xTicks]
	yTicks_Text=['{0:.2f}'.format(tick) for tick in yTicks]
	xticks(xTicks_N,xTicks_Text,rotation='vertical')
	yticks(yTicks_N,yTicks_Text)
	margins(0,2)
	subplots_adjust(bottom=0.15)



time_format = "%d-%m-%y %H:%M:%S"

fname = 'caso_1_camara_de_curado.csv' #Caso 1: Camara de curado

archivo = open(fname) 

sensores=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

tiempo = []
primer_paso = True
for linea in archivo:
	lin = linea.split(',')
	sens = 0
	dia_ = lin[0]
	hora_ = lin[1]

	if primer_paso:
		t1 = datetime.datetime.strptime(dia_+" "+hora_,time_format)
		primer_paso = False
	t2 = datetime.datetime.strptime(dia_+" "+hora_,time_format)
	t = (t2 - t1).total_seconds()

    #print(f"{dia_} {hora_} -->  t1 = {t1} t2 = {t2} dt = {t}")
	tiempo.append(t)

	for i in [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]:
		if i == 29:
			sensores[13].append(float(lin[i]))
		sens+=1


Superficie = interp1d(tiempo,sensores[13])(arange(int(Days/dt))*dt)


u_k[:,:,:] = Superficie[0]


P1 = zeros(int(Days/dt))
P2 = zeros(int(Days/dt))
P3 = zeros(int(Days/dt))
P4 = zeros(int(Days/dt))
P5 = zeros(int(Days/dt))
P6 = zeros(int(Days/dt))
P7  = zeros(int(Days/dt))
P8  = zeros(int(Days/dt))
P9  = zeros(int(Days/dt))
P10 = zeros(int(Days/dt))
P11 = zeros(int(Days/dt))
P12 = zeros(int(Days/dt))
P14 = zeros(int(Days/dt))

#Calor de Hidratacion
def Calor_de_hidratacion(t,DC = 360.):		#[tiempo, Dosificacion cemento en kg/m^3]
	#datos registrados por ensayo semi-adiabatico de Langavant

	x = [0, 130, 250, 400, 540, 720, 900, 1100, 1300, 1500, 1800, 3600, 5400, 7200, 9000, 10800, 14400, 18000, 21500, 25200, 28000, 32400, 36000, 39600, 43200, 46800, 50000, 54000, 61200, 72000, 79200, 86400, 100800, 111600, 122400, 133200, 144000, 180000, 216000, 234000, 259200, 288000, 324000, 360000, 432000, 468000, 504000, 540000, 604800, 648000, 684000, 720000, 756000, 792000, 828000, 864000, 900000, 936000, 972000, 1008000, 1044000, 1080000, 1116000, 1152000, 1188000, 1224000]
	y = [0.0, 0.019230769, 0.016666667, 0.013333333, 0.010714286, 0.009444444, 0.007222222, 0.006, 0.005150000, 0.00385, 0.003, 0.001444444, 0.001055556, 0.000722222, 0.000833333, 0.001111111, 0.002166667, 0.003472222, 0.005285714, 0.006756757, 0.007142857, 0.007272727, 0.006944444, 0.006250000, 0.005277778, 0.004027778, 0.003281250, 0.0025, 0.001875, 0.001111111, 0.000833333, 0.000694444, 0.000416667, 0.000296296, 0.000194444, 0.000111111, 0.000064815, 0.000027778, 0.000013889, 0.000027778, 0.000031746, 0.000031250, 0.000038889, 0.000061111, 0.000076389, 0.000072222, 0.000066667, 0.000055556, 0.000046296, 0.000023148, 0.000019444, 0.000030556, 0.000033333, 0.000041667, 0.000033333, 0.000030556, 0.000033333, 0.000041667, 0.000033333, 0.000030556, 0.000033333, 0.000041667, 0.000033333, 0.000030556, 0.000033333, 0.000041667]
	
	q = interp1d(x,y)


	return q(t)*DC
DC = .160 
x = [0, 130, 250, 400, 540, 720, 900, 1100, 1300, 1500, 1800, 3600, 5400, 7200, 9000, 10800, 14400, 18000, 21500, 25200, 28000, 32400, 36000, 39600, 43200, 46800, 50000, 54000, 61200, 72000, 79200, 86400, 100800, 111600, 122400, 133200, 144000, 180000, 216000, 234000, 259200, 288000, 324000, 360000, 432000, 468000, 504000, 540000, 604800, 648000, 684000, 720000, 756000, 792000, 828000, 864000, 900000, 936000, 972000, 1008000, 1044000, 1080000, 1116000, 1152000, 1188000, 1224000]
y = [0.0, 0.019230769, 0.016666667, 0.013333333, 0.010714286, 0.009444444, 0.007222222, 0.006, 0.005150000, 0.00385, 0.003, 0.001444444, 0.001055556, 0.000722222, 0.000833333, 0.001111111, 0.002166667, 0.003472222, 0.005285714, 0.006756757, 0.007142857, 0.007272727, 0.006944444, 0.006250000, 0.005277778, 0.004027778, 0.003281250, 0.0025, 0.001875, 0.001111111, 0.000833333, 0.000694444, 0.000416667, 0.000296296, 0.000194444, 0.000111111, 0.000064815, 0.000027778, 0.000013889, 0.000027778, 0.000031746, 0.000031250, 0.000038889, 0.000061111, 0.000076389, 0.000072222, 0.000066667, 0.000055556, 0.000046296, 0.000023148, 0.000019444, 0.000030556, 0.000033333, 0.000041667, 0.000033333, 0.000030556, 0.000033333, 0.000041667, 0.000033333, 0.000030556, 0.000033333, 0.000041667, 0.000033333, 0.000030556, 0.000033333, 0.000041667]
q = DC*interp1d(x,y)(arange(int(Days/dt))*dt)


tiempo_total=0

for k in range(int32(Days/dt)):
	Time = time.time()
	#dt=t/(k+1.)
	t = dt*(k+1.)

	α = K*dt/ (c * ρ * dx**2)
	#α = 5.902213547218499e-07
	print(f'aplha = {α:00.6f}, dt = {dt:02.02f} [s] , t = {t/(24.*3600.):00.04f} [dias] , {k} de {int32(Days/dt)} -> {k*100/int32(Days/dt):0.03f}%, tempMax = {u_k.max():.2f}')
	dias = truncate(t/dia,0)
	horas = truncate((t - dias*dia)/hora,0)
	minutos = truncate((t -dias*dia - horas*hora)/minuto,0)
	titulo ='k = {0:04.0f}, t = {1:02.0f}d {2:02.0f}h {3:02.0f}m'.format(k,dias,horas,minutos)

	#Condiciones de Borde

	BI = 0.*dx + u_k[1,:,:] #Borde izquierdo, gradiente = 0
	BIn= 0.*dy + u_k[:,1,:] #Borde Inferior,gradiente =  0
	BS = Superficie[k]      #Borde Superior
	BD = 0.*dx + u_k[-2,:,:]#Borde derecho, gradiente = 0
	BF = 0.*dz + u_k[:,:,1 ]#Borde frontal, gradiente = 0
	BT = 0.*dz + u_k[:,:,-2]#Borde trasero, gradiente = 0

	u_k[0,:,:] = BI  #Borde izquierdo
	u_k[:,0,:] = BIn #Brode inferior
	u_k[:,-1,:]= BS  #Borde superior
	u_k[-1,:,:]= BD  #Borde derecho
	u_k[:,:,1] = BF  #Borde frontal
	u_k[:,:,-1]= BT  #Borde trasero

	if t>next_t:
		texto = f'BS,{BS}\n'
		#texto = f'BI,BIn,BS,BD,BF,BT,{BI} ,{BI},{BS} ,{BD} ,{BF} ,{BT}\n'

		for i in range(1,Nx):
			for j in range(1,Ny):
				for l in range(1,Nz):
					#Algoritmo de diferencias finitas 2-D para difusion

					#Laplaciano
					#nabla_u_k= (u_k[i-1,j] + u_k[i,j-1] + u_k[i+1,j] + u_k[i,j+1] - 4*u_k[i,j])
					nabla_u_k= (u_k[i-1,j,l] + u_k[i+1,j,l] + u_k[i,j-1,l] + u_k[i,j+1,l] + u_k[i,j,l-1] + u_k[i,j,l+1] - 6*u_k[i,j,l])

					#Foward Euler
					u_km1[i,j,l] = u_k[i,j,l] + α*nabla_u_k + q[k]*dt
					texto += f'{i},{j},{l},{u_km1[i,j,l]}\n'
	else:
		for i in range(1,Nx):
			for j in range(1,Ny):
				for l in range(1,Nz):
					#Algoritmo de diferencias finitas 2-D para difusion

					#Laplaciano
					#nabla_u_k= (u_k[i-1,j] + u_k[i,j-1] + u_k[i+1,j] + u_k[i,j+1] - 4*u_k[i,j])
					nabla_u_k= (u_k[i-1,j,l] + u_k[i+1,j,l] + u_k[i,j-1,l] + u_k[i,j+1,l] + u_k[i,j,l-1] + u_k[i,j,l+1] - 6*u_k[i,j,l])

					#Foward Euler
					u_km1[i,j,l] = u_k[i,j,l] + α*nabla_u_k + q[k]*dt
	
	u_k = u_km1

	#CB de nuevo
	u_k[0,:,:] = BI  #Borde izquierdo
	u_k[:,0,:] = BIn #Brode inferior
	u_k[:,-1,:]= BS  #Borde superior
	u_k[-1,:,:]= BD  #Borde derecho
	u_k[:,:,1] = BF  #Borde frontal
	u_k[:,:,-1]= BT  #Borde trasero


	P1[k] =u_k[int(2*Nx/4),int(2*Ny/4),int(3*Nz/4)]
	P2[k] =u_k[int(2*Nx/4),int(2*Ny/4),int(2*Nz/4)]
	P3[k] =u_k[1 ,int(2*Ny/4),int(2*Nz/4)]
	P4[k] =u_k[int(2*Nx/4),-2,int(2*Nz/4)]
	P5[k] =u_k[int(2*Nx/4),int(2*Ny/4), 1]
	P6[k] =u_k[-2,int(2*Ny/4),int(2*Nz/4)]
	P7[k] =u_k[int(2*Nx/4),int(3*Ny/4),int(2*Nz/4)]
	P8[k] =u_k[int(2*Nx/4),int(1*Ny/4),int(2*Nz/4)]
	P9[k] =u_k[int(1*Nx/4),int(2*Ny/4),int(2*Nz/4)]
	P10[k]=u_k[int(2*Nx/4),int(2*Ny/4),-2]
	P11[k]=u_k[int(2*Nx/4), 1,int(2*Nz/4)]
	P12[k]=u_k[int(2*Nx/4),int(2*Ny/4),int(1*Nz/4)]
	P14[k]=u_k[int(3*Nx/4),int(2*Ny/4),int(2*Nz/4)]

	if t>next_t:
		#figure(1)
		#imshowbien(u_k)
		#title(titulo)
		#savefig("Ejemplo/frame_{0:04.0f}.png".format(framenum))
		archivo=open(f'EjemploTxt/frame_{framenum:04.0f}.txt','w')
		archivo.write(texto)
		archivo.flush()
		archivo.close()
		framenum += 1
		next_t += dnext_t
		#close(1)
		
	Time1= time.time()
	tiempo_total += Time1-Time
	print(f'Tiempo estimado = {datetime.timedelta(seconds=(tiempo_total/(k+1))*(int32(Days/dt)-k))}')

archivo = open('Sensores.txt','w')
archivo.write(
	f'P1 = {P1}'+
	f'P2 = {P2}'+
	f'P3 = {P3}'+
	f'P4 = {P4}'+
	f'P5 = {P5}'+
	f'P6 = {P6}'+
	f'P7 = {P7}'+
	f'P8 = {P8}'+
	f'P9 = {P9}'+
	f'P10 = {P10}'+
	f'P11 = {P11}'+
	f'P12 = {P12}'+
	f'P14 = {P14}'
	)

figure(1)
plot(arange(int32(Days/dt))*dt/dia,Superficie,label='Superficie')
plot(arange(int32(Days/dt))*dt/dia,P1,label='P1')
plot(arange(int32(Days/dt))*dt/dia,P2,label='P2')
plot(arange(int32(Days/dt))*dt/dia,P3,label='P3')
plot(arange(int32(Days/dt))*dt/dia,P4,label='P4')
plot(arange(int32(Days/dt))*dt/dia,P5,label='P5')
plot(arange(int32(Days/dt))*dt/dia,P6,label='P6')
plot(arange(int32(Days/dt))*dt/dia,P7,label='P7')
plot(arange(int32(Days/dt))*dt/dia,P8,label='P8')
plot(arange(int32(Days/dt))*dt/dia,P9,label='P9')
plot(arange(int32(Days/dt))*dt/dia,P10,label='P10')
plot(arange(int32(Days/dt))*dt/dia,P11,label='P11')
plot(arange(int32(Days/dt))*dt/dia,P12,label='P12')
plot(arange(int32(Days/dt))*dt/dia,P14,label='P14')
title('Evolucion de temperatura en puntos')
legend()

xlabel('Tiempo [dias]')
ylabel('Temperatura [°C]')
savefig("EjemploTxt/Grafico.png")
show()