#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 11:42:11 2025

@author: rnunes
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig   #bliblioteca p sinais e filtros
import control as ct  # Biblioteca de Controle (para LGR)


# --- Função para simplificar a normalização de a FFT ---
# A saída bruta da FFT precisa ser normalizada para mostrar a amplitude correta
def get_amplitude_spectrum(signal, N, mask):
    yf = np.fft.fft(signal) # Calcula a FFT
    yf_positive = yf[mask] # Pega só a metade positiva
    
    # Normalização:
    # 1. Divide pelo número de pontos N
    # 2. Multiplica por 2 (porque "jogamos fora" a metade negativa)
    yf_mag = np.abs(yf_positive) / N
    yf_mag[1:] = yf_mag[1:] * 2 # Multiplica tudo por 2, exceto o componente DC (índice 0)
    
    return yf_mag







#p um chebyshev tipo 1 (planificado na banda de rejeição) e de 2a ordem o polinomio caracteristico Bn(s) é dado por s² + 1.144s + 1
#onde a função de trasnferencia H(s) em sua forma genérica é dada por

#H(s) = ____________s² + K*Wc²________________
#           s² + 2*(zeta)*Wc*s + Wc²

# zeta->Fator de amortecimento
# Wc -> não é mais Frequencia de corte do filtro pois esta trata-se de onde o mesmo oscila pela ultima vez

#no caso de maneira a atender o polinomio de 2a ordem p o filtro chebyshev temos de especificar tipo do chebyshev, a frequencia de corte e o riple de projeto e ordem do filtro
#como vamos buscar ajuste da solução p implementação Sallen-Key escolhemos ordem N=2
#Frequencia de coret WF=2000; Riple 2db e tipo oscilação na banda passante (para diferenciarmos o resultado de butterworth)

# --- 1. PROJETO TEÓRICO (scipy) ---
N = 2
wcF = 2000.0
rp = 2.0  # Ripple de passagem: 1.0 dB
rs = 40.0 # Atenuação de rejeição: 40 dB

# coeficientes do polinomio do filtro (eliptico)
num, den = sig.ellip(N, rp, rs, wcF, btype='low', analog=True)

print(f"--- Filtro Cauer N={N}, wc={wcF}, rp={rp}dB, rs={rs}dB ---")
print(f"Coeficientes b (Numerador): {num}")
print(f"Coeficientes a (Denominador): {den}")
# Criar FT do filtro para comparar o polinomio característico a solução sallen-key
HF_scipy = sig.lti(num, den)
HF_control = ct.tf(num, den)
print("\nFunção de Transferência H(s):")
print(HF_control)

#a função generica do arranjo biquadratico é dada por:
    # H(s) = __Ghps² + Gbps + Glpw0²___
    #              s² + (w0/Q)s + w02
                 
w0=np.sqrt(den[2])
Q=w0/den[1]

Ghp=num[0]
Gbp=0
Glp=num[2]/w0**2
#_________________Componentes p os polos


# w0/Q=1/R1C1
# p/ R1*C1= Q/(w0)
C1=100e-9 #100nF
R1=Q/(w0*C1)
R1
# #componentes para os zeros
R6=500
R8=R6*num[0]
R8

R3=10e3
C2=100e-9
R4=10e3
R7 = (R1 * R6) / R4
R7
R2 = R8 / (den[2] * R3 * R7 * C1 * C2)
R2
R5 = (R8 / R7) / (num[2] * R3 * C1 * C2)
R5
#R8/(R2R3R7C1C2) = 3.317e+06
#R7=R8/(den[2]*(R2*R3*C1*C2))
R7
#R8/(R3R5C1C2) = num[2]

R5=((R8/R7)*(1/(R3*C2*C1)))/num[2]
R5





#definição FT go filtro
#H1(s) = Num(s)/Den(s)
numF=num
denF=den






# Criando um objeto de Função de Transferência (TF) da bib. 'control'
HF_control = ct.tf(numF, denF)

print("\nFunção de Transferência H(s):")
print(HF_control)


# Criando um objeto de sistema LTI (Linear Time-Invariant) do scipy
HF_scipy=sig.lti((HF_control.num[0][0]), (HF_control.den[0][0]))


# --- 4. ANÁLISE E SIMULAÇÃO ---

# --- a) Diagrama de Bode ---
# bode com faixa de frequencia de 0.01*wc até 100*wc 
w_inicio = wcF*0.01 
w_final = wcF*100

w = np.logspace(np.log10(w_inicio), np.log10(w_final), 500) # 500 pontos logaritmicamente espaçados

w, mod, fase = (sig.bode(HF_scipy, w=w))

plt.figure(figsize=(10,6))
plt.suptitle(f'Filtro Passa-Baixa RC (2ª Ordem) - $\omega_c$={wcF:.2f} rad/s', fontsize=14)

plt.subplot(2, 1, 1)
plt.semilogx(w, mod)
plt.xlabel('Frequencia (rad/s)')
plt.ylabel('Ganho (dB)')
plt.grid(which='both', linestyle='--')
#marcação do ponto em frequencia de corte
plt.axvline(wcF, color='red', linestyle='--')
plt.plot(wcF, -2, 'ro') # 'ro' = red circle
plt.text(wcF * 1.1, -2, f'$\omega_c$={wcF:.2f} rad/s (-2 dB)', color='red')

#fase
plt.subplot(2, 1, 2)
plt.semilogx(w, fase)
plt.xlabel('Frequência (rad/s)')
plt.ylabel('Fase (graus)')
plt.grid(which='both', linestyle='--')
# Linha vertical na frequência de corte
plt.axvline(wcF, color='red', linestyle='--')
# Ponto em -45 graus na frequência de corte
plt.plot(wcF, -90, 'ro')
plt.text(wcF * 1.1, -85, '-90°', color='red')
plt.show()


# --- b) Mapa de Polos e Zeros ---
plt.Figure(figsize=(7,6))
ct.pzmap(HF_control, title='posição dos polos e zeros do filtro')
plt.grid(True)
plt.show()

#teste de simulação dos filtros em ação
#1o teste submetemos o fitros a sinais antes da frequencia de corte, na frequencia de corte e depois da frequencia de corte
f_sinal_A = 60 #Frequência do sinal em Hz
f_sinal_B = 300 #Frequência do sinal em Hz
f_sinal_C = 1000 #Frequência do sinal em Hz

w_sinal_A = f_sinal_A*2*np.pi       #frequencia dos sinais em rad/s  wA, WB, WC
w_sinal_B = f_sinal_B*2*np.pi
w_sinal_C = f_sinal_C*2*np.pi

#criar vetor de tempo de simulação utilizado p todos os sinal de forma que o mesmo garante pelo menos 5 cilcos do sinal mais lento 60Hz
t_final = 5 / f_sinal_A
N_pontos = 10000
T_sim = np.linspace(0, t_final, N_pontos) # 10  mil pontos igualmente espaçãdo entre o tempo de 0 até t_final

#preparação dos parametros de frequencia p avaliação dos conteudos de frequencia das simulações

dt= T_sim[1] - T_sim[0] #periodo de amostragem

# 'fftfreq' calcula "as raias" de frequência para o eixo X
xf = np.fft.fftfreq(N_pontos, dt)
# Nós só queremos a metade positiva das frequências (0 até Nyquist)
# Criamos máscaras (índices) para pegar só a parte positiva
positive_freq_mask = (xf >= 0)
xf_positive = xf[positive_freq_mask]




#criar os vetores senoides de entrada
Amp = 2.5
ofset = 2.5 #2.5  #adequados a entrada do conversor AD de ucontrolador (0 - 5V)

U_A = ofset + Amp*np.sin(w_sinal_A*T_sim) 
U_B = ofset + Amp*np.sin(w_sinal_B*T_sim) 
U_C = ofset + Amp*np.sin(w_sinal_C*T_sim) 

t_out, y_out_A, _ = sig.lsim(HF_scipy, U_A, T_sim)
t_out, y_out_B, _ = sig.lsim(HF_scipy, U_B, T_sim)
t_out, y_out_C, _ = sig.lsim(HF_scipy, U_C, T_sim)

# Calcula as FFTs para o sinal A
fft_U_A_mag = get_amplitude_spectrum((U_A), N_pontos, positive_freq_mask)
fft_y_A_mag = get_amplitude_spectrum(y_out_A, N_pontos, positive_freq_mask)

# Para o Sinal B
yf_U_B_mag = get_amplitude_spectrum(U_B, N_pontos, positive_freq_mask)
yf_y_B_mag = get_amplitude_spectrum(y_out_B, N_pontos, positive_freq_mask)

# Para o Sinal B
yf_U_C_mag = get_amplitude_spectrum(U_C, N_pontos, positive_freq_mask)
yf_y_C_mag = get_amplitude_spectrum(y_out_C, N_pontos, positive_freq_mask)


#Sinal c ruido branco
Amp_ruido = 0.1*Amp #ruido limitado a 10% do sinal senoidal
 # através da regra 3-sigma: 99.7% do ruído estará abaixo do pico.
# Ajustamos o desvio padrão (sigma) do ruído.
desvio_ruido = Amp_ruido / 3.0
#gera o ruido na mesma qtda de pontos dos sinais (N_pontos)
U_noise = desvio_ruido * np.random.randn(N_pontos)
U_A_ruido = U_A + U_noise
#U_A_ruido = U_noise


# 2. Criamos uma janela de Hanning com o mesmo tamanho dos nossos sinais
hann_window = np.hanning(N_pontos)
print(f"Sinal de 60Hz (Amp={Amp}V) + Ruído (Pico max. aprox.={Amp_ruido}V)")

# 3. Simula a resposta do filtro ao sinal ruidoso
t_out, y_out_A_ruido, _ = sig.lsim(HF_scipy, U_A_ruido, T_sim)

# 1. Ainda removemos a média, o que é uma boa prática.
U_A_ruido_AC = U_A_ruido - np.mean(U_A_ruido)
y_out_A_ruido_AC = y_out_A_ruido - np.mean(y_out_A_ruido)

# # 4. Calcula as FFTs para a entrada ruidosa e a saída filtrada
# fft_U_A_ruido = get_amplitude_spectrum((U_A_ruido - np.mean(U_A_ruido)* hann_window), N_pontos, positive_freq_mask) 
# fft_y_A_ruido = get_amplitude_spectrum((y_out_A_ruido - np.mean(y_out_A_ruido)* hann_window), N_pontos, positive_freq_mask) 

fft_U_A_ruido = get_amplitude_spectrum((U_A_ruido_AC * hann_window), N_pontos, positive_freq_mask) 
fft_y_A_ruido = get_amplitude_spectrum((y_out_A_ruido_AC * hann_window), N_pontos, positive_freq_mask)


plt.figure(figsize=(20, 10))
plt.subplot(2,1,1)
plt.plot(t_out, U_A, 'b--', label='entrada 60hz')
plt.plot(t_out, y_out_A, 'r-', label='resposta do filtro' )
plt.title('Resposta do filtro p entrada a 60hz')
plt.xlabel('Tempo(s)')
plt.ylabel('sinal (V)')
plt.legend()
plt.grid(True)
#plt.xlim(0, 3 / f_sinal_A) # Zoom em 3 ciclos para ver melhor

plt.subplot(2, 1, 2)
plt.plot(xf_positive, fft_U_A_mag, 'b--', label='FFT Entrada (U_A)')
plt.plot(xf_positive, fft_y_A_mag, 'r-', label='FFT Saída (y_out_A)')
plt.title('Domínio da Frequência (FFT)')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude (V)')
plt.legend()
plt.grid(True)
plt.xlim(0, f_sinal_C + 200) # Limita o eixo X para ver os picos
plt.show()


# Subplot 1: Domínio do Tempo
plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.plot(t_out, U_B, 'b--', label=f'Entrada (U_B) {f_sinal_B} Hz')
plt.plot(t_out, y_out_B, 'r-', label='Saída (y_out_B)')
plt.title('Domínio do Tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Sinal (V)')
plt.legend()
plt.grid(True)
plt.xlim(0, 5 / f_sinal_B) # Zoom em 5 ciclos

# Subplot 2: Domínio da Frequência (FFT)
#plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 2)
plt.plot(xf_positive, yf_U_B_mag, 'b--', label='FFT Entrada (U_B)')
plt.plot(xf_positive, yf_y_B_mag, 'r-', label='FFT Saída (y_out_B)')
plt.title('Domínio da Frequência (FFT)')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude (V)')
plt.legend()
plt.grid(True)
plt.xlim(0, f_sinal_C + 200) 
plt.show()

# --- GRÁFICO 3: SINAL C (1000 Hz) ---

# Calcula as FFTs para o sinal C
#yf_U_C_mag = get_amplitude_spectrum(U_C, N_pontos)
#yf_y_C_mag = get_amplitude_spectrum(y_out_C, N_pontos)

plt.figure(figsize=(14, 10))
plt.suptitle(f'Análise do Sinal C: {f_sinal_C} Hz (Banda de Rejeição)', fontsize=16)

# Subplot 1: Domínio do Tempo
plt.subplot(2, 1, 1)
plt.plot(t_out, U_C, 'b--', label=f'Entrada (U_C) {f_sinal_C} Hz')
plt.plot(t_out, y_out_C, 'r-', label='Saída (y_out_C)')
plt.title('Domínio do Tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Sinal (V)')
plt.legend()
plt.grid(True)
plt.xlim(0, 10 / f_sinal_C) # Zoom em 10 ciclos

# Subplot 2: Domínio da Frequência (FFT)
plt.subplot(2, 1, 2)
plt.plot(xf_positive, yf_U_C_mag, 'b--', label='FFT Entrada (U_C)')
plt.plot(xf_positive, yf_y_C_mag, 'r-', label='FFT Saída (y_out_C)')
plt.title('Domínio da Frequência (FFT)')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude (V)')
plt.legend()
plt.grid(True)
plt.xlim(0, f_sinal_C + 200) 
plt.show()


# Subplot 1: Domínio do Tempo
plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
plt.plot(t_out, U_A_ruido, 'b--', label=f'Entrada (U_A + Ruído)')
plt.plot(t_out, y_out_A_ruido, 'r-', label='Saída Filtrada (y_out_A_noisy)', linewidth=2)
plt.title('Domínio do Tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Sinal (V)')
plt.legend()
plt.grid(True)
plt.xlim(0, 3 / f_sinal_A) # Zoom em 3 ciclos para ver melhor

# Subplot 2: Domínio da Frequência (FFT)
plt.subplot(3, 1, 2)
plt.plot(xf_positive, fft_U_A_ruido, 'b--', label='FFT Entrada (U_A + Ruído)')
plt.plot(xf_positive, fft_y_A_ruido, 'r-', label='FFT Saída Filtrada')

# Linha vertical para mostrar a frequência de corte do filtro
fc_hz = wcF / (2 * np.pi)
plt.axvline(fc_hz, color='green', linestyle=':', label=f'Corte do Filtro ($f_c$={fc_hz:.2f} Hz)')

plt.title('Domínio da Frequência (FFT)')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude (V)')
plt.legend()
plt.grid(True)
plt.xlim(0, f_sinal_C + 200) # Limita o eixo X para ver os picos

# --- Subplot 3: Domínio da Frequência (FFT - Zoom no Piso de Ruído) ---

plt.subplot(3, 1, 3)
plt.plot(xf_positive, fft_U_A_ruido, 'b--', label='FFT Entrada (U_A + Ruído)')
plt.plot(xf_positive, fft_y_A_ruido, 'r-', label='FFT Saída Filtrada')

# Linha vertical de corte
plt.axvline(fc_hz, color='green', linestyle=':', label=f'Corte do Filtro ($f_c$={fc_hz:.2f} Hz)')

plt.title('Domínio da Frequência (FFT) - Zoom no Piso de Ruído') # Título novo
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude (V)')
plt.legend()
plt.grid(True)
plt.xlim(0, f_sinal_C + 200) #

# --- AQUI ESTÁ O AJUSTE ---
# (a FFT espalha essa energia).
plt.ylim(0, 0.01) 

plt.show()

plt.show()

