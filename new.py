#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:53:56 2024

@author: jiangruihua
"""

import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy import interpolate
from scipy.ndimage import laplace
from numpy import array,sin,pi,ones_like,meshgrid,linspace,abs,zeros,zeros_like
from numpy import where,pad,arange,rint,load,argwhere

from matplotlib import cm


# Paramètres globaux
Lx,Ly = 4, 4  # Largeur, longueur (m)
N_point = 201  # Nombre de points minimum selon x ou y
c = 1  # Vitesse de propagation des ondes dans le milieu (m/s)
T = 6  # Temps final de simulation (s)
dt = 0.003 # Pas de temps
coeff_amorti = 20  # Coefficient d'amortissement
L_absorb = 1 # Épaisseur de la couche d'absorption
durée_RT = 3  # Durée en temps réel, utilisé seulement lors de la lecture des données externes, sinon identique à début_RT
début_RT = 3  # Utilisé seulement lors de l'auto-simulation, sinon 0 (RT dès le début)

c_varie = False  # True pour activer la variation de c

n = 0  # Compteur d'itérations
N_cache = 10  # Nombre d'étapes de temps pour l'enregistrement des ondes

lire_données_externes = False

rendu_temps_réel = True
fps = 30 # Images par seconde pour le rendu
vitesse_rendu = 10 # 1 seconde du temps réel correspond à combien seconde du temps de rendu

# Fonction pour calculer le laplacien en utilisant SciPy
def laplacien_sp(u_t, dl):
    return laplace(u_t) / dl**2



# Espacement de la grille
dl = Lx / (N_point - 1)
Nx=Ny = int(Lx / dl) + 1 

# Créer la grille de maillage pour X et Y

X,Y = np.meshgrid(linspace(0, Lx, Nx),linspace(0, Ly, Ny))

Nt = int(T / dt) + 1 
T = (Nt - 1) * dt

# Nombre de points dans la couche d'absorption
N_absorb = int(L_absorb / dl)

# Définir la durée en temps réel et le début en temps réel
if lire_données_externes:
    début_RT = 0
else:
    durée_RT = début_RT

N_RT = int(durée_RT / dt)
n_début_RT = int(début_RT / dt)

# Paramètres de simulation pour l'affichage
paramètres = f"c={c}, T={T}, Nt={Nt}, N_point={N_point}, Lx={Lx}, Ly={Ly}, α={coeff_amorti}, n_absorb={N_absorb}"


# Coordonnées des sources ( modifier manuellement)
coordonnées_sources = array(
    [
        [2, 2]
    ]
)
indices_sources = rint(coordonnées_sources / dl).astype(int)
i_sources, j_sources = indices_sources.T

u = zeros(
    [N_cache, Nx + 2 * N_absorb, Ny + 2 * N_absorb]
)
u_sim = u[
     :, N_absorb : -N_absorb, N_absorb : -N_absorb
]
u_1erederivee = zeros_like(u)
if c_varie:
    c = (
         c * ones_like(u[0])
         + 0.2 * sin(2 * pi * arange(u.shape[1]) * dl / 2)[:, None]
        )
    
α = pad(
    zeros_like(u_sim[0]),
    N_absorb,
    "linear_ramp",
    end_values=coeff_amorti,
)

# Fonctions pour créer des formes de capteur 
def créer_coeur(largeur=0.01, a=2, b=1.2, taille=1.3):
    fonction_coeur = ((X - a) / 1.3) ** 2 + (
        (Y - b) - (abs(X - a) / 1.3) ** (2 / 3)
    ) ** 2
    return (fonction_coeur <= taille + largeur) & (fonction_coeur >= taille - largeur)


def créer_cercle(largeur=0.005, a=2, b=1.5, taille=1.4):
    fonction_cercle = ((X - a) ** 2 + (Y - b) ** 2) ** 0.5
    return (fonction_cercle <= taille + largeur) & (fonction_cercle >= taille - largeur)


def créer_carré(largeur=0.005, a=2, b=1.5, taille=1.4):
    fonction_carré = abs(X - a) + abs(Y - b)
    return (fonction_carré <= taille + largeur) & (fonction_carré >= taille - largeur)


def créer_triangle(largeur=0.02, a=0.6, b=0, taille=4):
    fonction_triangle = (
        (taille - (X - a) - (Y - b)) * (X - a) * (Y - b)
    )
    return (fonction_triangle <= 1 + largeur) & (fonction_triangle >= 1 - largeur)

i_capteurs = None
j_capteurs = None
données_capteurs = None
forme_capteurs = None


forme_capteurs = créer_triangle()
i_capteurs, j_capteurs = where(forme_capteurs)
données_capteurs = zeros((len(i_capteurs), N_RT)) # Initialise un tableau pour les données des capteurs, de taille nombre de capteurs par nombre de pas de temps

# Fonction pour obtenir les valeurs de u aux emplacements des capteurs
def u_cap(n): # 这个函数用于获取在传感器位置上的u的值
    res = zeros_like(X) # Une grille qui ressemble à X
    res[i_capteurs, j_capteurs] = données_capteurs[:, n]
    return res

# Fonction pour calculer la deuxième dérivée temporelle de u
def u_2emederivee(n):
    C = c**2 * laplacien_sp(u[n % N_cache], dl)
    A = (
        -α
        * (u[n % N_cache] - u[n % N_cache - 1])
        / dt
    )

    T_source = 0.05
    S = zeros_like(X)
    if n * dt < T_source and not lire_données_externes:
        S[i_sources, j_sources] = 20001 * sin(pi * n * dt / T_source)
    S = pad(
        S,
        N_absorb,
        "constant",
        constant_values=0,
    )

    if n_début_RT <= n < n_début_RT + N_RT:
        T = (
            -130
            * where(
                forme_capteurs,
                (
                    u_cap(-(n - n_début_RT) - 1)
                    - u_cap(-(n - n_début_RT))
                ),
                0,
            )
            / dt
        )

        T = pad(
            T,
            N_absorb,
            "constant",
            constant_values=0,
        )

    else:
        T = 0
    return C + A + T + S

# Fonction pour configurer le tracé
def configurer_plot():
    global fig, ax, u_img, cap_img, N_frame

    N_frame = int(fps * T / vitesse_rendu)

    plt.ioff()
    fig, ax = plt.subplots(figsize=(7, 5) if rendu_temps_réel else (16, 9))
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    u_max = 0.1

    u_img = ax.imshow(
        [[]],
        cmap="YlGnBu", # Pastel1, tab20c & black ; YlGnBu,BuPu & white ; cm.coolwarm
        vmin=-u_max,
        vmax=u_max,
        extent=[0, Lx, 0, Ly],
        zorder=0,
        interpolation="none" if rendu_temps_réel else "antialiased",
    )

    cap_img = ax.scatter([], [], c="lightgrey", s=1, zorder=5)

# Fonction pour simuler la propagation des ondes
def simuler(n_rendu):
    global n, t0

    ax.set_title(f"t={n_rendu*dt:.5f}")

    while n <= n_rendu:
        if n == n_début_RT and not lire_données_externes:
            u[:] = 0

        if n >= 2:
            u[n % N_cache] = (
                2 * u[(n - 1) % N_cache]
                - u[(n - 2) % N_cache]
                + dt**2 * u_2emederivee(n - 1)
            )

        if n < N_RT and not lire_données_externes:
            données_capteurs[:, n] = u_sim[
                n % N_cache, i_capteurs, j_capteurs
            ]

        n += 1

    u_img.set_data(
        u_sim[n_rendu % N_cache, ::1, ::-1].T
    )

    cap_img.set_offsets(argwhere(forme_capteurs) * dl)
    if not n_rendu % 10:
        t1 = time.time()
        print(
            f"\r{n_rendu}/{Nt} le temps reste estimé : {(Nt-n_rendu)*(t1-t0)/10:.2f} s",
            end="",
            flush=True,
        )
        t0 = t1
    return u_img, cap_img


def rendre():
    global t0
    t0 = time.time()

    ns_rendu = [
        int(vitesse_rendu / dt / fps * n_frame) 
        for n_frame in range(N_frame)
    ] + [Nt - 1]
    anim = animation.FuncAnimation(
        fig,
        simuler,
        frames=ns_rendu,
        interval=1,
        blit=True,
        repeat=False,
    )
    if rendu_temps_réel:
        plt.show()
    else:
        anim.save(
            "./wave/" + paramètres + ".mp4", writer="ffmpeg", fps=fps
        )
    print("\nterminé")


if "save" in sys.argv:
    rendu_temps_réel = False

configurer_plot()
rendre()
