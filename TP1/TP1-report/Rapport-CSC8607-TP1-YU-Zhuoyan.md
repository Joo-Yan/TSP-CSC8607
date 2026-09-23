## Ex1
### 1.c
![[nvidia.png]]
Nvidia L4
### 1.d![[squeue.png]]
![[scancel.png]]
scancel 1580
### 1.e
![[sbatch.png]]
![[logs.png]]
hello-slurm-1583.out
### 1.f
![[sacct.png]]
ReqMem: requested memory
MaxRSS: maximum resident set sizemax, the max of memory that was used.
### 1.g
![[scp.png]]
## Ex2
### 2.a
![[2-a.png]]
### 2.b
![[2-b-cont.png]]
pour vérifier la version exacte de Python: python --version
resultat: Python 3.10.20
le chemin du binaire utilisé: which python
resultat: /mnt/hdd/homes/zyu1/miniforge3/envs/deeplearning/bin/python
### 2.d
la sortie de ce script: 
PyTorch version: 2.10.0
CUDA available: False
Attention, aucun GPU détecté !

Le version n'est pas correspondant.
### 2.f
![[2-f.png]]
tensorboard --version
## Ex 3
### 3.a
![[3-a.png|400]]
Sans biais: 
couche 1 = 3 \* 4 = 12
couche 2 = 4 \* 2 = 8
le nombre total = couche 1 + couche 2 = 20 

Avec biais:
biais 1 = 4
biais 2 = 2
le nombre total = couche 1 + biais 1 + couche 2 + biais 2 = 26

### 3.b
H = ReLU( X · W1^T + b1 )
Y = H · W2^T + b2

Dimensions :
X  : (N, 3)
W1 : (4, 3)
b1 : (1, 4) -> diffusé en (N, 4)
H  : (N, 4)
W2 : (2, 4)
b2 : (1, 2) -> diffusé en (N, 2)
Y  : (N, 2)

### 3.c
![[3-c.jpg|336]]
f = x / y + z = 0.5

∂f/∂x = ∂f/∂q \* ∂q/∂x = 1/y \* 1 = 0.25
∂f/∂y = ∂f/∂q \* ∂q/∂y = -x/y^2 \* 1 = -0.125
∂f/∂z = 1

### 3.d
x' = x - η \* ∂f/∂x = 1.75
y' = y - η \* ∂f/∂y = 4.125
z' = z - η \* ∂f/∂z = -1
f' = f(x', y', z') = -0.576
Oui.

### 3.e
3.e.1 
Parce que c'est plus pratique de calculer le gradient local.
3.e.2 
C'est plus stable et peut exploiter le capacité de parallèle comparant sur un seul exemple à la fois.
C'est plus rapide est requit moins de ressources que sur l’ensemble total des données.

### 3.f
| Tâche                  | Fonction finale (Sortie) | Fonction de perte (Loss)    |
| ---------------------- | ------------------------ | --------------------------- |
| Classification binaire | 1. Sigmoid               | A. Binary Cross-Entropy     |
| Classification multi   | 2. Softmax               | B. Cross-Entropy            |
| Régression pure        | 3. Identité (aucune)     | C. MSE (Mean Squared Error) |

## Ex4 
### 4.a
batch_size: Nombre d'exemples participé à calculation chaque itération, pour faire tradeoff entre efficacité et ressources.
shuffle: Pour faire shuffle. 
différence entre train et test: Pour training, il faut avoir samples différents, pour avoir mieux généralisation. Pour testing, fixer le séquences est plus facile pour reproducer le résultat.
### 4.b
4.b.1 Pour correspondre le format d'input.
4.b.2 Il y a déjà un fonction d'activation dans nn.CrossEntropyLoss
### 4.c
optimizer.zero_grad(): reset the grad to 0
loss.backward(): calculate the new grad
### 4.d
4.d.1 On n'a pas besoin de mettre à jour le grad. Cela prend moins de ressources.
4.d.2 0.1
![[4-d.png]]

### 4.e
ok.
## Ex5
### 5.a
Pour savoir c'est quel expérience et on a faire quoi dans l'expérience.
### 5.d
![[5-d.png]]
![[Pasted image 20260923234144.png]]
![[Pasted image 20260923234242.png]]

![[Pasted image 20260923234921.png]]
Vers 0.97
Loss/train_step est de batch de 32 exemples, qui est comparativement très petit donc a plus de bruit.

### 5.e
![[Pasted image 20260924000010.png]]
![[Pasted image 20260924000847.png]]
![[Pasted image 20260924000919.png]]
5.e.1 Run 2
5.e.2 Training loss continue de diminuer, et validation loss diminue puis remonte.
