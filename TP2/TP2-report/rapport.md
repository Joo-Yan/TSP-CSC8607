## 1.a
![[Pasted image 20261001195137.png]]
### 1.a.1 
Le calcul de la moyenne et de l'écart-type a inclus des échantillons des ensembles de validation et de test, ce qui a provoqué une fuite de données.
### 1.a.2 
torch.utils.data.IterableDataset
## 2.a
l1_lambda = 1e-4, l2_lambda = 1e-3
![[Pasted image 20261001201420.png]]
l1_lambda = 0.1, l2_lambda = 0
![[Pasted image 20261001201615.png]]
### 2.a.1 
La perte ne diminue pas, et la validation équivaut à peu près à une prédiction aléatoire.
Le poids du terme de pénalité est trop élevé, ce qui pousse l'optimiseur à réduire ce terme de pénalité plutôt qu'à suivre l'apprentissage, c'est underfitting.
### 2.a.2
weight_decay
### 2.a.3
La régularisation L1 a tendance à ramener de nombreux poids à zéro, ce qui conduit à la parcimonie. La régularisation L2 a tendance à réduire les poids sans toutefois les amener à zéro.
### 3.a
### 3.a.1
![[Pasted image 20261001204445.png]]
### 3.a.2
Adam
### 3.a.3
La baisse est plus rapide.
## 4.a
![[Pasted image 20261001213921.png]]
### 4.a.1
precision = TP / (TP + FP)
recall = TP / (TP + FN)
### 4.a.2
Rappel.
Nous voulons identifier tous les patients malades, car le coût d'un faux négatif est bien plus élevé que celui d'un faux positif.
### 4.a.3
Il ne dépend pas d'un seuil pour évaluer la capacité des modèles à distinguer le data.
