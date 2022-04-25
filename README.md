Prérequis
Django 3.2.6 ou plus recent

Le dossier 'traitement_app' contient l'applicatrion qui sera deployé avec et qu permettra de mettre en oeuvre la détection de voitures et de maisons à travers nos modèles crées.

Le dossier 'traitement_app/exported_models' contient les modèles crées
Le dossier 'traitement_app/annotations' contient les annotations

Le fichier 'views' contient le code source de la mise en oeuvre du déploiement de nos modèles.

La fonction de traitement qui utilise le modèle MobileNet est nommée MobileNet et celle qui utilise le modèle FasterCNN ResNet est nommée ResNet101

# Lancement

Ouvrir le dossier 'Projet_Big_Data_ML' dans un terminal
Exécuter la commande 'python manage.py runserver'
Ourvir le navigateur et aller à l'adresse 'http://127.0.0.1:8000/'

Pour charger une image cliquer sur 'Charger une image', une fenêtre s'ouvre et vous choisissez l'image voulue.
Après le chargement de l'image, pour appliquer un modèle de détection d'objet cliquer sur 'Détection d'objets' puis choisir le modèle correspondant.



Le taux d'apprentissage est un hyperparamètre qui contrôle dans quelle mesure le modèle doit être modifié en réponse à l'erreur estimée chaque fois que les poids du modèle sont mis à jour. Le choix du taux d'apprentissage est un défi, car une valeur trop petite peut entraîner un long processus d'apprentissage qui peut rester bloqué, tandis qu'une valeur trop grande peut entraîner l'apprentissage trop rapide d'un ensemble de poids sous-optimaux ou un processus d'apprentissage instable.

Un bon taux d'apprentissage donne généralement de bons résultats dans toutes les étapes de formation (par exemple, si nous rencontrons un taux d'apprentissage qui donne de bons résultats à une étape de formation particulière, nous pouvons l'utiliser pour toute la formation).

Nous avons augmenté le taux d'apprentissage assez rapidement pour qu'un bon taux soit essayé avant que le réseau ne trouve que ses paramètres convergent vers un minimum de perte (s'il est arrivé jusque là avant que le taux d'apprentissage n'augmente suffisamment pour provoquer une divergence).