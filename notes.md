Note sur l'implémentation de l'agent DQN pour la tâche 1

Dans le cadre de la tâche 1, j'ai implémenté un agent DQN (Deep Q-Network) visant à résoudre un problème d'apprentissage par renforcement. Le modèle a été testé sur plusieurs épisodes, et les résultats sont enregistrés à chaque étape pour analyser l'évolution de l'agent au fur et à mesure de son apprentissage.

Résultats obtenus :
L'agent commence par obtenir des récompenses relativement faibles, mais ces dernières augmentent progressivement à mesure que l'agent explore et apprend les meilleures actions à entreprendre.

La récompense totale par épisode varie au fil du temps, avec quelques épisodes où des récompenses élevées ont été obtenues, ce qui montre que l'agent commence à trouver des stratégies plus efficaces.

L'epsilon (paramètre de l'exploration) diminue lentement au cours des épisodes, ce qui incite l'agent à exploiter de plus en plus ses connaissances acquises au lieu d'explorer de nouvelles actions de manière aléatoire.

Points à améliorer :
L'agent montre parfois des fluctuations importantes dans les récompenses, ce qui indique que certaines parties de l'environnement ou de la politique de l'agent nécessitent encore des ajustements.

Il serait intéressant d'explorer d'autres techniques d'optimisation des hyperparamètres, telles que la modification de la fonction de récompense ou l'ajustement du taux d'apprentissage.


commentaire de la courbe : 

La courbe montre une forte variabilité des récompenses brutes au fil des épisodes, ce qui est typique des débuts d’entraînement en apprentissage par renforcement. Malgré cette variance, on observe une tendance générale à la hausse mise en évidence par la moyenne mobile (fenêtre de 10 épisodes). Cette progression suggère que l’agent apprend progressivement une politique plus efficace.

À partir d’environ l’épisode 150, la moyenne mobile commence à s’élever plus nettement, ce qui peut indiquer un point d’inflexion dans l’apprentissage. Vers la fin de l’entraînement, les récompenses deviennent plus fréquentes et de plus grande amplitude, ce qui reflète une amélioration de la performance de l’agent.

En résumé, la courbe valide que l’agent converge progressivement vers un comportement optimal, bien que quelques pics de variance subsistent, ce qui est courant dans des environnements stochastiques.