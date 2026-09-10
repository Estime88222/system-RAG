"""
evaluation/build_test_set_template.py

Script à exécuter une seule fois pour générer le brouillon de jeu de test
(evaluation/test_set_template.json) à partir d'une liste de questions.

Ce script interroge ton vrai retriever (Chroma) pour chaque question et
liste les chunks candidats trouvés, avec leur ID (hash de contenu) et un
aperçu du texte — pour que tu n'aies plus qu'à cocher manuellement lesquels
sont réellement pertinents.

Usage :
    python -m evaluation.build_test_set_template
"""

from evaluation.retrieval_eval import generate_test_set_template

# Questions de test basées sur TARA (transport supervision platform).
# Ajuste/complète cette liste selon ce qui existe réellement dans le produit.
QUESTIONS = [
    "Qu'est-ce que TARA et à quoi ça sert ?",
    "Quels modes de transport TARA permet-il de superviser ?",
    "Comment TARA assure-t-il le suivi en temps réel des véhicules ?",
    "Quelle est la couverture géographique ou le corridor pilote de TARA ?",
    "Comment créer ou enregistrer une nouvelle unité (véhicule) dans TARA ?",
    "Que se passe-t-il si la connexion internet est perdue pendant le suivi d'un véhicule ?",
    "Quelles sont les fonctionnalités principales du tableau de bord TARA ?",
    "Comment contacter le support ou demander une démonstration de TARA ?",
    "TARA propose-t-il des alertes en cas d'incident ou de risque sur un trajet ?",
    "Quelle est la vision ou la mission de TARA en tant que plateforme ?",
]

if __name__ == "__main__":
    generate_test_set_template(
        questions=QUESTIONS,
        k=10,
        output_path="evaluation/test_set_template.json",
    )