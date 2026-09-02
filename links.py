import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests

# 1. METTEZ ICI L'URL DU SITE QUE VOUS VOULEZ SCRAPER
url_depart = "https://taramoney.com/"  # À remplacer par votre cible

visites = set()

# Simulation d'un navigateur pour éviter d'être bloqué
INTELLIGENT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def scraper_liens(url):
  if url in visites:
    return
  visites.add(url)

  print(f"[Analyse en cours] : {url}")

  try:
    reponse = requests.get(url, headers=INTELLIGENT_HEADERS, timeout=10)

    # Si le site renvoie une erreur (403 bloqué, 404 introuvable, etc.)
    if reponse.status_code != 200:
      print(f"[Échec] Code erreur {reponse.status_code} sur {url}")
      return

    soup = BeautifulSoup(reponse.text, "html.parser")
    liens_trouves = soup.find_all("a", href=True)

    print(f"--> {len(liens_trouves)} liens bruts trouvés sur la page.")

    for balise_a in liens_trouves:
      lien = str(balise_a["href"])
      lien_absolu = urljoin(url, lien)

      # Nettoyer les ancres de page (ex: #contact, #about)
      lien_absolu = lien_absolu.split("#")[0]

      # Vérification : Reste-t-on sur le même domaine ?
      domaine_depart = urlparse(url_depart).netloc.replace("www.", "")
      domaine_actuel = urlparse(lien_absolu).netloc.replace("www.", "")

      if domaine_depart == domaine_actuel:
        if lien_absolu not in visites:
          print(f"[TROUVÉ] Sous-lien valide : {lien_absolu}")

          # Pause de 1 seconde pour ne pas surcharger le site web
          time.sleep(1)

          # Lancement de l'analyse du sous-lien (Récursivité)
          scraper_liens(lien_absolu)

  except Exception as e:
    print(f"[Erreur] Impossible de lire {url} : {e}")


# Lancement du script
print(f"Début du crawling sur : {url_depart}")
scraper_liens(url_depart)
print("Fin du crawling.")
