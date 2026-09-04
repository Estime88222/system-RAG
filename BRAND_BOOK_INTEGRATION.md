# Améliorations Respect du Ton du Brand-Book

## 📋 Résumé des changements

### 1. **Correction du bug critique du filtre brand-book** ⚠️

- **Fichier**: [src/retrieval/search.py](src/retrieval/search.py#L66)
- **Problème**: Le filtre utilisait `"doc-type"` (tiret) au lieu de `"doc_type"` (underscore)
- **Impact**: Le contexte du brand-book n'était **jamais** récupéré
- **Solution**: Correction du filtre et ajout de gestion d'erreur

### 2. **Renforcement des directives de ton de marque**

- **Fichier**: [src/generation/prompts.py](src/generation/prompts.py)
- **Changement**: Ajout d'une section `DIRECTIVES DE TON DE MARQUE (TARA)` au SYSTEM_PROMPT
- **Instructions explicites** que le LLM doit:
- Imiter le ton du brand-book (formalité, expressions privilégiées)
- Respecter les valeurs de la marque (principes éthiques, priorités)
- Utiliser la même voix narrative et termes clés
- Assurer l'harmonie avec l'identité de TARA

---

## 🔄 Pipeline complet (après corrections)

```Question utilisateur
    ↓
[1] Recherche de chunks pertinents (general documents)
    ↓
[2] ✓ CORRECTION: Récupération du contexte brand-book
    → Utilise maintenant le filtre: {"doc_type": "BRAND_BOOK_TARA_2026-08-17_v2.5.pdf"}
    → Extrait les guidelines de ton/style/valeurs
    ↓
[3] Construction du contexte complet
    → Brand-book context EN PREMIER (pour influencer le ton)
    → Suivi du contexte des documents pertinents
    ↓
[4] Build du prompt système + user
    → SYSTEM_PROMPT inclut les directives de ton TARA
    ↓
[5] Appel LLM (Groq)
    → Température 0.2 (précision + stabilité du ton)
    → Génération ancrée dans le contexte ET respectueuse du brand
    ↓
Réponse finale avec TON DE MARQUE TARA
```

---

## ✅ Vérification que le brand-book est bien utilisé

Après avoir lancé `src/ingestion/ingetion_branding.py` une fois, voici comment vérifier que tout fonctionne:

```python
# Dans Python, vous pouvez tester:
from src.retrieval.search import get_branding_context

# Ceci doit retourner le contexte du brand-book
branding_ctx = get_branding_context(k=2)
print(f"✓ Contexte brand-book ({len(branding_ctx)} chars):", branding_ctx[:200])

# Puis, une question devrait inclure ce contexte dans sa génération
from src.main import ask
reponse = ask("Comment se présente la marque TARA?")
```

---

## 🎯 Résultats attendus

Grâce à ces corrections, vos réponses générées par le LLM vont maintenant:

1. **Inclure automatiquement** les guidelines du brand-book
2. **Respecter le ton** défini par TARA (formalité, voix, expressions)
3. **Prioriser les valeurs** de la marque
4. **Utiliser la terminologie** officielle de TARA
5. **Rester ancrées** dans les documents tout en ayant une identité de marque cohérente

---

## 📝 Notes

- Le brand-book doit être indexé une seule fois : `python src/ingestion/ingetion_branding.py`
- Après cela, chaque question inclura automatiquement le contexte brand-book
- Si vous mettez à jour le brand-book PDF, réindexez-le
- La température du LLM (0.2) assure la stabilité du ton tout en respectant la précision factuelle
