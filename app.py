"""
main.py — Point d'entrée de l'application.

Responsabilité unique : importer l'app Gradio et la lancer.
Toute la logique métier est dans app/backend.py
Toute l'interface est dans app/ui.py
"""
from app.ui import build_app

if __name__ == "__main__":
    build_app().launch(debug=True)
