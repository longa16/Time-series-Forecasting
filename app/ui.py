"""
ui.py — Interface Gradio : styles CSS, layout, callbacks, lancement de l'app.
"""
from __future__ import annotations
from typing import Any
import gradio as gr
from prophet import Prophet

from app.backend import (
    build_future_table,
    compute_kpis,
    evaluate_model,
    prepare_forecast_data,
    run_prophet,
)

# ---------------------------------------------------------------------------
# Styles CSS
# ---------------------------------------------------------------------------

# Styles réutilisables définis comme constantes pour être inlinés dans le HTML
# (évite les conflits de spécificité avec le thème Gradio)
_S = {
    "info":    "background:#eaf4fb; border-radius:8px; padding:12px 16px; margin:10px 0; font-size:.92rem; color:#1a3d5c;",
    "warning": "background:#fef9e7; border-radius:8px; padding:12px 16px; margin:10px 0; font-size:.92rem; color:#7d4e00;",
    "success": "background:#eafaf1; border-radius:8px; padding:12px 16px; margin:10px 0; font-size:.92rem; color:#1d6637;",
    "kpi_grid": "display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin-bottom:18px;",
    "kpi_card": "background:#f4f6f7; border-radius:12px; padding:14px 18px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,.07);",
    "kpi_value": "font-size:1.6rem; font-weight:700; color:#1a5276; display:block;",
    "kpi_label": "font-size:.8rem; color:#555; margin-top:4px; display:block;",
    "badge_good": "display:inline-block; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; background:#d5f5e3; color:#1e8449;",
    "badge_warn": "display:inline-block; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; background:#fdebd0; color:#d35400;",
    "badge_bad":  "display:inline-block; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; background:#fadbd8; color:#922b21;",
}

CUSTOM_CSS = """
.gradio-container { font-family: 'Inter', ui-sans-serif, system-ui; }

/* Force les éléments inline à hériter la couleur de leur conteneur stylé en inline.
   Gradio dark-theme injecte color:white sur strong/p/small/code ce qui les rend
   invisibles sur nos fonds colorés clairs. */
.gradio-container [style*="background:#eaf4fb"] strong,
.gradio-container [style*="background:#eaf4fb"] small,
.gradio-container [style*="background:#eaf4fb"] code,
.gradio-container [style*="background:#fef9e7"] strong,
.gradio-container [style*="background:#fef9e7"] small,
.gradio-container [style*="background:#fef9e7"] code,
.gradio-container [style*="background:#eafaf1"] strong,
.gradio-container [style*="background:#eafaf1"] small,
.gradio-container [style*="background:#eafaf1"] code,
.gradio-container [style*="background:#f4f6f7"] strong,
.gradio-container [style*="background:#f4f6f7"] small,
.gradio-container [style*="background:#f4f6f7"] code {
    color: inherit !important;
}
"""

# ---------------------------------------------------------------------------
# Rendu HTML des KPIs et métriques
# ---------------------------------------------------------------------------

def _render_kpis(kpis: dict, forecast_days: int) -> str:
    trend_arrow = "▲" if kpis["trend_pct"] >= 0 else "▼"
    trend_color = "#1e8449" if kpis["trend_pct"] >= 0 else "#922b21"
    b = 'style="font-weight:700; color:inherit"'  # <b> avec couleur forcée héritée
    return f"""
    <div style="{_S['kpi_grid']}">
      <div style="{_S['kpi_card']}">
        <span style="{_S['kpi_value']}">{kpis['total_sales']:,.0f}</span>
        <span style="{_S['kpi_label']}">Ventes totales</span>
      </div>
      <div style="{_S['kpi_card']}">
        <span style="{_S['kpi_value']}">{kpis['avg_daily']:,.1f}</span>
        <span style="{_S['kpi_label']}">Moyenne / jour</span>
      </div>
      <div style="{_S['kpi_card']}">
        <span style="{_S['kpi_value']} color:{trend_color};">{trend_arrow} {abs(kpis['trend_pct']):.1f}%</span>
        <span style="{_S['kpi_label']}">Tendance récente (30j)</span>
      </div>
      <div style="{_S['kpi_card']}">
        <span style="{_S['kpi_value']}">{kpis['max_day_sales']:,.0f}</span>
        <span style="{_S['kpi_label']}">Record journalier — {kpis['max_day_date']}</span>
      </div>
    </div>
    <div style="{_S['info']}">
      <b {b}>Période analysée :</b>
      du <b {b}>{kpis['date_min']}</b> au <b {b}>{kpis['date_max']}</b>
    </div>
    """


def _render_metrics(metrics: dict, nb_days: int) -> str:
    b = 'style="font-weight:700; color:inherit"'
    if metrics["mae"] is None:
        return f'<div style="{_S["warning"]}">Pas assez de données pour évaluer le modèle.</div>'

    mape_val = metrics["mape"]
    if mape_val < 10:
        badge = f'<span style="{_S["badge_good"]}">Excellente précision</span>'
        color = "#1e8449"
    elif mape_val < 20:
        badge = f'<span style="{_S["badge_warn"]}">Précision correcte</span>'
        color = "#d35400"
    else:
        badge = f'<span style="{_S["badge_bad"]}">Précision à améliorer</span>'
        color = "#922b21"

    return f"""
    <div style="{_S['success']}">
      <b {b}>Évaluation du modèle (sur les {min(90, nb_days)} derniers jours)</b><br/>
      {badge}<br/>
      &bull; <b {b}>MAE</b> (Erreur absolue moyenne) :
        <b {b}>{metrics['mae']:,.1f} unités</b>
        — le modèle se trompe en moyenne de {metrics['mae']:,.1f} ventes/jour.<br/>
      &bull; <b {b}>MAPE</b> (Erreur relative) :
        <b style="font-weight:700; color:{color};">{mape_val:.1f}%</b>
        — le modèle explique <b {b}>{100 - mape_val:.1f}%</b> de la demande réelle.
    </div>
    """


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _build_forecast_plot(model: Prophet, forecast: Any) -> Any:
    """Graphique principal Prophet avec légende métier."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig = model.plot(forecast)
    ax = fig.axes[0]

    # — Titre —
    ax.set_title(
        "Prévision des ventes agrégées (tous magasins / tous produits)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Ventes journalières (unités)", fontsize=11)

    # — Légende manuelle (les artistes Prophet ne sont pas nommés) —
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#222",
               markersize=6, label="Ventes réelles observées"),
        Line2D([0], [0], color="#0072B2", linewidth=2,
               label="Prévision Prophet"),
        Patch(facecolor="#AEC6E8", alpha=0.5,
              label="Intervalle de confiance 80 %"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=9,
        framealpha=0.85,
        title="Légende",
        title_fontsize=9,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


_COMPONENT_LABELS = {
    "trend":  ("Tendance",              "Ventes (unités)",   "#1a5276"),
    "weekly": ("Saisonnalité hebdo",    "Effet relatif",     "#117a65"),
    "yearly": ("Saisonnalité annuelle", "Effet relatif",     "#884ea0"),
    "additive_terms": ("Composantes additives", "Effet",     "#cb4335"),
}

_WEEKDAY_FR = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

def _build_components_plot(model: Prophet, forecast: Any) -> Any:
    """Décomposition Prophet avec titres et légendes en français."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig = model.plot_components(forecast)
    axes = fig.axes

    for ax in axes:
        # Récupère le titre existant généré par Prophet (texte brut)
        raw_title = ax.get_title().lower().strip()

        match = next(
            ((fr_title, y_label, color)
             for key, (fr_title, y_label, color) in _COMPONENT_LABELS.items()
             if key in raw_title),
            None,
        )

        if match:
            fr_title, y_label, color = match
            ax.set_title(fr_title, fontsize=11, fontweight="bold", color=color)
            ax.set_ylabel(y_label, fontsize=10)

            # Colorier la courbe principale
            for line in ax.get_lines():
                line.set_color(color)
                line.set_linewidth(2)

            # Colorier la zone d'incertitude
            for coll in ax.collections:
                coll.set_facecolor(color)
                coll.set_alpha(0.15)

        # Renommer l'axe X pour la saisonnalité hebdo (jours 0-6 → Lun-Dim)
        if "weekly" in raw_title:
            try:
                ticks = ax.get_xticks()
                # Prophet place les ticks en secondes depuis epoch pour les jours
                # On force les labels en français
                ax.set_xticklabels(
                    _WEEKDAY_FR[: len(ax.get_xticklabels())]
                    if len(ax.get_xticklabels()) <= 7
                    else [l.get_text() for l in ax.get_xticklabels()]
                )
            except Exception:
                pass

        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_xlabel("")

    fig.suptitle(
        "Décomposition du modèle — Tendance & Saisonnalités",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Callback principal
# ---------------------------------------------------------------------------

def forecast_sales(file=None, forecast_days: int = 90):
    """Point d'entrée Gradio : orchestre le backend et retourne tous les outputs."""
    df = prepare_forecast_data(file)
    kpis = compute_kpis(df)
    model, forecast = run_prophet(df, forecast_days)
    metrics = evaluate_model(forecast, df)
    table = build_future_table(forecast, df, forecast_days)

    kpi_html = _render_kpis(kpis, forecast_days)
    metrics_html = _render_metrics(metrics, kpis["nb_days"])
    plot_forecast = _build_forecast_plot(model, forecast)
    plot_components = _build_components_plot(model, forecast)

    return kpi_html, metrics_html, plot_forecast, plot_components, table


# Construction du Blocks Gradio

def _tab_forecast() -> tuple:
    """Construit le contenu de l'onglet Prévision et retourne les composants I/O."""
    with gr.Row():
        file_input = gr.File(
            label=" Importer un fichier CSV",
            file_types=[".csv"],
            type="filepath",
            scale=3,
        )
        forecast_days = gr.Slider(
            minimum=7, maximum=365, value=90, step=1,
            label=" Horizon de prévision en jours",
            info="Combien de jours dans le futur voulez-vous prévoir ?",
            scale=2,
        )

    predict_button = gr.Button("▶ Lancer la prévision", variant="primary", size="lg")

    welcome_msg = gr.HTML(
        value=(
            f'<div style="{_S["info"]} text-align:center; padding:20px;">'
            '<b style="font-weight:700;color:inherit">Bienvenue !</b><br/>'
            'Importez votre fichier CSV et cliquez sur <b style="font-weight:700;color:inherit">▶ Lancer la prévision</b> pour démarrer l\'analyse.'
            '<br/><span style="font-size:.85em;color:inherit;">Aucun fichier ? Le bouton utilisera automatiquement le jeu de données de démonstration <code style="color:inherit">train.csv</code>.</span>'
            '</div>'
        )
    )

    gr.Markdown("### Indicateurs clés")
    kpi_output = gr.HTML()

    gr.Markdown("### Performance du modèle Prophet")
    metrics_output = gr.HTML()

    gr.Markdown("### Graphique de prévision")
    gr.HTML(
        f'<div style="{_S["info"]}">La courbe bleue représente les ventes prévues. '
        "La zone ombrée correspond à l'intervalle de confiance à 80 % "
        "— utile pour dimensionner un stock de sécurité.</div>"
    )
    forecast_plot = gr.Plot(label="Prévision des ventes")

    gr.Markdown("### Décomposition des composantes")
    gr.HTML(
        f'<div style="{_S["info"]}">'
        '<b style="font-weight:700;color:inherit">Tendance :</b> évolution générale de la demande.<br/>'
        '<b style="font-weight:700;color:inherit">Saisonnalité hebdomadaire :</b> quels jours génèrent le plus de ventes ?<br/>'
        '<b style="font-weight:700;color:inherit">Saisonnalité annuelle :</b> quelles périodes sont les plus fortes (fêtes, soldes…) ?'
        '</div>'
    )
    components_plot = gr.Plot(label="Composantes du modèle")

    gr.Markdown("### Prévisions détaillées (jours futurs uniquement)")
    gr.HTML(
        f'<div style="{_S["warning"]}">'
        '<b style="font-weight:700;color:inherit">Borne basse / haute :</b> fourchette à 80 % de probabilité. '
        'Utilisez la borne haute pour les commandes maximales, la borne basse pour les minimales.'
        '</div>'
    )
    forecast_table = gr.DataFrame(label="Prévisions journalières futures", wrap=True)

    return (
        file_input, forecast_days, predict_button,
        welcome_msg, kpi_output, metrics_output, forecast_plot, components_plot, forecast_table,
    )


def _tab_guide() -> None:
    gr.Markdown("""
## Comment utiliser cet outil ?

### 1. Préparer vos données
Votre fichier CSV doit contenir **au minimum** :

| Colonne | Type | Description |
|---------|------|-------------|
| `date`  | `YYYY-MM-DD` | Date de la transaction |
| `sales` | entier / décimal | Volume de ventes |

Si le fichier contient aussi `store` et `item`, l'app agrégera automatiquement par jour.

---

### 2. Choisir l'horizon de prévision
- **7–30 jours** — réassort rapide  
- **30–90 jours** — planification mensuelle / trimestrielle  
- **90–365 jours** — plan annuel, budgétisation

---

### 3. Lire les graphiques

**Graphique principal**
- Points noirs = ventes historiques réelles  
- Courbe bleue = prévision Prophet  
- Zone bleue claire = intervalle de confiance 80 %

**Décomposition**
- *Trend* — croissance ou déclin structurel  
- *Weekly* — effet jour de la semaine  
- *Yearly* — saisonnalités annuelles (Noël, soldes…)

---

### 4. Interpréter MAE / MAPE

| MAPE | Interprétation | Action recommandée |
|------|---------------|-------------------|
| < 10 % | Excellent | Modèle prêt pour la production |
| 10–20 % | Correct | Ajouter les jours fériés / promotions |
| > 20 % | Faible | Vérifier la qualité des données |

---

### 5. Exporter les prévisions
Le tableau peut être copié-collé dans Excel ou Google Sheets.
""")


def _tab_about() -> None:
    gr.Markdown("""
## Contexte business

> *"Comment prévoir la demande future pour optimiser les achats et réduire les ruptures de stock ?"*

---

## Objectifs

| Objectif | Impact attendu |
|----------|---------------|
| Prévoir J+1 à J+365 | Réduction des ruptures de stock |
| Détecter les saisonnalités | Meilleure planification des promotions |
| Quantifier l'incertitude | Dimensionnement du stock de sécurité |
| Automatiser la prévision | Gain de temps pour les équipes supply chain |

---

## Jeu de données
- **Source** : `train.csv` — ventes journalières sur 5 ans  
- **Périmètre** : 10 magasins × 50 articles = 500 séries temporelles  
- **Volume** : ~900 000 lignes

---

## Méthodologie

```
Données brutes (CSV)
        │
        ▼
Agrégation journalière (somme toutes séries)
        │
        ▼
Modèle Prophet (tendance + saisonnalité hebdo & annuelle)
        │
        ├── Prévision point
        └── Intervalle de confiance 80 %
```

**Pourquoi Prophet ?** Robuste aux données manquantes, capte automatiquement les tendances et saisonnalités, facile à interpréter par des équipes non-data.

---

## Stack technique
Python 3.13 · Prophet · Pandas / NumPy · Gradio · Matplotlib

---

## Cas d'usage typiques
- **Directeur Supply Chain** — anticiper les besoins en stock 3 mois à l'avance  
- **Category Manager** — identifier les produits en forte croissance  
- **Logisticien** — planifier les livraisons lors des pics saisonniers  
- **Contrôleur de gestion** — valider les budgets prévisionnels de CA
""")


# Point d'entrée public

def build_app() -> gr.Blocks:
    """Construit et retourne l'instance Gradio Blocks."""
    with gr.Blocks(
        title="Sales Forecasting – Tableau de bord prévisionnel",
        css=CUSTOM_CSS,
        theme=gr.themes.Default(
            primary_hue="blue",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        ),
    ) as app:

        gr.HTML("""
        <div>
          <h1>Tableau de bord prévisionnel des ventes</h1>
          <p>
            Prévision de la demande à court et moyen terme pour
            une chaine de magasins alimenté par <strong>Facebook Prophet</strong>
            (détection automatique de tendance et saisonnalité).
          </p>
        </div>
        """)

        with gr.Tabs():
            with gr.TabItem(" Prévision & KPIs"):
                (
                    file_input, forecast_days, predict_button,
                    welcome_msg, kpi_output, metrics_output,
                    forecast_plot, components_plot, forecast_table,
                ) = _tab_forecast()

            with gr.TabItem(" Guide d'utilisation"):
                _tab_guide()

            with gr.TabItem(" À propos du projet"):
                _tab_about()

        # Câblage des événements
        outputs = [kpi_output, metrics_output, forecast_plot, components_plot, forecast_table]

        def _run_and_hide_welcome(file, days):
            """Cache le message d'accueil puis retourne les résultats."""
            results = forecast_sales(file, days)
            return (gr.HTML(visible=False), *results)

        predict_button.click(
            fn=_run_and_hide_welcome,
            inputs=[file_input, forecast_days],
            outputs=[welcome_msg, *outputs],
        )

    return app
