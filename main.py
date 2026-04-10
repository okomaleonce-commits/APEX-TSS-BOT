import logging
import os
import math
import asyncio
import requests
from datetime import datetime
from typing import Dict, Optional

# Librairies Telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

# Tes librairies TSS (Maths)
import numpy as np
import pandas as pd

# Charger les variables d'environnement
load_dotenv()

# Récupération des variables
TOKEN = os.getenv("TELEGRAM_TOKEN")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
FOOTYSTATS_KEY = os.getenv("FOOTYSTATS_KEY")
API_KEY = os.getenv("API_KEY") # Clé générique
CHAT_ID = os.getenv("CHAT_ID")  # ID du chat autorisé

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ==============================================================================
# 1. LOGIQUE TSS (MATHÉMATIQUES)
# ==============================================================================

def poisson_pmf(k: int, lam: float) -> float:
    if k < 0 or lam < 0: return 0.0
    return (np.exp(-lam) * (lam ** k)) / math.factorial(k)

def poisson_cdf(k: int, lam: float) -> float:
    return sum(poisson_pmf(i, lam) for i in range(k + 1))

class Demarginalizer:
    @staticmethod
    def shin(odds_list):
        n = len(odds_list)
        if n == 0: return []
        p_brutes = np.array([1.0 / o if o > 0 else 0 for o in odds_list])
        overround = np.sum(p_brutes)
        if overround <= 1.0: return (p_brutes / overround).tolist()
        z = (overround - 1) / (overround * n - 1)
        denominator = 1 - (z * n)
        if abs(denominator) < 1e-6: return (p_brutes / overround).tolist()
        p_nettes = (p_brutes - z) / denominator
        p_nettes = np.clip(p_nettes, 0.0001, 0.9999)
        return (p_nettes / np.sum(p_nettes)).tolist()

class MatchOdds:
    def __init__(self):
        self.odds_1, self.odds_x, self.odds_2 = 0.0, 0.0, 0.0
        self.odds_over, self.odds_under = 0.0, 0.0
        self.odds_btts_yes, self.odds_btts_no = 0.0, 0.0
        self.odds_home_over05, self.odds_away_over05 = 0.0, 0.0
        self.ah_line = 0.0
        self.odds_ah_home, self.odds_ah_away = 0.0, 0.0
        self.team_stats_home = {} 
        self.team_stats_away = {}

class TriangulationCore:
    def __init__(self, odds: MatchOdds):
        self.raw = odds
        self.p1, self.px, self.p2 = Demarginalizer.shin([odds.odds_1, odds.odds_x, odds.odds_2])
        self.p_over, self.p_under = Demarginalizer.shin([odds.odds_over, odds.odds_under])
        self.p_btts_y, self.p_btts_n = Demarginalizer.shin([odds.odds_btts_yes, odds.odds_btts_no])
        
        # Estimation Team Totals si pas dispo
        self.ph_over05 = Demarginalizer.shin([odds.odds_home_over05, 1/(1-1/odds.odds_home_over05)*0.95])[0] if odds.odds_home_over05 > 0 else (self.p1 * 0.7 + 0.1)
        self.pa_over05 = Demarginalizer.shin([odds.odds_away_over05, 1/(1-1/odds.odds_away_over05)*0.95])[0] if odds.odds_away_over05 > 0 else (self.p2 * 0.7 + 0.1)
        
        self.p_ah_home = Demarginalizer.shin([odds.odds_ah_home, odds.odds_ah_away])[0]
        self.lambda_total = 0.0 

    def run_analysis(self) -> Dict:
        base_prob = self.ph_over05 * self.pa_over05
        factor = 1.30
        if base_prob < 0.20: factor = 1.08
        elif base_prob < 0.30: factor = 1.21
        elif base_prob < 0.40: factor = 1.33
        elif base_prob < 0.50: factor = 1.29
        elif base_prob < 0.60: factor = 1.31
        p_synth_btts = base_prob * factor
        
        target_p_over = self.p_over
        low, high = 0.5, 5.0
        for _ in range(25):
            mid = (low + high) / 2
            if (1 - poisson_cdf(2, mid)) > target_p_over: high = mid
            else: low = mid
        lambda_total = (low + high) / 2
        
        delta = p_synth_btts - self.p_btts_y
        signal = "NEUTRE ⚪"
        if delta > 0.05: signal = "VALUE DETECTÉE 🔴"
        elif delta < -0.05: signal = "SURÉVALUÉ 🔵"
        
        return {
            "p_book_btts": self.p_btts_y,
            "p_synth_btts": p_synth_btts,
            "delta": delta,
            "signal": signal,
            "lambda": lambda_total,
            "cote_btts": self.raw.odds_btts_yes
        }

# ==============================================================================
# 2. INTEGRATION API ODDS (AVEC FALLBACK ROBUSTE)
# ==============================================================================

LEAGUE_MAP = {
    "PL": "soccer_epl", "LL": "soccer_la_liga", "SA": "soccer_serie_a",
    "BL1": "soccer_bundesliga", "FL1": "soccer_ligue_one", "ELC": "soccer_championship",
    "CL": "soccer_uefa_champions_league", "EL": "soccer_uefa_europa_league"
}

def fetch_real_odds_from_api(league, home, away):
    """
    Récupère les cotes via The-Odds-API avec stratégie de repli (Fallback).
    """
    odds = MatchOdds()
    
    sport_key = LEAGUE_MAP.get(league)
    if not sport_key:
        logging.warning(f"Ligue {league} non reconnue.")
        return None

    if not ODDS_API_KEY:
        logging.error("ODDS_API_KEY manquante.")
        return None

    target_books = os.getenv("ODDS_API_BOOKMAKERS", "pinnacle")
    
    # Fonction interne pour tenter une requête
    def attempt_fetch(use_specific_books=True):
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
        params = {
            'apiKey': ODDS_API_KEY,
            'regions': 'eu',
            'markets': 'h2h,btts,totals',
            'oddsFormat': 'decimal',
            'dateFormat': 'iso',
        }
        
        if use_specific_books and target_books:
            params['bookmakers'] = target_books

        try:
            logging.info(f"Requête API : Books={target_books if use_specific_books else 'TOUS'}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logging.critical("ERREUR 401 : Clé API invalide.")
            elif e.response.status_code == 429:
                logging.critical("ERREUR 429 : Quota API dépassé.")
            else:
                logging.error(f"Erreur HTTP API: {e}")
            return None
        except Exception as e:
            logging.error(f"Erreur réseau: {e}")
            return None

    # --- TENTATIVE 1 : Avec tes books spécifiques ---
    data = attempt_fetch(use_specific_books=True)
    
    # --- TENTATIVE 2 : Si échec et qu'on avait un filtre, on réessaie SANS filtre ---
    if data is None and target_books:
        logging.warning("Aucun résultat avec le filtre bookmaker. Tentative avec TOUS les books...")
        data = attempt_fetch(use_specific_books=False)

    if not data:
        return None

    # --- RECHERCHE DU MATCH ---
    match_found = None
    for event in data:
        h_match = event['home_team'].lower()
        a_match = event['away_team'].lower()
        
        if home.lower() in h_match and away.lower() in a_match:
            match_found = event
            break
    
    if not match_found:
        logging.warning(f"Match {home} vs {away} non trouvé.")
        return None

    # --- EXTRACTION DES COTES ---
    for bookmaker in match_found['bookmakers']:
        current_odds = {}
        for market in bookmaker['markets']:
            if market['key'] == 'h2h':
                for outcome in market['outcomes']:
                    if outcome['name'] == match_found['home_team']: current_odds['1'] = outcome['price']
                    if outcome['name'] == 'Draw': current_odds['X'] = outcome['price']
                    if outcome['name'] == match_found['away_team']: current_odds['2'] = outcome['price']
            elif market['key'] == 'btts':
                for outcome in market['outcomes']:
                    if outcome['name'] == 'Yes': current_odds['btts_yes'] = outcome['price']
                    if outcome['name'] == 'No': current_odds['btts_no'] = outcome['price']
            elif market['key'] == 'totals':
                for outcome in market['outcomes']:
                    name = outcome['name']
                    if 'Over' in name and '2.5' in name: current_odds['over'] = outcome['price']
                    if 'Under' in name and '2.5' in name: current_odds['under'] = outcome['price']

        if all(k in current_odds for k in ['1', 'X', '2', 'btts_yes', 'over']):
            odds.odds_1 = current_odds['1']
            odds.odds_x = current_odds['X']
            odds.odds_2 = current_odds['2']
            odds.odds_btts_yes = current_odds['btts_yes']
            odds.odds_btts_no = current_odds.get('btts_no', 0)
            odds.odds_over = current_odds['over']
            odds.odds_under = current_odds.get('under', 0)
            
            odds.odds_home_over05 = 1.0 / (current_odds['1'] * 0.6)
            odds.odds_away_over05 = 1.0 / (current_odds['2'] * 0.6)
            odds.ah_line = 0.0
            odds.odds_ah_home = current_odds['1']
            odds.odds_ah_away = current_odds['2']
            
            logging.info(f"✅ Cotes trouvées chez {bookmaker['title']} pour {home} vs {away}")
            return odds

    logging.warning("Aucun bookmaker n'avait tous les marchés.")
    return None

# ==============================================================================
# 3. INTEGRATION FOOTYSTATS (STRUCTURE)
# ==============================================================================

def fetch_footystats_stats(team_name):
    """
    À compléter selon la documentation de FootyStats.
    Retourne None pour l'instant pour éviter le crash.
    """
    if not FOOTYSTATS_KEY:
        return None
    # logique d'appel API ici
    return None 

# ==============================================================================
# 4. LOGIQUE TELEGRAM & SÉCURITÉ
# ==============================================================================

def check_permission(update: Update) -> bool:
    if not CHAT_ID:
        return True 
    return str(update.effective_chat.id) == str(CHAT_ID)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_permission(update):
        await update.message.reply_text("⛔ Accès refusé.")
        return
        
    await update.message.reply_text(
        "👋 Bienvenue sur **APEX-TSS BOT vFinal** !\n\n"
        "Intégration : OddsAPI (Auto-Fallback)\n"
        "Utilise : /analyse JJ/MM HH:MM LIGUE HOME AWAY\n\n"
        "Exemple : /analyse 13/04 16:00 PL Arsenal Villa"
    )

async def analyze_match(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_permission(update):
        await update.message.reply_text("⛔ Accès refusé.")
        return

    try:
        args = context.args
        if len(args) < 5:
            await update.message.reply_text("❌ Format incorrect.\n/analyse JJ/MM HH:MM LIGUE HOME AWAY")
            return

        date_match, heure_match, league, home, away = args[0], args[1], args[2], args[3], args[4]
        status_msg = await update.message.reply_text(f"⏳ Analyse de {home} vs {away} en cours...")

        # 1. Récupération des Cotes
        odds_data = fetch_real_odds_from_api(league, home, away)
        if odds_data is None:
            await status_msg.edit_text(
                f"❌ **Impossible de trouver les cotes.**\n"
                f"Vérifie ligue ({league}) et noms d'équipes.\n"
                f"Ou vérifie que le match n'est pas terminé."
            )
            return

        # 2. Récupération des Stats
        stats_home = fetch_footystats_stats(home)
        stats_away = fetch_footystats_stats(away)
        
        # 3. Calcul TSS
        engine = TriangulationCore(odds_data)
        results = engine.run_analysis()

        # 4. Construction du Rapport
        rapport = (
            f"🔺 **RAPPORT TSS - {league}**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏟️ **Match** : {home} vs {away}\n"
            f"📅 **Date**  : {date_match} à {heure_match}\n\n"
            
            f"📊 **ANALYSE BTTS YES**\n"
            f"--------------------------\n"
            f"📉 Cote Marché  : {results['cote_btts']:.2f}\n"
            f"🧮 P(Book)      : {results['p_book_btts']:.1%}\n"
            f"🚀 P(Modèle TSS): {results['p_synth_btts']:.1%}\n"
            f"⚡ **DELTA**    : {results['delta']:+.1%}\n\n"
            
            f"🎯 **SIGNAL** : {results['signal']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
        )

        await status_msg.edit_text(rapport, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"Erreur critique: {e}")
        if 'status_msg' in locals():
            await status_msg.edit_text(f"⚠️ Erreur système : {e}")

# ==============================================================================
# 5. MAIN
# ==============================================================================

async def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyse", analyze_match))

    async with application:
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
