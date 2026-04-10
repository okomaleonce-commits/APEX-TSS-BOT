import logging
import os
import asyncio
from datetime import datetime
from typing import Dict

# Librairies Telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

# Tes librairies TSS (Maths)
import numpy as np
import pandas as pd

# Charger les variables d'environnement
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ==============================================================================
# 1. LOGIQUE TSS
# ==============================================================================

def poisson_pmf(k: int, lam: float) -> float:
    if k < 0 or lam < 0: return 0.0
    return (np.exp(-lam) * (lam ** k)) / np.math.factorial(k)

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

class TriangulationCore:
    def __init__(self, odds: MatchOdds):
        self.raw = odds
        self.p1, self.px, self.p2 = Demarginalizer.shin([odds.odds_1, odds.odds_x, odds.odds_2])
        self.p_over, self.p_under = Demarginalizer.shin([odds.odds_over, odds.odds_under])
        self.p_btts_y, self.p_btts_n = Demarginalizer.shin([odds.odds_btts_yes, odds.odds_btts_no])
        self.ph_over05 = Demarginalizer.shin([odds.odds_home_over05, 1/(1-1/odds.odds_home_over05)*0.95])[0]
        self.pa_over05 = Demarginalizer.shin([odds.odds_away_over05, 1/(1-1/odds.odds_away_over05)*0.95])[0]
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
# 2. MOCK API
# ==============================================================================

def fetch_real_odds_from_api(league, home, away):
    odds = MatchOdds()
    if "Arsenal" in home and "Liverpool" in away:
        odds.odds_1, odds.odds_x, odds.odds_2 = 2.52, 3.55, 2.90
        odds.odds_over, odds.odds_under = 1.78, 2.16
        odds.odds_btts_yes, odds.odds_btts_no = 1.64, 2.38
        odds.odds_home_over05, odds.odds_away_over05 = 1.14, 1.28
        odds.ah_line = 0.0
        odds.odds_ah_home, odds.odds_ah_away = 1.99, 1.99
    else:
        import random
        odds.odds_1, odds.odds_x, odds.odds_2 = 2.50, 3.40, 2.80
        odds.odds_over, odds.odds_under = 1.90, 1.90
        odds.odds_btts_yes, odds.odds_btts_no = 1.75, 2.10
        odds.odds_home_over05, odds.odds_away_over05 = 1.20, 1.40
        odds.ah_line = 0.0
        odds.odds_ah_home, odds.odds_ah_away = 1.95, 1.95
    return odds

# ==============================================================================
# 3. LOGIQUE TELEGRAM
# ==============================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Bienvenue sur **APEX-TSS BOT** !\n\n"
        "Utilise : /analyse JJ/MM HH:MM LIGUE HOME AWAY\n\n"
        "Exemple : /analyse 10/04 21:00 PL Arsenal Liverpool"
    )

async def analyze_match(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        args = context.args
        if len(args) < 5:
            await update.message.reply_text("❌ Format incorrect.\nUtilise : /analyse JJ/MM HH:MM LIGUE HOME AWAY")
            return

        date_match, heure_match, league, home, away = args[0], args[1], args[2], args[3], args[4]
        status_msg = await update.message.reply_text(f"⏳ Analyse de {home} vs {away} en cours...")

        odds_data = fetch_real_odds_from_api(league, home, away)
        engine = TriangulationCore(odds_data)
        results = engine.run_analysis()

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
        logging.error(f"Erreur : {e}")
        await update.message.reply_text(f"⚠️ Une erreur est survenue : {e}")

# ==============================================================================
# 4. MAIN EXECUTION (FIX BOUCLE ASYNCIO)
# ==============================================================================

async def main():
    """Point d'entrée du bot."""
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyse", analyze_match))

    # Cette méthode 'async with' est la plus stable pour éviter les crashs de boucle
    async with application:
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        # Attendre indéfiniment pour garder le bot en vie
        await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
