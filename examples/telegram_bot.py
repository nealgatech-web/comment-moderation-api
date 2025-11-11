import os, asyncio, httpx
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def analyze_text(text: str) -> float:
    async with httpx.AsyncClient(timeout=10) as h:
        r = await h.post(f"{API_BASE}/analyze", json={"texts":[text]})
        r.raise_for_status()
        return r.json()["results"][0]["toxicity"]

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    tox = await analyze_text(update.message.text)
    if tox >= 0.8:
        try:
            await update.message.delete()
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text=f"⚠️ Message removed (toxicity {tox:.2f})")
        except Exception:
            # If bot lacks delete permission, just warn
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text=f"⚠️ Toxicity {tox:.2f}. Please adhere to community rules.")

def main():
    if not TOKEN:
        print("Set TELEGRAM_BOT_TOKEN env")
        return
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), on_message))
    app.run_polling()

if __name__ == "__main__":
    main()
