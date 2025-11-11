import os, asyncio, httpx
import discord

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

async def analyze_text(text: str) -> float:
    async with httpx.AsyncClient(timeout=10) as h:
        r = await h.post(f"{API_BASE}/analyze", json={"texts":[text]})
        r.raise_for_status()
        return r.json()["results"][0]["toxicity"]

@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return
    try:
        tox = await analyze_text(message.content or "")
        if tox >= 0.8:
            await message.delete()
            await message.channel.send(f"⚠️ Message removed for toxicity score {tox:.2f}", delete_after=5)
    except Exception as e:
        print("Analyze error:", e)

if __name__ == "__main__":
    if not TOKEN:
        print("Set DISCORD_BOT_TOKEN env")
    else:
        client.run(TOKEN)
