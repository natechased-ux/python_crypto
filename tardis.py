import websockets, asyncio, json

async def watch_orderbook(symbol="BTCUSDT"):
    url = "wss://api.tardis.dev/ws"  # hypothetical example
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type":"subscribe","symbol": symbol}))
        async for msg in ws:
            data = json.loads(msg)
            # detect any level with, say, size > 100 BTC
            for level in data.get("asks",[]) + data.get("bids",[]):
                if float(level["size"]) > 100:
                    print(f"🚨 Large order at {level['price']} size {level['size']}")

asyncio.run(watch_orderbook())
