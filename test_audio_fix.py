import asyncio
import edge_tts
import os

async def test():
    tts = edge_tts.Communicate('Hello test', 'en-US-JennyNeural')
    await tts.save('test.mp3')
    print('Audio created:', os.path.exists('test.mp3'))

asyncio.run(test()) 