import asyncio
from datetime import datetime, timedelta
from memory import memoryStore

memory_timeout = timedelta(minutes=5)

async def clean_up_memory():
    while(True):
        print("looking for sessions ")
        for session_id in list(memoryStore.keys()):
            print(f"found session - {session_id}")
            if(isExpired(session_id)):
                del memoryStore[session_id]
                print(f"Session {session_id} was deleted due to timeout")


        await asyncio.sleep(60*5)

def isExpired(session_id: str) -> bool:
    if(session_id in memoryStore):
        last_accessed = memoryStore[session_id]["last_accessed"]
        return datetime.now() - last_accessed > memory_timeout
    
    return False
        
    
    
