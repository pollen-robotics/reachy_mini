from fastapi import APIRouter

router = APIRouter(
    prefix="/daemon",
)


@router.post("/start")
async def start_daemon():
    return {"status": {"daemon": "started"}}


@router.post("/stop")
async def stop_daemon():
    return {"status": {"daemon": "stopped"}}


@router.post("/restart")
async def restart_daemon():
    return {"status": {"daemon": "restarted"}}


@router.get("/status")
async def get_daemon_status():
    return {"status": {"daemon": "enabled"}}
