import uvicorn
from .settings import settings

uvicorn.run(
    'skill_diplom_api.app:app',
    host=settings.server_host,
    port=settings.server_port,
    reload=True,
)
