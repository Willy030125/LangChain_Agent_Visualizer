"""Dev entrypoint: `python run.py` starts the API with reload."""
import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("app.api.main:app", host=s.api_host, port=s.api_port, reload=True)
