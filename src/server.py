import os
import uvicorn

# This is a legacy entrypoint that redirects to the refactored api structure
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    # We import here to avoid circular dependencies during refactoring
    from api.server import app

    uvicorn.run(app, host=host, port=port)
