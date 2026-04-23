# Deploying to Render (Backend)

## Option 1: Automatic Setup with render.yaml

1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure the service:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 8000`
   - **Environment**: `Python 3.11`
6. Add Environment Variables:
   - `SECRET_KEY` (auto-generate or set your own)
   - `ALGORITHM` = `HS256`
   - `ACCESS_TOKEN_EXPIRE_MINUTES` = `30`
   - `CORS_ORIGINS` = `*`
7. Click "Create Web Service"

## Option 2: Using Docker on Render

1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure the service:
   - **Environment**: `Docker`
   - **Build Command**: (leave empty)
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 8000`
6. Click "Create Web Service"

## Health Check

The backend includes a health check endpoint at `/health`. Make sure to set the Health Check Path to `/health` in Render settings.

## Troubleshooting

### "failed to read dockerfile: open Dockerfile: no such file or directory"

This error occurs when:
1. You're using Docker deployment but the Dockerfile is not in the repository root
2. The Dockerfile path is incorrect in the build configuration

**Solution**: Ensure the Dockerfile is at the root of your repository OR use the Python environment option instead of Docker.

### Port Configuration

Render exposes the app on port 10000 by default. Update your start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Or set the PORT environment variable in Render.

