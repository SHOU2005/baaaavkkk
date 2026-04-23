# Deploying to Vercel (Frontend)

## Quick Deploy

1. Push your code to GitHub
2. Go to [Vercel Dashboard](https://vercel.com/dashboard)
3. Click "Add New..." → "Project"
4. Import your GitHub repository
5. Configure the project:
   - **Framework Preset**: `Vite` (or `Other` if not detected)
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
6. Add Environment Variables:
   - `VITE_API_URL` = Your Render backend URL (e.g., `https://your-backend.onrender.com`)
7. Click "Deploy"

## API URL Configuration

After deploying your Render backend, get the backend URL and add it to Vercel:

1. In Vercel dashboard, go to your frontend project
2. Settings → Environment Variables
3. Add: `VITE_API_URL` = `https://your-backend-service.onrender.com`
4. Redeploy the frontend

## Troubleshooting

### CORS Errors

If you see CORS errors in the browser console:

1. Make sure your Render backend has `CORS_ORIGINS=*` set
2. Or set it to your specific V:
   ```
  ercel domain CORS_ORIGINS=https://your-project.vercel.app
   ```

### API Not Reaching Backend

Check the browser Network tab:
1. Verify the API requests are going to the correct `VITE_API_URL`
2. Ensure the backend is running and accessible
3. Check browser console for any error messages

### Hot Reload Not Working

Vercel deployments are production builds. Use `npm run dev` locally for development.

## Multiple Frontend/Backend Setup

If you want a single repository for both:

1. Deploy backend to Render
2. Deploy frontend to Vercel
3. Set `VITE_API_URL` in Vercel to your Render backend URL

