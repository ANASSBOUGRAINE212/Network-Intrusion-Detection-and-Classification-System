# Deploy to Railway RIGHT NOW - Simple Steps

## Prerequisites
- Node.js installed (for Railway CLI)
- Git repository pushed to GitHub

## Step-by-Step Deployment:

### Method 1: Using Railway Dashboard (EASIEST - No CLI needed!)

1. **Go to Railway**: https://railway.app
2. **Sign up** with GitHub (no credit card!)
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose your repository**
6. **Configure**:
   - Root Directory: `backend`
   - Railway auto-detects everything else!
7. **Click Deploy**
8. **Wait 2-3 minutes** for build
9. **Get your URL**: Settings → Generate Domain
10. **Done!** Your API is live at: `https://your-app.up.railway.app`

### Method 2: Using Railway CLI

```bash
# 1. Install Railway CLI
npm i -g @railway/cli

# 2. Go to backend folder
cd backend

# 3. Login to Railway
railway login

# 4. Create new project
railway init

# 5. Deploy!
railway up

# 6. Generate domain
railway domain

# Done! Your API is live!
```

## After Deployment:

### Test Your API:

```bash
# Replace with your Railway URL
curl https://your-app.up.railway.app/health

# Should return:
# {"status": "healthy", "model_loaded": true, ...}
```

### Get Your URL:

**In Railway Dashboard:**
- Go to your project
- Click Settings → Networking
- Click "Generate Domain"
- Copy the URL: `https://your-app.up.railway.app`

### Update Your Frontend:

In `js/app.js`, change:
```javascript
const API_URL = 'https://your-app.up.railway.app';
```

## Troubleshooting:

### Build Failed?
- Check logs in Railway dashboard
- Make sure you set Root Directory to `backend`

### Models Not Loading?
- Railway should auto-include all files
- Check logs: `railway logs`

### App Crashes?
- Check logs: `railway logs`
- Verify environment variables

## What Railway Does Automatically:

✅ Detects Python from requirements.txt
✅ Installs all dependencies
✅ Loads your .keras models (they're small enough!)
✅ Starts gunicorn server
✅ Provides HTTPS URL
✅ Auto-deploys on git push

## Cost:

- **$5 free credit per month**
- Your app uses ~$0.50-1/month
- **No credit card needed initially**

## Next Steps:

1. Deploy backend to Railway (2 minutes)
2. Get your Railway URL
3. Update frontend API_URL
4. Deploy frontend to Netlify (optional)
5. Done! 🎉

## Need Help?

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Check logs: `railway logs`
