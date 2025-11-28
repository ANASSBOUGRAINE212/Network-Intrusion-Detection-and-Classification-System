# Deploy Flask Backend to Railway (No Credit Card!)

Railway offers $5 free credit monthly without requiring a credit card initially.

## Step 1: Sign Up

1. Go to https://railway.app
2. Sign up with GitHub (no credit card needed)
3. You get $5 free credit per month

## Step 2: Deploy from GitHub

### Option A: Using Railway Dashboard (Easiest)

1. Go to https://railway.app/new
2. Click "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects Python
5. Click "Deploy"

### Option B: Using Railway CLI

```bash
# Install Railway CLI
npm i -g @railway/cli

# Or using curl
curl -fsSL https://railway.app/install.sh | sh

# Login
railway login

# Initialize in your backend folder
cd backend
railway init

# Deploy
railway up
```

## Step 3: Configure Service

Railway should auto-detect your Python app, but verify:

**In Railway Dashboard → Your Service → Settings:**

- **Root Directory**: `backend`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`

**Environment Variables:**
- Railway sets `PORT` automatically
- Add if needed:
  - `PYTHON_VERSION`: `3.11`
  - `DEBUG`: `False`

## Step 4: Get Your URL

1. Go to Settings → Networking
2. Click "Generate Domain"
3. Your app will be at: `your-app.up.railway.app`

## Step 5: Handle Model Files

Same issue as Render - models are too large. Options:

### Option A: Railway Volume (Persistent Storage)
```bash
# In Railway Dashboard
Settings → Volumes → Add Volume
Mount path: /app/models
```

### Option B: Cloud Storage (Recommended)
Use AWS S3, Google Cloud Storage, or Cloudinary

### Option C: Git LFS
```bash
git lfs install
git lfs track "*.keras"
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Track models with LFS"
git push
```

## Alternative: PythonAnywhere (100% Free Forever)

If Railway's $5/month isn't enough, try PythonAnywhere:

### PythonAnywhere Setup:

1. **Sign up**: https://www.pythonanywhere.com/registration/register/beginner/
2. **Upload files**:
   - Go to Files tab
   - Upload your backend folder
   - Or clone from GitHub: `git clone <your-repo>`

3. **Install dependencies**:
   ```bash
   # Open Bash console
   cd backend
   pip install --user -r requirements.txt
   ```

4. **Configure Web App**:
   - Go to Web tab → Add a new web app
   - Choose Flask
   - Python version: 3.10
   - Set source code: `/home/yourusername/backend`
   - Edit WSGI file:
   
   ```python
   import sys
   path = '/home/yourusername/backend'
   if path not in sys.path:
       sys.path.append(path)
   
   from app import app as application
   ```

5. **Reload** and visit: `yourusername.pythonanywhere.com`

## Cost Comparison

| Platform | Free Tier | Card Required | Best For |
|----------|-----------|---------------|----------|
| **Railway** | $5 credit/month | No (initially) | Easy deployment |
| **PythonAnywhere** | Always free | No | Long-term free hosting |
| **Fly.io** | 3 VMs free | No | Good performance |
| **Render** | Free tier | Yes | Not recommended |
| **Vercel** | Unlimited | No | Serverless only |

## Recommendation

1. **For quick deployment**: Use **Railway** ($5 credit is plenty for testing)
2. **For permanent free hosting**: Use **PythonAnywhere**
3. **For production**: Eventually upgrade to paid tier on any platform

## Troubleshooting

### Railway Build Fails
- Check build logs in dashboard
- Verify `requirements.txt` is correct
- Ensure Python version compatibility

### PythonAnywhere Issues
- Free tier has limited CPU/memory
- TensorFlow might be slow on free tier
- Consider using lighter models

### Model Files Too Large
- Use cloud storage (S3, GCS)
- Or compress models
- Or use model quantization to reduce size

## Need Help?

- Railway Docs: https://docs.railway.app
- PythonAnywhere Help: https://help.pythonanywhere.com
- Railway Discord: https://discord.gg/railway
