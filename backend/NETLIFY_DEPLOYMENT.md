# Deploy Flask Backend to Netlify (Serverless Functions)

⚠️ **WARNING**: Netlify has a 50MB function size limit. Your TensorFlow models are likely too large!

## Limitations

- ❌ Function size limit: 50MB (your models are probably larger)
- ❌ 10-second execution timeout on free tier
- ❌ Cold starts can be slow for ML models
- ✅ Good for lightweight APIs only

## If You Still Want to Try:

### Option 1: Netlify Functions (Serverless)

You need to convert your Flask app to serverless functions.

**1. Install Netlify CLI:**
```bash
npm install -g netlify-cli
```

**2. Create netlify.toml:**
```toml
[build]
  command = "pip install -r requirements.txt -t functions/dependencies"
  functions = "functions"
  publish = "."

[functions]
  directory = "functions"
```

**3. Convert Flask to Serverless Function:**

Create `functions/api.py`:
```python
import json
import sys
sys.path.insert(0, 'functions/dependencies')

from flask import Flask, request
import tensorflow as tf
import pickle
import numpy as np

app = Flask(__name__)

# Your Flask routes here...

def handler(event, context):
    # Netlify serverless handler
    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'API working'})
    }
```

**4. Deploy:**
```bash
netlify login
netlify init
netlify deploy --prod
```

### Option 2: Netlify for Frontend + External Backend

**BETTER APPROACH**: Use Netlify for your frontend only!

**Deploy Frontend to Netlify:**
```bash
# In your project root
netlify init
netlify deploy --prod
```

**Deploy Backend Elsewhere:**
- Railway (no card, $5 credit)
- PythonAnywhere (free forever)
- Fly.io (free tier)

**Update Frontend to Call External API:**
```javascript
// In js/app.js
const API_URL = 'https://your-backend.railway.app';
// or 'https://yourusername.pythonanywhere.com'
```

## Recommended Architecture:

```
┌─────────────────┐         ┌──────────────────┐
│   Netlify       │         │   Railway/       │
│   (Frontend)    │ ──────> │   PythonAnywhere │
│   HTML/CSS/JS   │  API    │   (Backend)      │
│   FREE          │  Calls  │   Flask + ML     │
└─────────────────┘         └──────────────────┘
```

## Why This is Better:

1. **Netlify strengths**: Fast CDN, free SSL, auto-deploy from Git
2. **Backend platform strengths**: Better for Python, ML models, long-running processes
3. **No size limits**: Your models can be any size on Railway/PythonAnywhere
4. **Better performance**: Dedicated backend server vs serverless cold starts

## Deploy Frontend to Netlify:

**1. Create netlify.toml in project root:**
```toml
[build]
  publish = "."
  command = "echo 'No build needed for static site'"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

**2. Deploy:**
```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Deploy
netlify deploy --prod
```

**3. Update API URL:**

In `js/app.js`, change:
```javascript
const API_URL = 'https://your-backend.railway.app';
```

**4. Configure CORS on Backend:**

In `backend/config.py`:
```python
CORS_ORIGINS = [
    'https://your-site.netlify.app',
    'http://localhost:*'  # for local development
]
```

## Summary:

| What | Where | Why |
|------|-------|-----|
| **Frontend** (HTML/CSS/JS) | Netlify | Free, fast CDN, easy deployment |
| **Backend** (Flask + ML) | Railway/PythonAnywhere | No size limits, better for ML |

## Full Deployment Steps:

**1. Deploy Backend to Railway:**
```bash
cd backend
railway login
railway init
railway up
# Note your URL: https://your-app.railway.app
```

**2. Deploy Frontend to Netlify:**
```bash
cd ..  # back to project root
netlify login
netlify init
netlify deploy --prod
# Your site: https://your-site.netlify.app
```

**3. Update Frontend API URL:**
```javascript
// js/app.js
const API_URL = 'https://your-app.railway.app';
```

**4. Push changes:**
```bash
git add .
git commit -m "Update API URL"
git push
```

Netlify auto-deploys on push!

## Cost:

- **Netlify (Frontend)**: FREE forever ✅
- **Railway (Backend)**: $5 credit/month (no card) ✅
- **Total**: FREE for testing/demos ✅

## Alternative: All-in-One Platforms

If you want everything in one place:

1. **Vercel** - Frontend + Serverless (but same 50MB limit)
2. **Railway** - Can host both frontend + backend
3. **PythonAnywhere** - Can serve static files + Flask

But the **Netlify (frontend) + Railway (backend)** combo is the best free option!
