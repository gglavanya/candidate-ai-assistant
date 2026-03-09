# 🚀 Deployment Guide — Lavanya's RAG Assistant
# ================================================

## 📦 What's in the Docker setup

| File               | Purpose                              |
|--------------------|--------------------------------------|
| Dockerfile         | Builds the FastAPI RAG API image     |
| docker-compose.yml | Runs API + Frontend together locally |
| nginx.conf         | Serves index.html via nginx          |
| requirements.txt   | Python dependencies                  |
| .dockerignore      | Keeps image lean                     |


## ✅ Step 1 — Test Docker locally first

```bash
# Make sure Docker Desktop is running on your Mac

# Build and start everything
docker compose up --build

# API runs at:   http://localhost:8000
# UI  runs at:   http://localhost:3000
# API docs at:   http://localhost:8000/docs

# Stop everything
docker compose down
```


## ☁️ Cloud Platform Comparison

| Platform  | Free Tier | Ease      | Best For              |
|-----------|-----------|-----------|----------------------|
| Render    | ✅ Yes    | ⭐⭐⭐⭐⭐ | Easiest, recommended |
| Railway   | ✅ Yes    | ⭐⭐⭐⭐   | Fast deploys         |
| AWS EC2   | ⚠️ Limited| ⭐⭐⭐     | Production scale     |


## 🟢 Option A — Deploy on Render (RECOMMENDED — Free)

### Why Render?
- Free tier available
- Deploys directly from GitHub
- Auto-deploys on every git push
- No credit card needed

### Steps:

1. Push your project to GitHub:
```bash
git init
git add .
git commit -m "Lavanya RAG Assistant"
git remote add origin https://github.com/YOUR_USERNAME/lavanya-rag
git push -u origin main
```

2. Go to → https://render.com → Sign up

3. Click "New +" → "Web Service"

4. Connect your GitHub repo

5. Fill in the settings:
   - **Name**: lavanya-rag-api
   - **Runtime**: Docker
   - **Dockerfile Path**: ./Dockerfile
   - **Port**: 8000

6. Add Environment Variable:
   - Key:   GROQ_API_KEY
   - Value: your_groq_key_here

7. Click "Deploy" → wait ~3 minutes

8. Your API will be live at:
   https://lavanya-rag-api.onrender.com

9. Update index.html line:
```javascript
const API_URL = "https://lavanya-rag-api.onrender.com";
```

10. Deploy index.html to Render Static Site or Netlify (free)


## 🚂 Option B — Deploy on Railway

1. Go to → https://railway.app → Sign up with GitHub

2. Click "New Project" → "Deploy from GitHub repo"

3. Select your repo

4. Railway auto-detects Dockerfile ✅

5. Add environment variable:
   - GROQ_API_KEY = your_key

6. Set port to 8000 in Settings

7. Live in ~2 minutes at a railway.app URL


## ☁️ Option C — Deploy on AWS EC2

### Launch EC2 instance:
1. Go to AWS Console → EC2 → Launch Instance
2. Choose: Ubuntu 22.04 LTS, t2.micro (free tier)
3. Allow ports: 22 (SSH), 8000 (API), 3000 (UI) in Security Group

### SSH and setup:
```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker ubuntu
newgrp docker

# Clone your repo
git clone https://github.com/YOUR_USERNAME/lavanya-rag
cd lavanya-rag

# Set env variable
echo "GROQ_API_KEY=your_key_here" > .env

# Run!
docker compose up -d

# API at: http://YOUR_EC2_IP:8000
# UI  at: http://YOUR_EC2_IP:3000
```


## 📁 Final Project Structure

```
lavanya-rag/
├── knowledge_base.json           ✅ Phase 1
├── phase2_chunk_and_embed.py     ✅ Phase 2
├── phase3_rag_pipeline.py        ✅ Phase 3
├── phase4_fast_api.py            ✅ Phase 4
├── index.html                    ✅ Phase 5
├── Dockerfile                    ✅ Phase 6
├── docker-compose.yml            ✅ Phase 6
├── nginx.conf                    ✅ Phase 6
├── requirements.txt              ✅ Phase 6
├── .dockerignore                 ✅ Phase 6
├── .env                          ← never commit this!
├── .gitignore                    ← add .env here
└── chroma_db/                    ← auto-created by Phase 2
```


## 🔒 Important — .gitignore

Create a .gitignore file:
```
.env
my_env/
__pycache__/
*.pyc
```

NEVER push your .env or GROQ_API_KEY to GitHub!


## 🎯 My Recommendation

Start with **Render** — it's the easiest path:
- Free
- Just connect GitHub → it reads your Dockerfile → deploys
- You get a public URL to share with HR in minutes
