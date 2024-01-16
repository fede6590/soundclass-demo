# Sound Classification with BEATs 🎵

## Demo deployment

### Option 1: deployment in Streamlit Cloud (very easy but account needed)

Log in to https://share.streamlit.io/deploy.
Use this repository URL and the app.py script as entry point to deploy the app. In "advanced settings", select "Python 3.11"

That's it, ¡good luck!

### Option 2: local deployment using Docker client (less easy and needs Docker)

Make sure Docker is running, and use the Bash terminal to run the following commands:

```
docker build -t beats-app .
```

```
docker run -p 8501:8501 beats-app .
```

Now, access the app at http://localhost:8501 in your browser (you can also try http://0.0.0.0:8501).