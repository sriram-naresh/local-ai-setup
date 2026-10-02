# Docker workflow

The Compose service publishes the app on port `8501`, persists Chroma and chat history in `chroma_db`, and keeps downloaded model files in a named Docker volume.

## Build and run a new feature version

After changing the app, run this from PowerShell in this folder:

```powershell
.\build-feature.ps1
```

From Bash (Git Bash, WSL, or Linux), use:

```bash
bash ./build-feature.sh
```

The script increments the patch version in `VERSION`, builds the matching image tag with Docker Compose, then recreates the app on port `8501`. For a larger release, choose the bump level:

```powershell
.\build-feature.ps1 -Bump minor
.\build-feature.ps1 -Bump major
```

The Bash script accepts the same bump levels: `bash ./build-feature.sh minor` or `bash ./build-feature.sh major`.

The script stops the legacy container named `local-rag` if it is running, because it already owns port `8501`. It leaves that container available to restart if the new Compose service fails to start.

## Compose commands

To build and run the version currently in `VERSION` without incrementing it:

```powershell
$env:APP_VERSION = Get-Content .\VERSION
docker compose up --build --detach
Remove-Item Env:APP_VERSION
```

To view logs or stop the Compose service:

```powershell
docker compose logs --follow app
docker compose down
```

`docker compose down` keeps the Chroma/chat files and the model-cache volume. Use `docker compose up -d` to start the current version again.
