param(
    [ValidateSet("patch", "minor", "major")]
    [string]$Bump = "patch"
)

$ErrorActionPreference = "Stop"
$projectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$versionFile = Join-Path $projectPath "VERSION"
$docker = Get-Command docker -ErrorAction SilentlyContinue

if ($docker) {
    $dockerCommand = $docker.Source
} else {
    $dockerCommand = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe"
    if (-not (Test-Path -LiteralPath $dockerCommand)) {
        throw "Docker CLI was not found. Start Docker Desktop or add docker.exe to PATH."
    }
}

$currentVersion = (Get-Content -LiteralPath $versionFile -Raw).Trim()
if ($currentVersion -notmatch '^(\d+)\.(\d+)\.(\d+)$') {
    throw "VERSION must contain a semantic version such as 1.1.0. Found: $currentVersion"
}

$major = [int]$Matches[1]
$minor = [int]$Matches[2]
$patch = [int]$Matches[3]
switch ($Bump) {
    "major" { $major++; $minor = 0; $patch = 0 }
    "minor" { $minor++; $patch = 0 }
    "patch" { $patch++ }
}

$newVersion = "$major.$minor.$patch"
Set-Content -LiteralPath $versionFile -Value $newVersion -NoNewline
$previousAppVersion = $env:APP_VERSION
$env:APP_VERSION = $newVersion

Push-Location $projectPath
try {
    Write-Host "Building local-rag:$newVersion..."
    & $dockerCommand compose build app
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose image build failed."
    }

    $legacyContainer = & $dockerCommand ps --filter "name=^/local-rag$" --format "{{.Names}}"
    $legacyRunning = ([string]::Join("", $legacyContainer)).Trim() -eq "local-rag"
    if ($legacyRunning) {
        Write-Host "Stopping the existing local-rag container to free port 8501..."
        & $dockerCommand stop local-rag
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stop the existing local-rag container."
        }
    }

    & $dockerCommand compose up --detach --force-recreate app
    if ($LASTEXITCODE -ne 0) {
        if ($legacyRunning) {
            & $dockerCommand start local-rag | Out-Null
        }
        throw "Docker Compose could not start local-rag:$newVersion."
    }

    Write-Host "Running local-rag:$newVersion on http://localhost:8501"
} finally {
    Pop-Location
    if ($null -eq $previousAppVersion) {
        Remove-Item Env:APP_VERSION -ErrorAction SilentlyContinue
    } else {
        $env:APP_VERSION = $previousAppVersion
    }
}
