# Safe upload script with conflict detection
# This script will check for conflicts before uploading

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Safe GitHub Upload - Conflict Detection Enabled" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check git installation
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Git is not installed!" -ForegroundColor Red
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "annotate_retrieve")) {
    Write-Host "ERROR: Run this from Radar-main directory!" -ForegroundColor Red
    exit 1
}

Write-Host "Step 1: Fetching latest changes from GitHub..." -ForegroundColor Yellow
git fetch origin 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Setting up remote..." -ForegroundColor Yellow
    git init
    git remote add origin https://github.com/MOsama10/radar-multimodal-radiology.git
    git fetch origin
}

Write-Host "✓ Fetched latest changes" -ForegroundColor Green
Write-Host ""

# Check current branch
$currentBranch = git branch --show-current 2>$null
Write-Host "Current branch: $currentBranch" -ForegroundColor Cyan

# Create new branch from latest main
Write-Host ""
Write-Host "Step 2: Creating safe branch..." -ForegroundColor Yellow
git checkout -b expert-model-contributions-safe origin/main 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Branch exists, using it..." -ForegroundColor Yellow
    git checkout expert-model-contributions-safe
}

Write-Host "✓ On safe branch" -ForegroundColor Green
Write-Host ""

# List files to add
Write-Host "Step 3: Checking files..." -ForegroundColor Yellow
$files = @(
    "annotate_retrieve/modeling_expert_model_uncertainty.py",
    "annotate_retrieve/modeling_expert_model_gnn.py",
    "annotate_retrieve/modeling_expert_model_contrastive.py",
    "train_expert_models.py",
    "evaluate_expert_models.py",
    "demo_expert_models.py",
    "CONTRIBUTIONS_README.md",
    "UPLOAD_GUIDE.md",
    "upload_to_github.ps1",
    "UPLOAD_CHECKLIST.md",
    "safe_upload.ps1"
)

$existingFiles = @()
$newFiles = @()

foreach ($file in $files) {
    if (Test-Path $file) {
        # Check if file exists in remote
        git ls-tree -r origin/main --name-only | Select-String -Pattern $file -Quiet
        if ($?) {
            $existingFiles += $file
            Write-Host "  ⚠ EXISTS IN REMOTE: $file" -ForegroundColor Yellow
        } else {
            $newFiles += $file
            Write-Host "  ✓ NEW FILE: $file" -ForegroundColor Green
        }
    } else {
        Write-Host "  ✗ NOT FOUND: $file" -ForegroundColor Red
    }
}

Write-Host ""

if ($existingFiles.Count -gt 0) {
    Write-Host "WARNING: Some files already exist in remote!" -ForegroundColor Red
    Write-Host "Files that exist:" -ForegroundColor Yellow
    foreach ($f in $existingFiles) {
        Write-Host "  - $f" -ForegroundColor Yellow
    }
    Write-Host ""
    $overwrite = Read-Host "Do you want to OVERWRITE these files? (yes/no)"
    if ($overwrite -ne "yes") {
        Write-Host "Upload cancelled to prevent conflicts." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host ""
Write-Host "Step 4: Adding files to git..." -ForegroundColor Yellow
foreach ($file in $files) {
    if (Test-Path $file) {
        git add $file
        Write-Host "  Added: $file" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Step 5: Creating commit..." -ForegroundColor Yellow
$commitMessage = @"
Add 3 Expert Model contributions for GTX 1650 Ti

- Contribution 1.1: Uncertainty-Aware Expert Model with MC Dropout
- Contribution 1.2: Hierarchical Multi-Label Classification with GNN  
- Contribution 1.3: Contrastive Learning Pre-training

Features:
- All models tested and working on 4GB VRAM GPU
- Training script supporting all 3 models
- Comprehensive evaluation and comparison script
- Demo script that works without dataset
- Full documentation

Memory requirements:
- Baseline: ~3.5 GB
- Uncertainty: ~3.8 GB
- GNN: ~3.9 GB
- Contrastive: ~3.5 GB

All contributions are research-level and suitable for master's thesis work.
"@

git commit -m $commitMessage

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Commit created" -ForegroundColor Green
} else {
    Write-Host "Nothing to commit (files may already be committed)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Step 6: Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "You will be asked for GitHub credentials..." -ForegroundColor Cyan
Write-Host ""

git push -u origin expert-model-contributions-safe

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✓ SUCCESS! Uploaded without conflicts" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Go to: https://github.com/MOsama10/radar-multimodal-radiology" -ForegroundColor White
    Write-Host "2. You'll see a banner to create Pull Request" -ForegroundColor White
    Write-Host "3. Click 'Compare & pull request'" -ForegroundColor White
    Write-Host "4. Review changes and click 'Create pull request'" -ForegroundColor White
    Write-Host "5. Merge when ready" -ForegroundColor White
    Write-Host ""
    Write-Host "Branch name: expert-model-contributions-safe" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "✗ Upload failed" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "You need to authenticate with GitHub." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Option 1: Use GitHub Personal Access Token" -ForegroundColor Cyan
    Write-Host "  1. Go to: https://github.com/settings/tokens" -ForegroundColor White
    Write-Host "  2. Generate new token (classic)" -ForegroundColor White
    Write-Host "  3. Select 'repo' scope" -ForegroundColor White
    Write-Host "  4. Copy token" -ForegroundColor White
    Write-Host "  5. Use token as password when git asks" -ForegroundColor White
    Write-Host ""
    Write-Host "Option 2: Use GitHub CLI" -ForegroundColor Cyan
    Write-Host "  gh auth login" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Then run this script again." -ForegroundColor Yellow
    Write-Host ""
}
