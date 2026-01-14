# PowerShell script to upload contributions to GitHub
# Usage: .\upload_to_github.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub Upload Script for Expert Model Contributions" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
$gitVersion = git --version 2>$null
if (-not $gitVersion) {
    Write-Host "ERROR: Git is not installed!" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Git found: $gitVersion" -ForegroundColor Green
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "annotate_retrieve")) {
    Write-Host "ERROR: Please run this script from the Radar-main directory!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ In correct directory" -ForegroundColor Green
Write-Host ""

# Initialize git if needed
if (-not (Test-Path ".git")) {
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
}

# Check if remote exists
$remoteUrl = git remote get-url origin 2>$null
if (-not $remoteUrl) {
    Write-Host "Adding remote repository..." -ForegroundColor Yellow
    git remote add origin https://github.com/MOsama10/radar-multimodal-radiology.git
    Write-Host "✓ Remote added" -ForegroundColor Green
} else {
    Write-Host "✓ Remote already configured: $remoteUrl" -ForegroundColor Green
}
Write-Host ""

# Create new branch
Write-Host "Creating branch 'expert-model-contributions'..." -ForegroundColor Yellow
git checkout -b expert-model-contributions 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Branch already exists, switching to it..." -ForegroundColor Yellow
    git checkout expert-model-contributions
}
Write-Host "✓ On branch: expert-model-contributions" -ForegroundColor Green
Write-Host ""

# List files to be added
Write-Host "Files to be uploaded:" -ForegroundColor Cyan
$files = @(
    "annotate_retrieve/modeling_expert_model_uncertainty.py",
    "annotate_retrieve/modeling_expert_model_gnn.py",
    "annotate_retrieve/modeling_expert_model_contrastive.py",
    "train_expert_models.py",
    "evaluate_expert_models.py",
    "demo_expert_models.py",
    "CONTRIBUTIONS_README.md",
    "UPLOAD_GUIDE.md"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file (NOT FOUND)" -ForegroundColor Red
    }
}
Write-Host ""

# Ask for confirmation
$confirm = Read-Host "Do you want to proceed with upload? (yes/no)"
if ($confirm -ne "yes") {
    Write-Host "Upload cancelled." -ForegroundColor Yellow
    exit 0
}

# Add files
Write-Host ""
Write-Host "Adding files to git..." -ForegroundColor Yellow
foreach ($file in $files) {
    if (Test-Path $file) {
        git add $file
        Write-Host "  Added: $file" -ForegroundColor Green
    }
}

# Commit
Write-Host ""
Write-Host "Creating commit..." -ForegroundColor Yellow
$commitMessage = @"
Add 3 Expert Model contributions for GTX 1650 Ti

- Contribution 1.1: Uncertainty-Aware Expert Model with MC Dropout
- Contribution 1.2: Hierarchical Multi-Label Classification with GNN
- Contribution 1.3: Contrastive Learning Pre-training

All models tested and working on 4GB VRAM GPU.
Includes training, evaluation, and demo scripts.
"@

git commit -m $commitMessage

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Commit created successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Commit failed" -ForegroundColor Red
    exit 1
}

# Push to GitHub
Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "You may be asked for GitHub credentials..." -ForegroundColor Cyan
Write-Host ""

git push -u origin expert-model-contributions

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✓ SUCCESS! Files uploaded to GitHub" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Go to: https://github.com/MOsama10/radar-multimodal-radiology" -ForegroundColor White
    Write-Host "2. Create a Pull Request from 'expert-model-contributions' branch" -ForegroundColor White
    Write-Host "3. Review and merge the changes" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "✗ Push failed" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "1. Authentication: You may need a Personal Access Token" -ForegroundColor White
    Write-Host "   Get one at: https://github.com/settings/tokens" -ForegroundColor White
    Write-Host ""
    Write-Host "2. Use token as password when prompted" -ForegroundColor White
    Write-Host ""
    Write-Host "3. Or configure git credentials:" -ForegroundColor White
    Write-Host "   git config --global credential.helper wincred" -ForegroundColor Gray
    Write-Host ""
}
