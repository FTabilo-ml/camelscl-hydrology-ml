param(
  [string]$Region   = "sa-east-1",
  [string]$Profile  = "camelscl",
  [string]$RepoName = "camelscl-serving",
  [string]$FnName   = "camelscl-inference",
  [string]$Tag      = "v1"
)

$ErrorActionPreference = "Stop"

# Descubre AccountId con el perfil indicado
$AccountId = (aws sts get-caller-identity --query Account --output text --profile $Profile)

# Construye la URI de la imagen (evita problemas de ":" usando -f)
$ImageUri  = "{0}.dkr.ecr.{1}.amazonaws.com/{2}:{3}" -f $AccountId, $Region, $RepoName, $Tag
Write-Host "ImageUri: $ImageUri"

# (Opcional) Login a ECR por si hace falta
aws ecr get-login-password --region $Region --profile $Profile `
  | docker login --username AWS --password-stdin "$AccountId.dkr.ecr.$Region.amazonaws.com"

# Build & push multi-arch (Lambda x86_64)
docker buildx build --platform linux/amd64 -f Dockerfile.lambda -t $ImageUri --provenance=false --sbom=false --push .

# Actualiza la Lambda con esa imagen
aws lambda update-function-code `
  --function-name $FnName `
  --image-uri $ImageUri `
  --region $Region --profile $Profile

aws lambda wait function-updated --function-name $FnName --region $Region --profile $Profile

# Smoke test opcional (solo si existe el archivo)
if (Test-Path .\signed_post.py) {
  Write-Host "Ejecutando smoke test con signed_post.py..."
  python .\signed_post.py
} else {
  Write-Host "signed_post.py no encontrado; omitiendo smoke test."
}
