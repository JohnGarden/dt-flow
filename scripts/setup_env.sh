#!/usr/bin/env bash
set -euo pipefail

# Detecta qual comando de Python 3 usar
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
elif command -v py >/dev/null 2>&1; then
    PYTHON="py -3"
else
    echo "Python 3 não encontrado no PATH."
    echo "Instale Python 3 (>=3.10) e/ou verifique as variáveis de ambiente."
    exit 1
fi

echo "Usando Python: $PYTHON"

# Criar ambiente virtual
$PYTHON -m venv --prompt "im12dt" .venv

# Ativar ambiente (Linux/macOS vs Windows/Git Bash)
if [ -f ".venv/bin/activate" ]; then
    # Linux / macOS
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    # Windows (Git Bash / MSYS2)
    source .venv/Scripts/activate
else
    echo "Não foi possível encontrar o script de ativação do venv (.venv/bin/activate ou .venv/Scripts/activate)." >&2
    exit 1
fi

# Instalar dependências do projeto
python -m pip install --upgrade pip

if [ -f pyproject.toml ]; then
    # Instala o projeto em modo editável (e as deps declaradas nele)
    python -m pip install -e .
elif [ -f requirements.txt ]; then
    # Instala dependências listadas
    python -m pip install -r requirements.txt
fi

# Instalar PyTorch (GPU se houver, CPU caso contrário)
python -m pip uninstall -y torch torchvision torchaudio || true

echo "[INFO] Tentando instalar PyTorch com CUDA 12.4..."
if python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio; then
    echo "[INFO] PyTorch CUDA instalado com sucesso."
else
    echo "[WARN] Falha ao instalar build CUDA. Fazendo fallback para PyTorch CPU..."
    python -m pip install torch torchvision torchaudio
fi

echo "[INFO] Build final do PyTorch:"
python - << 'EOF'
import torch
print("torch.__version__   =", torch.__version__)
print("torch.version.cuda  =", getattr(torch.version, "cuda", None))
print("cuda.is_available() =", torch.cuda.is_available())
EOF



# Mensagem final com instruções de ativação
if [ -f ".venv/bin/activate" ]; then
    echo "Ambiente im12-dt criado."
    echo "Ative com: source .venv/bin/activate"
elif [ -f ".venv/Scripts/activate" ]; then
    echo "Ambiente im12-dt criado."
    echo "Ative com:"
    echo "  - Git Bash:  source .venv/Scripts/activate"
    echo "  - CMD:       .venv\\Scripts\\activate.bat"
    echo "  - PowerShell: .venv\\Scripts\\Activate.ps1"
fi
