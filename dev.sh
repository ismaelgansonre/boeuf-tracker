#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# dev.sh — Démarre le worker Python + le serveur Bun en parallèle
#
# Usage:  ./dev.sh [args passés à python app.py]
# Ex:     ./dev.sh                          # MLX + vidéo par défaut
#         ./dev.sh --source webcam.mp4
# ═══════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"

# Couleurs
GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'
log() { echo -e "${GREEN}[dev]${NC} $1"; }
warn() { echo -e "${YELLOW}[dev]${NC} $1"; }

# Trap : tue les deux process au Ctrl+C
PYTHON_PID=""
BUN_PID=""
cleanup() {
    log "Arrêt..."
    [ -n "$PYTHON_PID" ] && kill "$PYTHON_PID" 2>/dev/null
    [ -n "$BUN_PID" ] && kill "$BUN_PID" 2>/dev/null
    wait 2>/dev/null
    log "Arrêté."
}
trap cleanup EXIT INT TERM

# 1. Worker Python (port 8100)
log "Démarrage worker Python (port 8100)..."
PYTHON_ARGS="${@:---mlx}"
source .venv/bin/activate 2>/dev/null || true
python app.py $PYTHON_ARGS --port 8100 &
PYTHON_PID=$!

# 2. Attendre que le worker soit prêt
log "Attente du worker Python..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8100/ >/dev/null 2>&1; then
        log "Worker Python prêt ✓"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        warn "Worker Python lent à démarrer, Bun démarre quand même..."
    fi
done

# 3. Serveur Bun (port 8000)
log "Démarrage serveur Bun (port 8000)..."
cd web
export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"

# Installe hono si manquant
if [ ! -d "node_modules" ]; then
    log "Installation des dépendances Bun..."
    bun install 2>&1 | tail -2
fi

bun run src/server.ts &
BUN_PID=$!
cd ..

# 4. Attendre que Bun soit prêt
sleep 2
if curl -sf http://localhost:8000/ >/dev/null 2>&1; then
    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✓ Boeuf Tracker prêt${NC}"
    echo -e "${GREEN}  UI     : http://localhost:8000${NC}"
    echo -e "${GREEN}  Worker : http://localhost:8100${NC}"
    echo -e "${GREEN}  Ctrl+C pour arrêter${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════${NC}"
else
    warn "Serveur Bun pas encore prêt, patientez quelques secondes..."
fi

# Attendre que l'un des deux meurt
wait
