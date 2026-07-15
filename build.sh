#!/bin/bash
# build.sh — Build l'app desktop Boeuf Tracker (Tauri)
#
# Prerequis : Rust + Tauri CLI installes
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#   cargo install tauri-cli --version "^2.0"
#
# Sortie : src-tauri/target/release/bundle/
#   macOS :  Boeuf Tracker.app + .dmg
#   Windows: Boeuf Tracker.exe (NSIS) + .msi
#   Linux:   boeuf-tracker.AppImage + .deb
set -e

cd "$(dirname "$0")"

echo "=== Build Boeuf Tracker (Tauri) ==="
echo "Plateforme: $(uname -s) $(uname -m)"
echo ""

# Lance le build Tauri (compile Rust + genere les bundles natifs)
cargo tauri build "$@"

echo ""
echo "=== Build termine ==="
BUNDLE_DIR="src-tauri/target/release/bundle"
if [ -d "$BUNDLE_DIR" ]; then
    echo "Bundles generes dans $BUNDLE_DIR/:"
    find "$BUNDLE_DIR" -maxdepth 2 -type f \( -name "*.app" -o -name "*.dmg" \
        -o -name "*.exe" -o -name "*.msi" -o -name "*.AppImage" -o -name "*.deb" \
        \) -exec ls -lh {} \;
else
    echo "(dossier bundle non trouve — verifier les logs ci-dessus)"
fi
