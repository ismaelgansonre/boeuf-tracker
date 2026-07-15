// Boeuf Tracker - Tauri desktop wrapper
//
// Lance le worker Python (app.py) au demarrage, attend qu'il soit pret,
// puis charge l'UI depuis http://127.0.0.1:8100. Tue le worker a la fermeture.
//
// Le worker Python sert a la fois l'API (/api/*, /video_feed) et l'UI
// statique (web/public/), donc pas besoin de serveur Bun separe.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::Manager;

/// Garde une reference sur le process Python pour le tuer a la fermeture.
struct PythonWorker(Mutex<Option<Child>>);

const WORKER_PORT: u16 = 8100;
const WORKER_URL: &str = "http://127.0.0.1:8100";
const HEALTH_TIMEOUT_SECS: u64 = 60;

/// Detecte le binaire Python a utiliser (venv en priorite, puis systeme).
fn find_python() -> Option<String> {
    let root = resolve_project_root();
    // 1. venv local (.venv/bin/python ou .venv/Scripts/python.exe)
    let venv_rel = if cfg!(windows) {
        ".venv/Scripts/python.exe"
    } else {
        ".venv/bin/python"
    };
    let venv_py = root.join(venv_rel);
    if venv_py.exists() {
        return Some(venv_py.to_string_lossy().into_owned());
    }
    // 2. Python systeme
    let candidates: &[&str] = if cfg!(windows) {
        &["python", "python3"]
    } else {
        &["python3", "python"]
    };
    for py in candidates {
        if Command::new(py)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
        {
            return Some((*py).to_string());
        }
    }
    None
}

/// Lance le worker Python en arriere-plan.
fn spawn_worker() -> Option<Child> {
    let python = find_python()?;
    println!("[boeuf] Worker Python: {}", python);

    // Le cwd de Tauri est src-tauri/ en dev, ou le bundle en release.
    // On doit lancer app.py depuis la racine du projet.
    let project_root = resolve_project_root();
    println!("[boeuf] Racine du projet: {}", project_root.display());

    let app_py = project_root.join("app.py");
    if !app_py.exists() {
        eprintln!(
            "[boeuf] app.py introuvable dans {}",
            project_root.display()
        );
        return None;
    }

    let mut cmd = Command::new(&python);
    cmd.current_dir(&project_root)
        .args([
            app_py.to_str().unwrap_or("app.py"),
            "--port",
            &WORKER_PORT.to_string(),
            "--host",
            "127.0.0.1",
        ]);
    #[cfg(debug_assertions)]
    {
        cmd.stdout(Stdio::inherit());
        cmd.stderr(Stdio::inherit());
    }
    #[cfg(not(debug_assertions))]
    {
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::null());
    }
    cmd.spawn().ok()
}

/// Resout la racine du projet (ou se trouve app.py).
/// En dev : parent de src-tauri/. En release : cwd ou repertoire du binaire.
fn resolve_project_root() -> std::path::PathBuf {
    // 1. Try le repertoire courant (marche si lance depuis la racine)
    if std::path::Path::new("app.py").exists() {
        return std::env::current_dir().unwrap_or_default();
    }
    // 2. Try le parent du cwd (marche en dev, cwd = src-tauri/)
    if let Ok(cwd) = std::env::current_dir() {
        let parent = cwd.parent().unwrap_or(&cwd).to_path_buf();
        if parent.join("app.py").exists() {
            return parent;
        }
    }
    // 3. Fallback : repertoire de l'executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if dir.join("app.py").exists() {
                return dir.to_path_buf();
            }
            // parent du binaire (cas bundle macOS .app/Contents/MacOS/)
            if let Some(parent) = dir.parent() {
                if parent.join("app.py").exists() {
                    return parent.to_path_buf();
                }
            }
        }
    }
    // Dernier recours : cwd
    std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
}

/// Attend que le worker Python reponde (health check TCP).
/// Retourne true si pret, false si timeout.
fn wait_for_worker() -> bool {
    use std::net::TcpStream;
    let start = Instant::now();
    let timeout = Duration::from_secs(HEALTH_TIMEOUT_SECS);
    let addr: std::net::SocketAddr = format!("127.0.0.1:{}", WORKER_PORT)
        .parse()
        .expect("adresse invalide");

    println!(
        "[boeuf] Attente du worker Python (max {}s)...",
        HEALTH_TIMEOUT_SECS
    );
    while start.elapsed() < timeout {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_ok() {
            println!(
                "[boeuf] Worker Python pret en {:.1}s",
                start.elapsed().as_secs_f64()
            );
            return true;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    eprintln!(
        "[boeuf] TIMEOUT: le worker Python n'a pas repondu en {}s",
        HEALTH_TIMEOUT_SECS
    );
    false
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(PythonWorker(Mutex::new(None)))
        .setup(|app| {
            let handle = app.handle().clone();

            // Affiche d'abord le splash screen (sert par Tauri en static via frontendDist).
            // Tauri charge index.html par defaut, on bascule vers splash.html.
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.navigate(
                    tauri::Url::parse("http://tauri.localhost/splash.html")
                        .unwrap_or_else(|_| {
                            tauri::Url::parse("https://tauri.localhost/splash.html").unwrap()
                        }),
                );
            }

            // Lance le worker Python avant d'afficher l'UI reelle.
            let worker = match spawn_worker() {
                Some(w) => w,
                None => {
                    eprintln!(
                        "[boeuf] ERREUR: Python introuvable. \
                         Installez Python 3 ou creez un venv (.venv)."
                    );
                    handle.exit(1);
                    return Ok(());
                }
            };

            // Stocke le handle du process pour le cleanup.
            let state = app.state::<PythonWorker>();
            *state.0.lock().unwrap() = Some(worker);

            // Attend que le worker soit pret (charge les modeles MLX + CLIP,
            // peut prendre 15-30s au 1er demarrage).
            let ready = wait_for_worker();
            if !ready {
                eprintln!("[boeuf] Le worker n'a pas demarre correctement.");
            }

            // Bascule de l'UI : du splash vers le worker Python.
            if let Some(window) = app.get_webview_window("main") {
                if let Ok(url) = tauri::Url::parse(WORKER_URL) {
                    let _ = window.navigate(url);
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            // Tue le worker Python a la fermeture de la fenetre.
            if let tauri::WindowEvent::Destroyed = event {
                let state = window.app_handle().state::<PythonWorker>();
                let child = { state.0.lock().unwrap().take() };
                if let Some(mut child) = child {
                    println!("[boeuf] Arret du worker Python...");
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("erreur lors du lancement de Boeuf Tracker");
}
