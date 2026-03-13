#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .setup(|app| {
      #[cfg(debug_assertions)]
      app.handle().plugin(
        tauri_plugin_log::Builder::default()
          .level(log::LevelFilter::Info)
          .build(),
      )?;

      std::thread::spawn(|| {
        let port = 8000u16;
        let host = "127.0.0.1";
        let addr = format!("{host}:{port}");

        let backend_running = std::net::TcpStream::connect(&addr).is_ok();
        if backend_running {
          return;
        }

        let exe_dir = std::env::current_exe()
          .ok()
          .and_then(|p| p.parent().map(|p| p.to_path_buf()))
          .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        let mut search = Some(exe_dir.clone());
        let mut repo_root: Option<std::path::PathBuf> = None;
        for _ in 0..8 {
          if let Some(dir) = &search {
            let candidate = dir.join("backend").join("main.py");
            if candidate.exists() {
              repo_root = Some(dir.clone());
              break;
            }
            search = dir.parent().map(|p| p.to_path_buf());
          }
        }

        let Some(root) = repo_root else {
          return;
        };

        let mut try_cmds: Vec<(String, Vec<String>)> = vec![
          ("py".into(), vec!["-3".into(), "-m".into(), "uvicorn".into(), "backend.main:app".into(), "--host".into(), host.into(), "--port".into(), port.to_string()]),
          ("python".into(), vec!["-m".into(), "uvicorn".into(), "backend.main:app".into(), "--host".into(), host.into(), "--port".into(), port.to_string()]),
          ("python3".into(), vec!["-m".into(), "uvicorn".into(), "backend.main:app".into(), "--host".into(), host.into(), "--port".into(), port.to_string()]),
        ];

        for (bin, args) in try_cmds.drain(..) {
          let mut cmd = std::process::Command::new(&bin);
          cmd.current_dir(&root).args(&args);
          #[cfg(target_os = "windows")]
          {
            cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
          }
          if let Ok(mut child) = cmd.spawn() {
            std::thread::sleep(std::time::Duration::from_millis(300));
            if std::net::TcpStream::connect(&addr).is_ok() {
              let _ = child; // keep running
              break;
            } else {
              let _ = child.kill();
            }
          }
        }
      });
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}