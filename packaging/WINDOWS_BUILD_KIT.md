# Bộ nguồn build Windows — chưa phải ứng dụng chạy được

ZIP này chứa mã nguồn và cấu hình build, không chứa `.exe`, model, ảnh cá nhân, cache hay trạng thái UI. PyInstaller phải build trên Windows để tạo ứng dụng Windows.

1. Giải nén ZIP trên máy Windows x64.
2. Cài Python 3.13 x64, Visual Studio Build Tools 2022 với C++ workload, Inno Setup 6 và FFmpeg/FFprobe Windows x64. Đưa `ISCC.exe` vào `PATH`.
3. Trong PowerShell tại thư mục đã giải nén:

```powershell
$env:FFMPEG_EXE = 'C:\path\to\ffmpeg.exe'
$env:FFPROBE_EXE = 'C:\path\to\ffprobe.exe'
.\packaging\build_windows.ps1
```

Script tạo `dist\desktop-windows\DeepLiveCam-Windows.zip`, chứa thư mục app portable với `DeepLiveCam.exe` và `DeepLiveCam-Setup-Windows.exe`. Nó cũng chạy smoke test executable. Sau đó cần thử camera, đổi mặt và installer trên Windows thật. Nếu có chứng chỉ, đặt `DLC_SIGN_THUMBPRINT` và `DLC_TIMESTAMP_URL` trước khi build để ký `.exe` rồi mới đóng và ký installer.

Xem `packaging\README.md` để biết thư mục dữ liệu, model tải lần đầu, quyền model và các bước kiểm thử.
