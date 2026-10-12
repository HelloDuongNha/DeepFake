# Đóng gói Deep Live Cam

## Nội dung phát hành

Bản desktop dùng PyInstaller ở chế độ `onedir`. macOS tạo `DeepLiveCam.app` và `DeepLiveCam-macOS.dmg`; Windows tạo `DeepLiveCam.exe` trong thư mục ứng dụng và `DeepLiveCam-Setup-Windows.exe` bằng Inno Setup. Icon camera có nền trong suốt. Không đưa `models/`, ảnh cá nhân, `switch_states.json` hoặc cache từ checkout vào gói.

Lần chạy đầu app hỏi trước khi tải khoảng 800 MB model cốt lõi (inswapper 128, GPEN 256, 3 model InsightFace) vào thư mục dữ liệu người dùng. Các model nâng cao như GFPGAN/GPEN 512 vẫn tải khi bật tính năng tương ứng. Cần Internet lần đầu, hoặc chép model đã có vào đúng các đường dẫn dưới đây trước khi mở app. **Các model InsightFace/inswapper được chủ dự án công bố cho nghiên cứu phi thương mại; phải xác nhận quyền sử dụng và phân phối trước khi phát hành công khai hoặc thương mại.** Mã nguồn và binary cũng phải được phát hành theo giấy phép dự án và các thư viện đi kèm.

| Hệ điều hành | Dữ liệu người dùng | Trạng thái | Model | Cache |
| --- | --- | --- | --- | --- |
| macOS | `~/Library/Application Support/DeepLiveCam` | `switch_states.json` | `models/` | `cache/` |
| Windows | `%LOCALAPPDATA%\DeepLiveCam` | `switch_states.json` | `models\` | `cache\` |

Biến `DLC_DATA_DIR` có thể đổi toàn bộ thư mục dữ liệu cho thử nghiệm. Khi chạy từ mã nguồn như trước, các đường dẫn vẫn ở checkout nếu không đặt biến này.

## macOS (build trên macOS Apple Silicon)

1. Dùng Python 3.14.7 trong `.venv`, cài `requirements.txt` và `packaging/requirements-build.txt`. Cài `ffmpeg`/`ffprobe` vào `PATH` (Homebrew được hỗ trợ).
2. Chạy `zsh packaging/build_macos.sh` từ repo. Tệp nằm trong `dist/desktop-macos/`.
3. Kiểm tra nhanh: `dist/desktop-macos/DeepLiveCam.app/Contents/MacOS/DeepLiveCam --package-smoke-test`.
4. Mở app từ Finder và cấp quyền camera. Trên máy build, có thể đặt `DLC_DATA_DIR` cho lần chạy thử và chép 5 model cốt lõi từ `models/` vào `$DLC_DATA_DIR/models/`, rồi chạy app để không tải lại.

Kiểm tra một bản đã cài nên bao gồm: chọn ảnh nguồn, camera, bật GPEN 256, Live Preview, xử lý một ảnh/video, mở lại app và xác nhận trạng thái đã lưu. Thử `ffmpeg -version` từ binary kèm trong app nếu chức năng video lỗi. PyInstaller tạo build cho kiến trúc của Python đang dùng; quy trình này tạo bản `arm64`, không phải universal Intel.

### Ký và notarize macOS

Cần Apple Developer Program, chứng chỉ **Developer ID Application**, và thông tin notarytool. Apple yêu cầu hardened runtime và notarization cho ứng dụng phát hành ngoài App Store. Ví dụ:

```sh
xcrun notarytool store-credentials DeepLiveCamNotary --apple-id 'APPLE_ID' --team-id 'TEAM_ID' --password 'APP_SPECIFIC_PASSWORD'
APPLE_IDENTITY='Developer ID Application: Tên tổ chức (TEAM_ID)' APPLE_NOTARY_PROFILE=DeepLiveCamNotary zsh packaging/sign_notarize_macos.sh
```

Script ký `.app`, tạo lại `.dmg` từ app đã ký, gửi `.dmg` cho Apple và staple kết quả. Kiểm tra trên một máy macOS sạch bằng `spctl --assess --type execute`, kéo app từ DMG vào Applications rồi mở. Không công bố DMG chưa ký như một bản cài thông thường; Gatekeeper có thể chặn.

Tài liệu: [Apple notarization](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution), [PyInstaller macOS](https://pyinstaller.org/en/stable/usage.html#mac-os-x-specific-options).

## Windows (build và test trên Windows x64)

1. Cài Python 3.13 x64, Visual Studio Build Tools 2022 với C++ workload và Inno Setup 6 (đưa `ISCC.exe` vào `PATH`). `insightface==0.7.3` chỉ phát hành source trên PyPI nên bước cài dependency phải biên dịch trên Windows. Cài Microsoft Visual C++ Redistributable nếu máy thử thiếu runtime.
2. Có `ffmpeg.exe` và `ffprobe.exe` Windows x64; đặt biến môi trường `FFMPEG_EXE` và `FFPROBE_EXE` trỏ vào hai tệp này. Không dùng binary macOS.
3. Trong PowerShell, chạy `packaging\build_windows.ps1`. Script cài dependency vào `.build-venv`, build executable, kiểm tra package và tạo installer trong `dist\desktop-windows`.
4. Trên một máy Windows x64 sạch, cài bằng `DeepLiveCam-Setup-Windows.exe`. Kiểm tra icon Start Menu/Desktop, đường dẫn LocalAppData, lần tải model đầu, ảnh nguồn + camera + GPEN 256, lưu/mở lại trạng thái, xử lý ảnh/video và gỡ cài đặt. Kiểm tra quyền webcam và FPS thực tế. Chỉ đánh dấu Windows đạt sau khi đã chạy các bước này trên Windows.

### Ký Windows

Cần chứng chỉ code signing được Windows tin cậy, SignTool của Windows SDK và máy chủ timestamp RFC 3161. Sau build:

```powershell
packaging\sign_windows.ps1 -CertificateThumbprint 'THUMBPRINT' -TimestampUrl 'https://timestamp.example.com'
```

Để chữ ký của `.exe` nằm trong bộ cài, đặt `DLC_SIGN_THUMBPRINT` và `DLC_TIMESTAMP_URL` trước khi chạy `build_windows.ps1`; script sẽ ký `.exe`, tạo installer rồi ký installer. `sign_windows.ps1` dùng để ký lại/kiểm tra hai tệp sau build, nhưng không thể thay thế thứ tự này cho một installer đã tạo.

Tài liệu: [Microsoft SignTool](https://learn.microsoft.com/en-us/windows/win32/seccrypto/signtool).

## Giới hạn

`--package-smoke-test` chỉ xác nhận executable mở được, thư mục dữ liệu và danh sách model; không thay thế thử camera, bộ xử lý GPU/CoreML hoặc UI thực. Build Windows chỉ có thể xác nhận trên Windows. Chữ ký và notarization chỉ làm được khi có chứng chỉ/tài khoản tương ứng. Việc phát hành model kèm gói cần quyền riêng từ chủ model; quy trình mặc định không nhúng model.
