"""
FFmpeg 自动下载脚本
自动检测平台并下载对应版本的 FFmpeg
"""
import os
import sys
import platform
import subprocess
import hashlib
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


FFMPEG_VERSION = "6.1.1"

DOWNLOAD_SOURCES = {
    "Windows": {
        "x86_64": [
            f"https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2024-04-20-15-35/ffmpeg-N-{FFMPEG_VERSION}-win64-lgpl.zip",
            f"https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl.zip",
        ],
    },
    "Linux": {
        "x86_64": [
            f"https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2024-04-20-15-35/ffmpeg-N-{FFMPEG_VERSION}-linux64-lgpl.tar.xz",
            f"https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-lgpl.tar.xz",
        ],
    },
    "Darwin": {
        "x86_64": [
            f"https://evermeet.cx/pub/ffmpeg/ffmpeg-{FFMPEG_VERSION}.zip",
        ],
        "arm64": [
            f"https://evermeet.cx/pub/ffmpeg/ffmpeg-{FFMPEG_VERSION}.zip",
        ],
    },
}


def get_platform():
    """获取当前平台信息"""
    system = platform.system()
    machine = platform.machine().lower()
    
    if machine in ["amd64", "x86_64"]:
        arch = "x86_64"
    elif machine in ["arm64", "aarch64"]:
        arch = "arm64"
    else:
        arch = "x86_64"
    
    return system, arch


def get_bin_dir():
    """获取项目 bin 目录路径"""
    return Path(__file__).parent.parent / "bin"


def download_file(url, dest_path, resume=True, timeout=30):
    """
    下载文件，支持断点续传
    """
    dest_path = Path(dest_path)
    file_size = 0
    headers = {}
    
    if resume and dest_path.exists():
        file_size = dest_path.stat().st_size
        headers["Range"] = f"bytes={file_size}-"
    
    req = Request(url, headers=headers)
    
    try:
        with urlopen(req, timeout=timeout) as response:
            total_size = int(response.headers.get("Content-Length", -1))
            if file_size > 0 and total_size > 0:
                total_size += file_size
            
            mode = "ab" if resume and file_size > 0 else "wb"
            
            if TQDM_AVAILABLE:
                pbar = tqdm(
                    total=total_size,
                    initial=file_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"下载 {dest_path.name}"
                )
            else:
                print(f"下载中: {dest_path.name} ({total_size / 1024 / 1024:.1f} MB)")
                pbar = None
            
            with open(dest_path, mode) as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    if pbar:
                        pbar.update(len(chunk))
            
            if pbar:
                pbar.close()
        
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def extract_executable(archive_path, dest_dir, exe_name):
    """
    从压缩包中提取指定的可执行文件
    """
    import zipfile
    import tarfile
    
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    
    exe_filename = f"{exe_name}.exe" if sys.platform.startswith("win") else exe_name
    exe_path = dest_dir / exe_filename
    
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if info.filename.endswith(exe_filename):
                    with zf.open(info) as src, open(exe_path, "wb") as dst:
                        dst.write(src.read())
                    print(f"已提取: {exe_filename}")
                    break
    elif archive_path.suffix in (".tar", ".gz", ".xz", ".bz2"):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if member.name.endswith(exe_filename):
                    src = tf.extractfile(member)
                    if src:
                        with open(exe_path, "wb") as dst:
                            dst.write(src.read())
                        print(f"已提取: {exe_filename}")
                        break
    
    if exe_path.exists():
        if not sys.platform.startswith("win"):
            os.chmod(exe_path, 0o755)
        return exe_path
    
    return None


def extract_ffmpeg(archive_path, dest_dir):
    """
    从压缩包中提取 ffmpeg 和 ffprobe 可执行文件
    """
    import zipfile
    import tarfile
    
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    
    print(f"解压文件: {archive_path.name}")
    
    ffmpeg_path = extract_executable(archive_path, dest_dir, "ffmpeg")
    ffprobe_path = extract_executable(archive_path, dest_dir, "ffprobe")
    
    return ffmpeg_path, ffprobe_path


def verify_ffmpeg(ffmpeg_path):
    """验证 FFmpeg 是否能正常执行"""
    try:
        result = subprocess.run(
            [str(ffmpeg_path), "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            print(f"FFmpeg 验证成功: {version_line}")
            return True, version_line
        return False, f"执行失败，返回码: {result.returncode}"
    except Exception as e:
        return False, str(e)


def download_ffmpeg(dest_dir=None, force=False):
    """
    下载对应平台的 FFmpeg
    
    Args:
        dest_dir: 目标目录，默认为项目 bin 目录
        force: 是否强制重新下载
    
    Returns:
        (成功与否, FFmpeg路径或错误信息)
    """
    if dest_dir is None:
        dest_dir = get_bin_dir()
    
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    system, arch = get_platform()
    print(f"检测到平台: {system} {arch}")
    
    ffmpeg_filename = "ffmpeg.exe" if system == "Windows" else "ffmpeg"
    ffmpeg_path = dest_dir / ffmpeg_filename
    
    if ffmpeg_path.exists() and not force:
        print(f"FFmpeg 已存在: {ffmpeg_path}")
        success, info = verify_ffmpeg(ffmpeg_path)
        if success:
            return True, ffmpeg_path
        print("现有 FFmpeg 验证失败，将重新下载")
    
    sources = DOWNLOAD_SOURCES.get(system, {}).get(arch)
    if not sources:
        return False, f"不支持的平台: {system} {arch}"
    
    archive_ext = ".zip" if system in ["Windows", "Darwin"] else ".tar.xz"
    archive_path = dest_dir / f"ffmpeg_download{archive_ext}"
    
    for url in sources:
        print(f"尝试下载: {url}")
        if download_file(url, archive_path, resume=True):
            ffmpeg_path, ffprobe_path = extract_ffmpeg(archive_path, dest_dir)
            if ffmpeg_path:
                archive_path.unlink(missing_ok=True)
                success, version_info = verify_ffmpeg(ffmpeg_path)
                if success:
                    version_file = dest_dir / "FFMPEG_VERSION"
                    version_file.write_text(version_info, encoding="utf-8")
                    return True, ffmpeg_path
        print("下载失败，尝试下一个源...")
    
    return False, "所有下载源均失败"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载 FFmpeg")
    parser.add_argument("--dest", type=str, help="目标目录")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    args = parser.parse_args()
    
    success, result = download_ffmpeg(dest_dir=args.dest, force=args.force)
    if success:
        print(f"\nFFmpeg 下载成功: {result}")
        sys.exit(0)
    else:
        print(f"\n下载失败: {result}")
        sys.exit(1)
