# file: redirect_to_file.py
import os
import sys
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

class _FDReader(threading.Thread):
    """后台线程：从原 pipe read_fd 读取 bytes，写到 file_obj 和 original_fd"""
    def __init__(self, read_fd: int, original_fd: int, file_obj, name: str):
        super().__init__(daemon=True, name=f"fd-reader-{name}")
        self.read_fd = read_fd
        self.original_fd = original_fd
        self.file_obj = file_obj
        self._stopped = threading.Event()

    def run(self):
        try:
            while not self._stopped.is_set():
                chunk = os.read(self.read_fd, 4096)
                if not chunk:
                    break
                try:
                    # 写回原始控制台（binary）
                    os.write(self.original_fd, chunk)
                except OSError:
                    pass
                try:
                    # 写入日志文件（解码为 str）
                    self.file_obj.write(chunk.decode("utf-8", errors="replace"))
                    self.file_obj.flush()
                except Exception:
                    # 保底：如果 file_obj 出问题，仍要继续把内容写回控制台
                    pass
        except Exception:
            pass

    def stop(self):
        self._stopped.set()

class RedirectStdoutStderrToFile:
    """
    上下文管理 / 一次性设置类 —— 在 enter 时开始 capture，
    exit 时恢复原状。
    """
    def __init__(self, log_path: Optional[str] = None, encoding="utf-8"):
        self.log_path = log_path
        self.encoding = encoding

        # 保存
        self._orig_stdout_fd = None
        self._orig_stderr_fd = None
        self._saved_stdout = None
        self._saved_stderr = None

        self._stdout_pipe = None  # (r, w)
        self._stderr_pipe = None
        self._stdout_reader = None
        self._stderr_reader = None
        self._log_file = None

    def __enter__(self):
        # 打开文件（文本模式）
        self._log_file = open(self.log_path, "a", encoding=self.encoding, buffering=1)  # line-buffered
        # 保存 Python 层的对象
        self._saved_stdout = sys.stdout
        self._saved_stderr = sys.stderr

        # 保存原始 file descriptors（复制）
        self._orig_stdout_fd = os.dup(1)
        self._orig_stderr_fd = os.dup(2)

        # 创建 pipe 并把 stdout/stderr 的 FD 指向 pipe 的 write 端
        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        self._stdout_pipe = (stdout_r, stdout_w)
        self._stderr_pipe = (stderr_r, stderr_w)

        # dup2: 将 fd 1/2 指向 pipe 的 write 端（从此底层写入会进入 pipe）
        os.dup2(stdout_w, 1)
        os.dup2(stderr_w, 2)

        # 启动后台线程读取 pipe 的 read 端，写回原始控制台 fd 并写入文件
        self._stdout_reader = _FDReader(stdout_r, self._orig_stdout_fd, self._log_file, "stdout")
        self._stderr_reader = _FDReader(stderr_r, self._orig_stderr_fd, self._log_file, "stderr")
        self._stdout_reader.start()
        self._stderr_reader.start()

        # 也替换 Python 层 sys.stdout/sys.stderr，防止部分库直接使用 sys.stdout.write()
        class _StdIOWrapper:
            def __init__(self, fd_write):
                self._fd = fd_write
            def write(self, s):
                if s is None:
                    return
                if not isinstance(s, (bytes, bytearray)):
                    s = str(s).encode("utf-8")
                try:
                    os.write(self._fd, s)
                except OSError:
                    pass
            def flush(self):
                try:
                    os.fsync(self._fd)
                except Exception:
                    pass
            def writelines(self, lines):
                for ln in lines:
                    self.write(ln)
            @property
            def encoding(self):
                return "utf-8"

        # sys.stdout 写入到 FD 1（已经被 dup2 到 pipe write）
        sys.stdout = _StdIOWrapper(1)
        sys.stderr = _StdIOWrapper(2)

        # 在日志文件写一行表头
        self._log_file.write(f"\n=== Log started at {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
        self._log_file.flush()

        return self

    def __exit__(self, exc_type, exc, tb):
        # 写结尾并 flush
        try:
            self._log_file.write(f"\n=== Log ended at {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
            self._log_file.flush()
        except Exception:
            pass

        # restore sys.stdout/sys.stderr
        try:
            sys.stdout = self._saved_stdout
            sys.stderr = self._saved_stderr
        except Exception:
            pass

        # Stop readers and close pipe write ends so readers see EOF
        try:
            if self._stdout_reader:
                self._stdout_reader.stop()
        except Exception:
            pass
        try:
            if self._stderr_reader:
                self._stderr_reader.stop()
        except Exception:
            pass

        # Close the write ends (fd 1/2 currently pointing to pipe write ends); replace them with original fds
        try:
            # dup original fds back to 1 and 2
            os.dup2(self._orig_stdout_fd, 1)
            os.dup2(self._orig_stderr_fd, 2)
        except Exception:
            pass

        # allow threads to finish/exit
        time.sleep(0.05)

        # close reader fds and join threads
        try:
            if self._stdout_reader:
                os.close(self._stdout_pipe[0])
                self._stdout_reader.join(timeout=0.5)
        except Exception:
            pass
        try:
            if self._stderr_reader:
                os.close(self._stderr_pipe[0])
                self._stderr_reader.join(timeout=0.5)
        except Exception:
            pass

        # close saved duplicated fds
        try:
            if self._orig_stdout_fd:
                os.close(self._orig_stdout_fd)
        except Exception:
            pass
        try:
            if self._orig_stderr_fd:
                os.close(self._orig_stderr_fd)
        except Exception:
            pass

        # close log file
        try:
            if self._log_file:
                self._log_file.close()
        except Exception:
            pass
