"""云端部署共用系统 OpenSSH 客户端，避免依赖旧版 Paramiko 的 RSA 签名支持；继续固定主机密钥。"""

from pathlib import Path
import os
import re
import shlex
import shutil
import subprocess


class CloudSSH:
    def __init__(
        self, host, user, key, known_hosts, log, jump_host=None, jump_user="ecs-user", jump_key=None
    ):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host):
            raise ValueError("Invalid cloud hostname")
        if not re.fullmatch(r"[a-z_][a-z0-9_-]*", user):
            raise ValueError("Invalid SSH username")
        for path in (key, known_hosts):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        executable = shutil.which("ssh.exe" if os.name == "nt" else "ssh")
        if not executable:
            raise RuntimeError("OpenSSH client is not installed")
        self.command = [
            executable,
            "-T",
            "-i",
            str(key),
            "-o",
            "UserKnownHostsFile=" + Path(known_hosts).resolve().as_posix(),
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            user + "@" + host,
        ]
        self.log = Path(log)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        if jump_host:
            jumper = CloudSSH(jump_host, jump_user, jump_key or key, known_hosts, log)
            proxy = jumper.command[:-1] + ["-W", "%h:%p", jumper.command[-1]]
            proxy_command = subprocess.list2cmdline(proxy) if os.name == "nt" else shlex.join(proxy)
            self.command[-1:-1] = ["-o", "ProxyCommand=" + proxy_command]

    def run(self, command, timeout=180, data=None):
        result = subprocess.run(
            self.command + [command],
            input=data,
            capture_output=True,
            timeout=timeout,
        )
        output = result.stdout.decode("utf8", errors="replace")
        errors = result.stderr.decode("utf8", errors="replace")
        with self.log.open("a", encoding="utf8") as stream:
            stream.write(output + errors + "\n")
        if result.returncode:
            raise RuntimeError("Cloud command failed: " + (errors + output)[-1800:])
        return output

    def put_bytes(self, destination, data, mode=0o600):
        path = shlex.quote(destination)
        self.run(
            "set -eu\numask 077\ncat > " + path + "\nchmod " + format(mode, "o") + " " + path,
            data=data,
        )

    def put_file(self, source, destination):
        self.put_bytes(destination, Path(source).read_bytes())
