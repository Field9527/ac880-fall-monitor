"""完成 HTTP 域名验证，再启用带登录保护的 HTTPS。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
import cache_env
from cloud_ssh import CloudSSH


def deploy(args):
    domain = args.domain.lower()
    if not re.fullmatch(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}", domain):
        raise ValueError("Invalid domain")
    source = Path(__file__).resolve().parents[1] / "cloud"
    args.output.mkdir(parents=True, exist_ok=True)
    client = CloudSSH(
        args.cloud_host,
        args.cloud_user,
        args.key,
        args.known_hosts,
        args.output / "public_provision.log",
        jump_host=args.jump_host,
        jump_user=args.jump_user,
        jump_key=args.jump_key,
    )
    stage = None
    root = "/opt/ac880-home-preview"
    try:
        client.run(
            "set -eu\nsudo -n true\ncommand -v nginx\n" "sudo test -s " + root + "/users.htpasswd"
        )
        if args.phase == "bootstrap":
            # HTTPS 启用后禁止重新引导，避免覆盖在线监听器。
            client.run(
                "if sudo test -f " + root + "/https-ready; then "
                "echo 'HTTPS is already active; use activate.' >&2; exit 1; fi"
            )
            print("Preparing certificate validation endpoint.", flush=True)
            client.run(
                "set -eu\n"
                "sudo env DEBIAN_FRONTEND=noninteractive apt-get update -qq\n"
                "sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends certbot\n"
                "sudo install -d -m 0755 " + root + "/acme",
                timeout=300,
            )
        else:
            check = (
                "import socket; values={x[4][0] for x in "
                "socket.getaddrinfo(" + repr(domain) + ",80,socket.AF_INET)}; "
                "print(sorted(values)); assert values=={" + repr(args.cloud_host) + "},values"
            )
            print("Checking domain DNS from the target server.", flush=True)
            print(client.run("python3 -c " + shlex.quote(check)), flush=True)
            cert_command = [
                "certbot",
                "certonly",
                "--webroot",
                "-w",
                root + "/acme",
                "-d",
                domain,
                "--cert-name",
                domain,
                "--non-interactive",
                "--agree-tos",
                "--keep-until-expiring",
            ]
            cert_command += (
                ["--email", args.email] if args.email else ["--register-unsafely-without-email"]
            )
            print(client.run("sudo " + shlex.join(cert_command), timeout=180), flush=True)
        template_name = (
            "nginx-challenge.conf.in" if args.phase == "bootstrap" else "nginx-public.conf.in"
        )
        config = (source / template_name).read_text(encoding="utf8").replace("__DOMAIN__", domain)
        stage = client.run('mktemp -d "$HOME/.ac880-public.XXXXXXXX"').strip()
        if not stage.startswith("/") or "\n" in stage:
            raise RuntimeError("Unexpected remote staging directory")
        client.put_bytes(stage + "/public.conf", config.encode("utf8"))
        client.put_file(source / "ac880-home-public.service", stage + "/public.service")
        if args.phase == "activate":
            client.put_file(source / "video_relay.py", stage + "/video_relay.py")
            client.put_file(source / "ac880-video-relay.service", stage + "/relay.service")
        # 先验证暂存配置，失败时保留当前监听器。
        client.run("sudo nginx -t -c " + shlex.quote(stage + "/public.conf"))
        relay_install = ""
        if args.phase == "activate":
            relay_install = (
                "sudo install -o root -g root -m 0644 "
                + shlex.quote(stage + "/video_relay.py")
                + " "
                + root
                + "/video_relay.py\n"
                "sudo install -o root -g root -m 0644 "
                + shlex.quote(stage + "/relay.service")
                + " /etc/systemd/system/ac880-video-relay.service\n"
            )
        install = (
            "set -eu\n"
            + relay_install
            + "sudo install -o root -g root -m 0644 "
            + shlex.quote(stage + "/public.conf")
            + " "
            + root
            + "/public.conf\n"
            "sudo install -o root -g root -m 0644 "
            + shlex.quote(stage + "/public.service")
            + " /etc/systemd/system/ac880-home-public.service\n"
            "sudo systemctl daemon-reload\n"
            + (
                "sudo systemctl enable ac880-video-relay.service\nsudo systemctl restart ac880-video-relay.service\nsudo systemctl is-active ac880-video-relay.service\n"
                if args.phase == "activate"
                else ""
            )
            + "sudo systemctl restart ac880-home-public.service\n"
            "sudo systemctl is-active ac880-home-public.service\n"
            "ss -lnt"
        )
        print(client.run(install), flush=True)
        if args.phase == "activate":
            hook = "#!/bin/sh\nset -eu\nsystemctl reload ac880-home-public.service\n"
            client.put_bytes(stage + "/renew-hook", hook.encode("utf8"))
            client.run(
                "set -eu\n"
                "sudo install -d -m 0755 /etc/letsencrypt/renewal-hooks/deploy\n"
                "sudo install -o root -g root -m 0755 "
                + shlex.quote(stage + "/renew-hook")
                + " /etc/letsencrypt/renewal-hooks/deploy/ac880-home-reload\n"
                "sudo touch " + root + "/https-ready\n"
                "sudo systemctl enable ac880-home-public.service\n"
                "sudo systemctl enable --now certbot.timer"
            )
        receipt = dict(
            domain=domain,
            cloud_host=args.cloud_host,
            cloud_user=args.cloud_user,
            jump_host=args.jump_host,
            phase=args.phase,
            https_configured=args.phase == "activate",
            live_video_verified=False,
            mobile_network_verified=False,
            config_sha256=hashlib.sha256(config.encode("utf8")).hexdigest(),
        )
        (args.output / ("public_" + args.phase + "_receipt.json")).write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf8"
        )
    finally:
        if stage:
            names = (
                "public.conf",
                "public.service",
                "renew-hook",
                "video_relay.py",
                "relay.service",
            )
            try:
                client.run(
                    "rm -f -- "
                    + " ".join(shlex.quote(stage + "/" + name) for name in names)
                    + "\nrmdir -- "
                    + shlex.quote(stage)
                )
            except Exception:
                print("Remote staging cleanup failed; inspect the log.", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("bootstrap", "activate"))
    parser.add_argument("--domain", default="shuaidao.online")
    parser.add_argument("--cloud-host", required=True)
    parser.add_argument("--cloud-user", default="root")
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, default=Path("E:/Zynq_keys/aliyun/known_hosts"))
    parser.add_argument("--jump-host")
    parser.add_argument("--jump-user", default="ecs-user")
    parser.add_argument("--jump-key", type=Path, default=Path("E:/Zynq_keys/aliyun/ac880_home.pem"))
    parser.add_argument("--email")
    parser.add_argument("--output", type=Path, required=True)
    deploy(parser.parse_args())
