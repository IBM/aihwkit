#!/bin/bash
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."

wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit \
   --toolkitpath="$PWD/.cuda" --no-man-page --override
rm cuda_12.8.1_570.124.06_linux.run

python3 - <<'PY'
from pathlib import Path

p = Path(".cuda/targets/x86_64-linux/include/crt/math_functions.h")
s = p.read_text()
p.with_suffix(p.suffix + ".bak").write_text(s)

for name in ("cospi", "sinpi", "cospif", "sinpif", "rsqrt", "rsqrtf"):
    s = s.replace(name + "(", name + "(")  # Locate declarations below.
    lines = s.splitlines()
    for i, line in enumerate(lines):
        if name + "(" in line and "noexcept" not in line and line.rstrip().endswith(";"):
            lines[i] = line.rstrip()[:-1] + " noexcept (true);"
            break
    s = "\n".join(lines) + "\n"

p.write_text(s)
PY
