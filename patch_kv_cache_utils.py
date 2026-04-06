"""Patch vLLM's kv_cache_utils.py to handle non-divisible page sizes.

Gemma 4 + TurboQuant produces heterogeneous page sizes (e.g. 16896 vs
8320 bytes) that aren't evenly divisible. The original code raises
NotImplementedError. This patch replaces that with page_size_padded
padding — vLLM's own mechanism for uniform page allocation.
"""

import importlib
import os
import re
import sys


def main():
    import vllm.v1.core.kv_cache_utils as m

    src_path = m.__file__
    assert src_path.endswith(".py"), f"Expected .py, got {src_path}"

    with open(src_path) as f:
        src = f.read()

    # Match the raise NotImplementedError block with flexible whitespace
    pattern = re.compile(
        r'([ \t]+)if max_page_size % layer_page_size != 0:\s*\n'
        r'\1[ \t]+raise NotImplementedError\(\s*\n'
        r'(?:\1[ \t]+.*\n)*?'  # continuation lines
        r'\1[ \t]+\)',
        re.MULTILINE,
    )

    match = pattern.search(src)
    if not match:
        print("WARNING: Could not find patch target in", src_path)
        print("Searching for 'raise NotImplementedError' near 'max_page_size'...")
        for i, line in enumerate(src.splitlines(), 1):
            if "NotImplementedError" in line and i > 900 and i < 960:
                print(f"  Line {i}: {line}")
        sys.exit(1)

    indent = match.group(1)
    replacement = (
        f"{indent}if max_page_size % layer_page_size != 0:\n"
        f"{indent}    # Padded for heterogeneous models (Gemma 4 + TQ4)\n"
        f"{indent}    new_spec = replace(layer_spec, page_size_padded=max_page_size)\n"
        f"{indent}    new_kv_cache_spec[layer_name] = new_spec\n"
        f"{indent}    continue"
    )

    new_src = src[:match.start()] + replacement + src[match.end():]

    with open(src_path, "w") as f:
        f.write(new_src)

    # Delete bytecache so Python uses the patched .py
    pyc_path = importlib.util.cache_from_source(src_path)
    if os.path.exists(pyc_path):
        os.remove(pyc_path)
        print(f"Removed bytecache: {pyc_path}")

    print(f"Patched: {src_path} (line {src[:match.start()].count(chr(10)) + 1})")


if __name__ == "__main__":
    main()
