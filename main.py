"""主程序入口（向后兼容）：委托给 test.py。

所有命令行参数和用法与之前一致。
实际测试逻辑在 test.py 中，部署逻辑在 deploy.py 中。
"""

from __future__ import annotations

import sys

from test import run_test, main as test_main

# 向后兼容旧调用：main.run(config_path, ...)
run = run_test

if __name__ == "__main__":
    test_main()