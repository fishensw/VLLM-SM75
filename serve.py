#!/usr/bin/env python3
"""Run the tested Flash-Next TP8 configuration bundled with this image."""
import json,os,pathlib,sys
os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
config=json.loads(pathlib.Path('/opt/flashnext/selected-config.json').read_text())
os.execv(sys.executable,[sys.executable,'-m','vllm.entrypoints.openai.api_server',*config['args'],*sys.argv[1:]])
