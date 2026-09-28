"""Download and build Laya checkpoints at image build time, so workers never fetch weights.

Loading through `Router.preload` (on CPU, no GPU needed) rather than a bare snapshot_download
fills the Hugging Face cache with exactly the files the runtime asks for, and fails the build
if this torch/transformers combination cannot load a checkpoint.
"""
import os
import sys
import time

from laya import Router


def main() -> None:
    names = [n.strip() for n in os.environ.get("LAYA_BAKE_MODELS", "").split(",") if n.strip()]
    if not names:
        print("LAYA_BAKE_MODELS is empty; workers will download checkpoints on cold start")
        return
    t0 = time.perf_counter()
    router = Router(device="cpu").preload(names)
    probe = {"probe": {"type": "noul", "instructions": "Is this a build check?"}}
    for name in names:
        result = router.predict("build check", probe, model=name)
        print("baked %s (%s) -> %s" % (name, router.loaded_revisions.get(name), result["answers"]["probe"]["noul"]))
    print("baked %s in %.0fs" % (names, time.perf_counter() - t0))


if __name__ == "__main__":
    sys.exit(main())
