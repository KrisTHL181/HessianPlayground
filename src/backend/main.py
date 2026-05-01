"""Entry point for Hessian Playground server.

Run with: python src/backend/main.py [--host HOST] [--port PORT]
"""

import argparse
import asyncio
import os
import signal
import sys

# Add src/ to sys.path for absolute imports of the backend package
_src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from aiohttp import web

from backend.config import DEFAULT_HOST, DEFAULT_PORT


def main():
    parser = argparse.ArgumentParser(description="Hessian Playground Server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    from backend.server import create_app

    app = create_app()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    runner = None

    def shutdown():
        async def _shutdown():
            if runner is not None:
                await runner.cleanup()
            loop.stop()

        loop.create_task(_shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            signal.signal(sig, lambda s, f: shutdown())

    async def run():
        nonlocal runner
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, args.host, args.port)
        await site.start()
        print(f"Hessian Playground running at http://{args.host}:{args.port}")

    try:
        loop.run_until_complete(run())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


if __name__ == "__main__":
    main()
