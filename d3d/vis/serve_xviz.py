import sys, os, logging
import asyncio, json

import xviz_avs
from xviz_avs.builder import XVIZBuilder, XVIZMetadataBuilder
from xviz_avs.server import XVIZServer, XVIZBaseSession


class ScenarioSession(XVIZBaseSession):
    def __init__(self, socket, request):
        super().__init__(socket, request)
        self._socket = socket

    def on_connect(self):
        print("Connected!")

    def on_disconnect(self):
        print("Disconnect!")

    async def main(self):
        with open("/tmp/mykitti/1-frame.glb", "rb") as fin:
            await self._socket.send(fin.read())

        n = 2
        while n < 100:
            with open("/tmp/mykitti/%d-frame.glb" % n, "rb") as fin:
                await self._socket.send(fin.read())

            await asyncio.sleep(0.1)

class ScenarioHandler:
    def __init__(self):
        pass

    def __call__(self, socket, request):
        return ScenarioSession(socket, request)

if __name__ == "__main__":
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logging.getLogger("xviz-server").addHandler(handler)

    server = XVIZServer(ScenarioHandler(), port=8081)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(server.serve())
    loop.run_forever()
