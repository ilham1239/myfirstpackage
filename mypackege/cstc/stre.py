import asyncio
import cv2
import pyautogui
import numpy as np
import pickle
import struct

class AsyncStreamingServer:
    def __init__(self, host, port, slots=8, quit_key='q'):
        self.host = host
        self.port = port
        self.slots = slots
        self.quit_key = quit_key
        self.used_slots = 0
        self.running = False
        self.server = None

    async def start_server(self):
        if self.running:
            print("Server is already running")
            return

        self.running = True
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        print(f"Server started on {self.host}:{self.port}")

        async with self.server:
            await self.server.serve_forever()

    async def stop_server(self):
        if not self.running:
            print("Server is not running")
            return

        self.running = False
        self.server.close()
        await self.server.wait_closed()
        print("Server stopped")

    async def handle_client(self, reader, writer):
        if self.used_slots >= self.slots:
            print("Connection refused! No free slots!")
            writer.close()
            await writer.wait_closed()
            return

        self.used_slots += 1
        address = writer.get_extra_info('peername')
        print(f"New connection from {address}")

        payload_size = struct.calcsize('>L')
        data = b""

        try:
            while self.running:
                while len(data) < payload_size:
                    packet = await reader.read(4096)
                    if not packet:
                        raise ConnectionResetError
                    data += packet

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]

                msg_size = struct.unpack('>L', packed_msg_size)[0]

                while len(data) < msg_size:
                    data += await reader.read(4096)

                frame_data = data[:msg_size]
                data = data[msg_size:]

                frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                cv2.imshow(f"Stream from {address}", frame)
                if cv2.waitKey(1) == ord(self.quit_key):
                    break

        except (ConnectionResetError, asyncio.IncompleteReadError):
            print(f"Connection lost with {address}")

        finally:
            self.used_slots -= 1
            writer.close()
            await writer.wait_closed()
            cv2.destroyWindow(f"Stream from {address}")


class AsyncStreamingClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.running = False
        self.encoding_parameters = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    async def start_stream(self):
        if self.running:
            print("Client is already streaming")
            return

        self.running = True
        reader, writer = await asyncio.open_connection(self.host, self.port)

        try:
            while self.running:
                frame = self.get_frame()
                if frame is None:
                    break

                _, encoded_frame = cv2.imencode('.jpg', frame, self.encoding_parameters)
                data = pickle.dumps(encoded_frame, 0)
                size = len(data)
                writer.write(struct.pack('>L', size) + data)
                await writer.drain()

        except (ConnectionResetError, BrokenPipeError):
            print("Connection to server lost")

        finally:
            writer.close()
            await writer.wait_closed()
            self.cleanup()

    def get_frame(self):
        """Override in subclasses."""
        return None

    def cleanup(self):
        cv2.destroyAllWindows()


class AsyncCameraClient(AsyncStreamingClient):
    def __init__(self, host, port, x_res=1024, y_res=576):
        super().__init__(host, port)
        self.camera = cv2.VideoCapture(0)
        self.x_res = x_res
        self.y_res = y_res
        self.camera.set(3, self.x_res)
        self.camera.set(4, self.y_res)

    def get_frame(self):
        ret, frame = self.camera.read()
        return frame

    def cleanup(self):
        self.camera.release()
        super().cleanup()


class AsyncScreenShareClient(AsyncStreamingClient):
    def __init__(self, host, port, x_res=1024, y_res=576):
        super().__init__(host, port)
        self.x_res = x_res
        self.y_res = y_res

    def get_frame(self):
        screen = pyautogui.screenshot()
        frame = np.array(screen)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.x_res, self.y_res), interpolation=cv2.INTER_AREA)
        return frame

# Example usage:
# server = AsyncStreamingServer('localhost', 9999)
# asyncio.run(server.start_server())

# client = AsyncCameraClient('localhost', 9999)
# asyncio.run(client.start_stream())
