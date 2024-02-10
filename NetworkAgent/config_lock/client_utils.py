import socket
from .message_utils import send_to


SPAWN_SERVER_MESSAGE = """

**********************************************************************************
The config locking server hasn't been started yet. It's necessary to ensure that
separate processes don't run into a lost-update problem and use duplicate configs.

You can start the config locking server in a separate terminal using:
python NetworkAgent/config_lock/server.py
**********************************************************************************
"""


def get_server_address() -> tuple[str, int]:
    local_IP = socket.gethostbyname(socket.gethostname())
    address = (local_IP, 4000)
    return address


def get_client_address(seed: int = 0) -> tuple[str, int]:
    local_IP = socket.gethostbyname(socket.gethostname())
    address = (local_IP, 4200 + seed)
    return address


def request_lock(seed: int) -> None:
    with socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) as client:
        client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        client_address = get_client_address(seed)
        client.bind(client_address)
        client.listen()
        try:
            send_to(
                get_server_address(), f"REQUEST {client_address[0]} {client_address[1]}"
            )
            server_socket, _ = client.accept()
            bytes_message = server_socket.recv(2048)
        except:
            raise Exception(SPAWN_SERVER_MESSAGE)
        message: str = bytes_message.decode("utf-8")
        print(f"Message '{message}' received.")
        if "GRANTED" not in message:
            raise Exception("Invalid message received from config lock server!")


def release_lock(seed: int) -> None:
    with socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) as client:
        client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        client_address = get_client_address(seed)
        client.bind(client_address)
        client.listen()
        send_to(
            get_server_address(), f"RELEASE {client_address[0]} {client_address[1]}"
        )
