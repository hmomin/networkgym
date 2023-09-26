import pickle
import socket
from .message_utils import send_to


def get_server_address() -> tuple[str, int]:
    local_IP = socket.gethostbyname(socket.gethostname())
    address = (local_IP, 4000)
    return address


def get_client_address(seed: int = 0) -> tuple[str, int]:
    local_IP = socket.gethostbyname(socket.gethostname())
    address = (local_IP, 4200 + seed)
    return address


def request_lock(seed: int) -> None:
    with socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM) as client:
        client.bind(get_client_address(seed))
        send_to(get_server_address(), "REQUEST", client)
        bytes_message, return_address = client.recvfrom(2048)
        message = pickle.loads(bytes_message)
        print(f"Message '{message}' received from {return_address}.")
        if message != "GRANTED":
            raise Exception(
                "Invalid message received from config lock server!"
            )


def release_lock(seed: int) -> None:
    with socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM) as client:
        client.bind(get_client_address(seed))
        send_to(get_server_address(), "RELEASE", client)
