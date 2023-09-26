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
        if message != "GRANTED":
            raise Exception(
                f"ERROR: message '{message}' received from config lock server at {return_address}."
            )


def release_lock(seed: int) -> None:
    with socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM) as client:
        client.bind(get_client_address(seed))
        send_to(get_server_address(), "RELEASE", client)
