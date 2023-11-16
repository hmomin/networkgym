import pickle
import socket
from threading import Thread
from typing import Any, Callable


def send_to(
    address: tuple[str, int],
    data: Any,
    source_socket: socket.socket,
) -> None:
    client_address = source_socket.getsockname()
    message_dict: dict[str, Any] = {"address": client_address, "data": data}
    bytes_data = pickle.dumps(message_dict)
    with socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) as dummy_source:
        dummy_source.connect(address)
        dummy_source.sendall(bytes_data)
    print(f"Message '{data}' sent to {address}.")


def general_connection_handler(
    in_socket: socket.socket,
    specific_message_handler: Callable[
        [socket.socket, Any, tuple[str, int], socket.socket], None
    ],
) -> None:
    while True:
        client_socket, _ = in_socket.accept()
        bytes_message = client_socket.recv(2048)
        # new thread handles message while current thread accepts incoming
        # connections
        Thread(
            target=general_message_handler,
            args=(
                bytes_message,
                in_socket,
                specific_message_handler,
                client_socket,
            ),
            daemon=True,
        ).start()


def general_message_handler(
    bytes_message: bytes,
    in_socket: socket.socket,
    specific_message_handler: Callable[
        [socket.socket, Any, tuple[str, int], socket.socket], None
    ],
    client_socket: socket.socket,
) -> None:
    try:
        message_dict: dict[str, Any] = pickle.loads(bytes_message)
        return_address: tuple[str, int] = message_dict["address"]
        message: str = message_dict["data"]
        print(f"Message '{message}' received from {return_address}.")
        specific_message_handler(in_socket, message, return_address, client_socket)
    except Exception as e:
        print("WARNING: exception occurred while trying to read bytes_message")
        print(e)