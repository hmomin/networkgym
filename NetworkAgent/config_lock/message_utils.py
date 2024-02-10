import socket
from threading import Thread
from typing import Callable


def send_to(
    address: tuple[str, int],
    message_str: str,
) -> None:
    bytes_message = message_str.encode("utf-8")
    with socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) as dummy_source:
        dummy_source.connect(address)
        dummy_source.sendall(bytes_message)
    print(f"Message '{message_str}' sent to {address}.")


def general_connection_handler(
    in_socket: socket.socket,
    specific_message_handler: Callable[[str, socket.socket], None],
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
                specific_message_handler,
                client_socket,
            ),
            daemon=True,
        ).start()


def general_message_handler(
    bytes_message: bytes,
    specific_message_handler: Callable[[str, socket.socket], None],
    client_socket: socket.socket,
) -> None:
    try:
        message = bytes_message.decode("utf-8")
        print(f"Message '{message}' received.")
        specific_message_handler(message, client_socket)
    except Exception as e:
        print("WARNING: exception occurred while trying to read bytes_message")
        print(e)
