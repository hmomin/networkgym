import pickle
import socket
from typing import Any, Callable


def send_to(
    address: tuple[str, int],
    data: Any,
    source_socket: socket.socket,
) -> None:
    bytes_data = pickle.dumps(data)
    source_socket.sendto(bytes_data, address)
    print(f"Message '{data}' sent to {address}.")


def general_message_handler(
    in_socket: socket.socket,
    specific_message_handler: Callable[[socket.socket, Any, tuple[str, int]], None],
) -> None:
    while True:
        bytes_message, return_address = in_socket.recvfrom(2048)
        message = pickle.loads(bytes_message)
        print(f"Message '{message}' received from {return_address}.")
        specific_message_handler(in_socket, message, return_address)
