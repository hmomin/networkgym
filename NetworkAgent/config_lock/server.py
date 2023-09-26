import socket
from threading import Thread
from typing import Any
from message_utils import general_connection_handler, send_to

SERVER_PORT = 4000
LOCK_REQUESTS: list[tuple[str, int]] = []


def initialize_server(server: socket.socket) -> None:
    local_IP = socket.gethostbyname(socket.gethostname())
    address = (local_IP, SERVER_PORT)
    server.bind(address)
    server.listen()
    print(f"Config locking server listening on port {SERVER_PORT}...")


def handle_message_for_server(
    server: socket.socket,
    message: Any,
    return_address: tuple[str, int],
    client_socket: socket.socket,
):
    if type(message) == str and "REQUEST" in message:
        handle_lock_request(server, return_address)
    elif type(message) == str and "RELEASE" in message:
        handle_lock_release(server, return_address)
    else:
        print("Invalid request sent.")
    client_socket.close()


def handle_lock_request(server: socket.socket, return_address: tuple[str, int]) -> None:
    LOCK_REQUESTS.append(return_address)
    if len(LOCK_REQUESTS) == 1:
        grant_lock(server, return_address)


def grant_lock(server: socket.socket, return_address: tuple[str, int]) -> None:
    send_to(return_address, "GRANTED", server)


def handle_lock_release(server: socket.socket, return_address: tuple[str, int]) -> None:
    if len(LOCK_REQUESTS) > 0 and LOCK_REQUESTS[0] == return_address:
        LOCK_REQUESTS.pop(0)
    else:
        print(
            f"WARNING: client releasing lock {return_address} is not at the top of the queue."
        )
        print("LOCK_REQUESTS:")
        print(LOCK_REQUESTS)
    if len(LOCK_REQUESTS) > 0:
        grant_lock(server, LOCK_REQUESTS[0])


def should_server_close(user_input: str) -> bool:
    user_input = user_input.lower()
    return len(user_input) != 0 and user_input[0] == "q"


def main() -> None:
    with socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) as server:
        initialize_server(server)
        # daemon thread handles messages while main thread handles user input
        Thread(
            target=general_connection_handler,
            args=(server, handle_message_for_server),
            daemon=True,
        ).start()
        server_closed = False
        while not server_closed:
            user_input = input()
            server_closed = should_server_close(user_input)


if __name__ == "__main__":
    main()
