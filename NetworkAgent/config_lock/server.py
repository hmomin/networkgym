import socket
from threading import Thread
from message_utils import general_connection_handler, send_to

SERVER_PORT = 4000
LOCK_REQUESTS: list[tuple[str, int]] = []


def get_server_address() -> tuple[str, int]:
    local_IP = socket.gethostbyname(socket.gethostname())
    address = (local_IP, SERVER_PORT)
    return address


def initialize_server(server: socket.socket) -> None:
    address = get_server_address()
    server.bind(address)
    server.listen()
    print(f"Config locking server listening on port {SERVER_PORT}...")


def get_return_address_from_message(message_str: str) -> tuple[str, int]:
    message_components = message_str.split()
    assert len(message_components) == 3
    local_IP = message_components[1]
    port = int(message_components[2])
    return (local_IP, port)


def handle_message_for_server(
    message: str,
    client_socket: socket.socket,
):
    if "REQUEST" in message:
        return_address = get_return_address_from_message(message)
        handle_lock_request(return_address)
    elif "RELEASE" in message:
        return_address = get_return_address_from_message(message)
        handle_lock_release(return_address)
    else:
        print(f"Invalid message sent: {message}")
    client_socket.close()


def handle_lock_request(return_address: tuple[str, int]) -> None:
    LOCK_REQUESTS.append(return_address)
    if len(LOCK_REQUESTS) == 1:
        grant_lock(return_address)


def grant_lock(return_address: tuple[str, int]) -> None:
    server_address = get_server_address()
    send_to(return_address, f"GRANTED {server_address[0]} {server_address[1]}")


def handle_lock_release(return_address: tuple[str, int]) -> None:
    if len(LOCK_REQUESTS) > 0 and LOCK_REQUESTS[0] == return_address:
        LOCK_REQUESTS.pop(0)
    else:
        print(
            f"WARNING: client {return_address} releasing lock is not at the top of the queue."
        )
        print("LOCK_REQUESTS:")
        print(LOCK_REQUESTS)
    if len(LOCK_REQUESTS) > 0:
        grant_lock(LOCK_REQUESTS[0])


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
