import socket
import threading
import time
import random

DEFAULT_HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
DEFAULT_PORT = 9999        # Port to listen on

class EmotionSocketStreamer:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = []
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def _handle_client(self, client_socket, client_address):
        print(f"[Socket Streamer] Accepted connection from {client_address}")
        with self.lock:
            self.clients.append(client_socket)
        
        try:
            while self.running:
                # Keep connection alive, or handle client-side messages if needed
                # For this use case, we primarily broadcast, so we might not need to receive much
                # Adding a small delay to prevent tight loop if we were to add receiving logic
                time.sleep(0.1)
                # Check if client is still connected by trying to send a small keep-alive or by other means
                # A simple way is to catch errors during send_emotion
        except ConnectionResetError:
            print(f"[Socket Streamer] Client {client_address} disconnected (reset).")
        except BrokenPipeError:
            print(f"[Socket Streamer] Client {client_address} disconnected (broken pipe).")
        except Exception as e:
            print(f"[Socket Streamer] Error with client {client_address}: {e}")
        finally:
            with self.lock:
                if client_socket in self.clients:
                    self.clients.remove(client_socket)
            client_socket.close()
            print(f"[Socket Streamer] Connection with {client_address} closed.")

    def start_server(self):
        if self.running:
            print("[Socket Streamer] Server is already running.")
            return

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            self.thread = threading.Thread(target=self._accept_connections, daemon=True)
            self.thread.start()
            print(f"[Socket Streamer] Server started on {self.host}:{self.port}")
        except OSError as e:
            print(f"[Socket Streamer] Error starting server: {e}. Port {self.port} might be in use.")
            self.server_socket = None
            self.running = False

    def _accept_connections(self):
        while self.running:
            try:
                client_socket, client_address = self.server_socket.accept()
                client_thread = threading.Thread(target=self._handle_client, args=(client_socket, client_address), daemon=True)
                client_thread.start()
            except OSError: # server_socket might be closed
                if self.running: # If still supposed to be running, it's an unexpected error
                    print("[Socket Streamer] Error accepting connections, server might be closing.")
                break # Exit loop if socket is closed
            except Exception as e:
                if self.running:
                    print(f"[Socket Streamer] Unexpected error in accept_connections: {e}")
                break

    def send_emotion(self, emotion_data: str):
        if not self.running or not self.clients:
            return

        print(f"[Socket Streamer] Sending emotion: {emotion_data}")
        message = (emotion_data + '\n').encode('utf-8')
        with self.lock:
            # Iterate over a copy of the list in case of modifications during iteration
            for client_socket in list(self.clients):
                try:
                    client_socket.sendall(message)
                    print(f"[Socket Streamer] Sent emotion to client: {emotion_data}")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    print(f"[Socket Streamer] Client disconnected during send. Removing client.")
                    if client_socket in self.clients:
                         self.clients.remove(client_socket)
                    client_socket.close()
                except Exception as e:
                    print(f"[Socket Streamer] Error sending data to a client: {e}")
                    # Optionally remove problematic client
                    if client_socket in self.clients:
                         self.clients.remove(client_socket)
                    client_socket.close()

    def stop_server(self):
        print("[Socket Streamer] Stopping server...")
        self.running = False
        # Close all client sockets
        with self.lock:
            for client_socket in self.clients:
                try:
                    client_socket.shutdown(socket.SHUT_RDWR) # Gracefully shutdown
                    client_socket.close()
                except OSError:
                    pass # Socket might already be closed
            self.clients.clear()
        
        # Close the server socket
        if self.server_socket:
            try:
                # Unblock the accept() call by connecting to it briefly
                # This is a common workaround for blocking socket.accept()
                if self.host and self.port:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.1) # Don't block for long
                        try:
                            s.connect((self.host, self.port))
                        except (socket.timeout, ConnectionRefusedError):
                            pass # Expected if server is already shutting down or nothing is listening
                self.server_socket.close()
            except OSError as e:
                print(f"[Socket Streamer] Error closing server socket: {e}")
            self.server_socket = None
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0) # Wait for the server thread to finish
        
        print("[Socket Streamer] Server stopped.")

if __name__ == '__main__':
    print("Testing EmotionSocketStreamer...")
    streamer = EmotionSocketStreamer()
    streamer.start_server()

    if streamer.running:
        print("Server is running. Sending random emotions every millisecond. Press Ctrl+C to stop.")
        # Define a list of emotions
        EMOTIONS_LIST = ["happy", "sad", "angry", "surprised", "neutral", "fear", "disgust"]
        try:
            while True:
                current_emotion = random.choice(EMOTIONS_LIST)
                persistence_duration = random.uniform(0.5, 3.0)  # Duration for this emotion
                # The user had print(f"Main: Sending: {emotion}") uncommented, so let's adapt it.
                # This print will now show when a new emotion is CHOSEN, not every send.
                print(f"Main: New emotion chosen: '{current_emotion}'. Will send repeatedly for {persistence_duration:.2f}s.")

                start_time = time.time()
                while time.time() - start_time < persistence_duration:
                    streamer.send_emotion(current_emotion)  # Send the same emotion repeatedly
                    time.sleep(0.001)  # Send data (approximately) every millisecond
                    # Allow graceful exit if server stops during this tight loop
                    if not streamer.running:
                        break
                
                if not streamer.running: # If server stopped, break outer loop too
                    break
        except KeyboardInterrupt:
            print("\nTest interrupted by user.") # Added newline for cleaner exit
        finally:
            streamer.stop_server()
    else:
        print("Server failed to start. Exiting test.")

    print("Socket streamer test finished.")

