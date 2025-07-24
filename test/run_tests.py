#!/usr/bin/env python3
"""
Script to run the server and tests
"""

import subprocess
import time
import sys
import os
import signal


def run_server_and_tests():
    """Start the server and run tests"""
    server_process = None

    try:
        # Start the server in background
        print("Starting server...")
        server_process = subprocess.Popen(
            [sys.executable, "../main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start
        time.sleep(5)

        # Check if server is running
        if server_process.poll() is None:
            print("Server started successfully")

            # Run tests
            print("Running tests...")
            test_process = subprocess.run(
                [sys.executable, "test_server.py"], capture_output=True, text=True
            )

            print(test_process.stdout)
            if test_process.stderr:
                print("Test errors:", test_process.stderr)

        else:
            print("Server failed to start")
            stdout, stderr = server_process.communicate()
            print("Server stdout:", stdout.decode())
            print("Server stderr:", stderr.decode())

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # Stop the server
        if server_process and server_process.poll() is None:
            print("Stopping server...")
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    run_server_and_tests()
