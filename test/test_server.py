#!/usr/bin/env python3
"""
Test script for the Voice Agent VAPI server
"""

import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_chat_completion_non_streaming():
    """Test non-streaming chat completion"""
    url = "http://localhost:8000/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('API_TOKEN', 'your_api_token_here')}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Non-streaming chat: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result['choices'][0]['message']['content']}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Non-streaming chat failed: {e}")
        return False

def test_chat_completion_streaming():
    """Test streaming chat completion"""
    url = "http://localhost:8000/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('API_TOKEN', 'your_api_token_here')}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Tell me a short joke"}
        ],
        "stream": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        print(f"Streaming chat: {response.status_code}")
        
        if response.status_code == 200:
            print("Streaming response:")
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_part = line[6:]
                        if data_part == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_part)
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    print(content, end='', flush=True)
                        except json.JSONDecodeError:
                            continue
            print("\n")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Streaming chat failed: {e}")
        return False

def test_transfer_call():
    """Test transfer call functionality"""
    url = "http://localhost:8000/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('API_TOKEN', 'your_api_token_here')}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "I want to speak to a human agent"}
        ],
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Transfer call test: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result['choices'][0]['message']['content']}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Transfer call test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Voice Agent VAPI server...")
    print("=" * 50)
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(2)
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Non-streaming Chat", test_chat_completion_non_streaming),
        ("Streaming Chat", test_chat_completion_streaming),
        ("Transfer Call", test_transfer_call)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("Test Results:")
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")