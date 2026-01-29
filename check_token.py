#!/usr/bin/env python3
"""Test LLM API connection"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_connection():
    api_key = os.getenv("CUSTOM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL_NAME")
    
    print("=" * 60)
    print("Testing LLM API Connection")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print(f"API Key (first 10 chars): {api_key[:10] if api_key else 'NOT SET'}...")
    print()
    
    # Test 1: Check models endpoint
    print("Test 1: Checking /models endpoint...")
    try:
        response = requests.get(
            f"{base_url}/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=10
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            models = [m.get('id') for m in data.get('data', [])]
            print(f"✅ Available models: {len(models)}")
            for m in models[:5]:
                print(f"  - {m}")
            if len(models) > 5:
                print(f"  ... and {len(models) - 5} more")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    
    print()
    
    # Test 2: Simple chat completion
    print("Test 2: Testing chat completion...")
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": "สวัสดี"}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            },
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            message = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"✅ Response: {message}")
        else:
            print(f"❌ Error: {response.text}")
            
            # Parse error details
            try:
                error_data = response.json()
                print(f"Error Name: {error_data.get('name')}")
                print(f"Error Message: {error_data.get('message')}")
            except:
                pass
                
    except requests.exceptions.Timeout:
        print("❌ Request timed out (60s)")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - cannot reach server")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    print("=" * 60)

if _name_ == "_main_":
    test_connection()