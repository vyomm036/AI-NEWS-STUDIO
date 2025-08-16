#!/usr/bin/env python3
"""
Test to simulate the web interface script-to-audio conversion
"""

import requests
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_web_interface_audio():
    """Test the web interface audio generation"""
    print("🌐 Testing Web Interface Audio Generation...")
    
    # Test script (similar to what would be generated)
    test_script = """
    Breaking news today as we cover the latest developments in technology. 
    The industry has seen remarkable advancements in artificial intelligence and machine learning. 
    Companies worldwide are investing heavily in these technologies, driving innovation across sectors. 
    Experts predict this trend will continue to accelerate in the coming years.
    """
    
    # Simulate the exact web interface request
    url = "http://127.0.0.1:5000/generate_audio"
    data = {
        'script': test_script,
        'voice': 'en-US-JennyNeural'
    }
    
    try:
        print(f"📝 Script length: {len(test_script)} characters")
        print(f"🎤 Voice: {data['voice']}")
        print(f"🌐 Sending request to: {url}")
        
        # Send POST request (same as web interface)
        response = requests.post(url, data=data)
        
        print(f"📡 Response status: {response.status_code}")
        print(f"📄 Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Response data: {result}")
            
            if result.get('success'):
                audio_path = result.get('audio_path')
                print(f"✅ Audio generated successfully!")
                print(f"🎵 Audio path: {audio_path}")
                
                # Check if file exists
                if audio_path and audio_path.startswith('/static/'):
                    file_path = audio_path[1:]  # Remove leading slash
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        print(f"📊 File size: {file_size} bytes")
                        print(f"📍 File location: {os.path.abspath(file_path)}")
                        return True
                    else:
                        print(f"❌ Audio file not found at: {file_path}")
                        return False
                else:
                    print(f"❌ Invalid audio path: {audio_path}")
                    return False
            else:
                print(f"❌ Error in response: {result.get('error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"📄 Response text: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_flask_function():
    """Test the Flask function directly without HTTP"""
    print("\n🔧 Testing Flask Function Directly...")
    
    try:
        # Import the Flask app function
        from app import generate_audio
        from flask import Flask, request
        from werkzeug.test import EnvironBuilder
        from werkzeug.wrappers import Request
        
        # Create a test Flask app
        test_app = Flask(__name__)
        
        # Test data
        test_script = "This is a test script for direct Flask function testing."
        
        # Create a mock request
        builder = EnvironBuilder(
            method='POST',
            data={'script': test_script, 'voice': 'en-US-JennyNeural'},
            content_type='application/x-www-form-urlencoded'
        )
        env = builder.get_environ()
        req = Request(env)
        
        # Test the function
        with test_app.test_request_context(req):
            result = generate_audio()
            
            if result.status_code == 200:
                data = result.get_json()
                if data.get('success'):
                    print("✅ Direct Flask function test passed")
                    return True
                else:
                    print(f"❌ Direct Flask function error: {data.get('error')}")
                    return False
            else:
                print(f"❌ Direct Flask function HTTP error: {result.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Direct Flask function exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Web Interface Audio Generation Test...")
    
    # Test 1: Direct Flask function
    test1 = test_direct_flask_function()
    
    # Test 2: Web interface simulation (requires Flask app running)
    print("\n⚠️  Note: Make sure Flask app is running on http://127.0.0.1:5000/")
    test2 = test_web_interface_audio()
    
    print(f"\n📊 Test Results:")
    print(f"Direct Flask function: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Web interface simulation: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 and test2:
        print("\n🎉 All tests passed! Web interface audio generation is working.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.") 