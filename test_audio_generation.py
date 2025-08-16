#!/usr/bin/env python3
"""
Comprehensive test for audio generation functionality
"""

import asyncio
import os
import sys
import edge_tts

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the functions from app.py
from app import convert_text_to_speech, AUDIO_FILE

def test_audio_generation():
    """Test the complete audio generation process"""
    print("🎵 Testing Audio Generation...")
    
    # Test script
    test_script = "Hello, this is a test of the audio generation functionality. The system should create an audio file successfully."
    
    try:
        # Ensure output directory exists
        os.makedirs('static/output', exist_ok=True)
        
        print(f"📁 Output directory: {os.path.abspath('static/output')}")
        print(f"🎯 Target file: {os.path.abspath(AUDIO_FILE)}")
        print(f"📝 Script length: {len(test_script)} characters")
        
        # Test the conversion function
        convert_text_to_speech(test_script, AUDIO_FILE, "en-US-JennyNeural")
        
        # Check if file was created
        if os.path.exists(AUDIO_FILE):
            file_size = os.path.getsize(AUDIO_FILE)
            print(f"✅ Audio file created successfully!")
            print(f"📊 File size: {file_size} bytes")
            print(f"📍 File location: {os.path.abspath(AUDIO_FILE)}")
            return True
        else:
            print("❌ Audio file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error in audio generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_tts_direct():
    """Test edge-tts directly"""
    print("\n🔧 Testing edge-tts directly...")
    
    try:
        async def test():
            tts = edge_tts.Communicate("Direct test", "en-US-JennyNeural")
            await tts.save("direct_test.mp3")
            return os.path.exists("direct_test.mp3")
        
        result = asyncio.run(test())
        if result:
            print("✅ Direct edge-tts test passed")
            return True
        else:
            print("❌ Direct edge-tts test failed")
            return False
            
    except Exception as e:
        print(f"❌ Direct edge-tts error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting comprehensive audio generation test...")
    
    # Test 1: Direct edge-tts
    test1 = test_edge_tts_direct()
    
    # Test 2: Flask app function
    test2 = test_audio_generation()
    
    print(f"\n📊 Test Results:")
    print(f"Direct edge-tts: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Flask app function: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 and test2:
        print("\n🎉 All tests passed! Audio generation is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.") 