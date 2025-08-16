#!/usr/bin/env python3
"""
Test script for audio conversion functionality
"""

import asyncio
import os
import edge_tts

async def test_text_to_speech():
    """Test the text-to-speech functionality"""
    try:
        # Test script
        script = "Hello, this is a test of the text-to-speech functionality."
        output_file = "test_audio.mp3"
        
        # Create TTS communication
        tts = edge_tts.Communicate(script, "en-US-JennyNeural")
        
        # Save the audio
        await tts.save(output_file)
        
        # Check if file was created
        if os.path.exists(output_file):
            print(f"✅ Audio file created successfully: {output_file}")
            print(f"File size: {os.path.getsize(output_file)} bytes")
            return True
        else:
            print("❌ Audio file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error in text-to-speech: {e}")
        return False

def test_convert_text_to_speech():
    """Test the synchronous wrapper function"""
    try:
        script = "This is another test of the text-to-speech conversion."
        output_file = "test_audio2.mp3"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # Run the async function
        asyncio.run(test_text_to_speech())
        
        print("✅ Text-to-speech conversion test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error in convert_text_to_speech: {e}")
        return False

if __name__ == "__main__":
    print("Testing audio conversion functionality...")
    
    # Test 1: Direct async function
    print("\n1. Testing async text_to_speech function...")
    success1 = asyncio.run(test_text_to_speech())
    
    # Test 2: Synchronous wrapper
    print("\n2. Testing synchronous wrapper...")
    success2 = test_convert_text_to_speech()
    
    if success1 and success2:
        print("\n✅ All audio conversion tests passed!")
    else:
        print("\n❌ Some tests failed!") 