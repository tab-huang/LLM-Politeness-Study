"""
Test GPT-5.2 API call in isolation
"""

import os
import asyncio
from openai import AsyncOpenAI

# ============================================================================
# SETUP
# ============================================================================

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

# ============================================================================
# TEST GPT-5.2 WITH STREAMING (as shown in OpenRouter docs)
# ============================================================================

async def test_gpt52_streaming():
    """Test GPT-5.2 with streaming (OpenRouter recommended method)"""
    print("\n🔍 Testing GPT-5.2 with STREAMING...\n")
    
    try:
        stream = await client.chat.completions.create(
            model="openai/gpt-5.2",
            messages=[
                {
                    "role": "user",
                    "content": "What is 2+2? Answer with only the letter.\nA) 3\nB) 4\nC) 5\nD) 6"
                }
            ],
            stream=True,
            max_tokens=20
        )
        
        response_text = ""
        reasoning_tokens = 0
        
        async for chunk in stream:
            # Get content
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response_text += content
                print(content, end='', flush=True)
            
            # Get usage info (comes in final chunk)
            if hasattr(chunk, 'usage') and chunk.usage:
                reasoning_tokens = getattr(chunk.usage, 'reasoning_tokens', 0)
        
        print(f"\n\n✅ GPT-5.2 STREAMING SUCCESS")
        print(f"Response: '{response_text.strip()}'")
        print(f"Reasoning tokens: {reasoning_tokens}")
        
        return response_text.strip()
        
    except Exception as e:
        print(f"\n❌ GPT-5.2 STREAMING FAILED: {e}")
        return None


# ============================================================================
# TEST GPT-5.2 WITHOUT STREAMING (standard method)
# ============================================================================

async def test_gpt52_no_stream():
    """Test GPT-5.2 without streaming"""
    print("\n🔍 Testing GPT-5.2 WITHOUT streaming...\n")
    
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-5.2",
            messages=[
                {
                    "role": "user",
                    "content": "What is 2+2? Answer with only the letter.\nA) 3\nB) 4\nC) 5\nD) 6"
                }
            ],
            max_tokens=20
        )
        
        response_text = response.choices[0].message.content
        
        print(f"✅ GPT-5.2 NO STREAM SUCCESS")
        print(f"Response: '{response_text}'")
        print(f"Model: {response.model}")
        
        return response_text.strip()
        
    except Exception as e:
        print(f"❌ GPT-5.2 NO STREAM FAILED: {e}")
        return None


# ============================================================================
# TEST IF MODEL EXISTS
# ============================================================================

async def check_model_exists():
    """Check if GPT-5.2 is available on your OpenRouter account"""
    print("\n🔍 Checking available models...\n")
    
    import requests
    
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}"
    }
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers
        )
        
        if response.status_code == 200:
            models = response.json()['data']
            
            # Find GPT models
            gpt_models = [m for m in models if 'gpt' in m['id'].lower() or 'o1' in m['id'].lower() or 'o3' in m['id'].lower()]
            
            print("📋 Available GPT/reasoning models:\n")
            for model in gpt_models:
                model_id = model['id']
                pricing = model.get('pricing', {})
                prompt_price = pricing.get('prompt', 'N/A')
                
                # Highlight GPT-5.2
                if 'gpt-5.2' in model_id:
                    print(f"  ✨ {model_id} (FOUND!)")
                else:
                    print(f"  • {model_id}")
                
                print(f"    Prompt: ${prompt_price}/1M tokens")
            
            # Check if GPT-5.2 exists
            has_gpt52 = any('gpt-5.2' in m['id'] for m in gpt_models)
            
            if has_gpt52:
                print("\n✅ GPT-5.2 is available on your account")
            else:
                print("\n⚠️  GPT-5.2 NOT FOUND - try these alternatives:")
                alternatives = [m['id'] for m in gpt_models if 'gpt-4' in m['id'] or 'o1' in m['id']]
                for alt in alternatives[:5]:
                    print(f"  • {alt}")
            
            return has_gpt52
            
        else:
            print(f"❌ Failed to fetch models: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False


# ============================================================================
# RUN ALL TESTS
# ============================================================================

async def main():
    """Run all GPT-5.2 tests"""
    
    print("="*70)
    print("GPT-5.2 DIAGNOSTIC TEST")
    print("="*70)
    
    
    
    # Test 1: Check if model exists
    model_exists = await check_model_exists()
    
    if not model_exists:
        print("\n⚠️  GPT-5.2 not available - stopping tests")
        return
    
    # Test 2: Try without streaming
    print("\n" + "="*70)
    response_no_stream = await test_gpt52_no_stream()
    
    # Test 3: Try with streaming
    print("\n" + "="*70)
    response_stream = await test_gpt52_streaming()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model exists: {'✅' if model_exists else '❌'}")
    print(f"No stream works: {'✅' if response_no_stream else '❌'}")
    print(f"Streaming works: {'✅' if response_stream else '❌'}")
    
    if response_no_stream:
        print(f"\nNo stream response: '{response_no_stream}'")
    if response_stream:
        print(f"Streaming response: '{response_stream}'")


if __name__ == "__main__":
    asyncio.run(main())