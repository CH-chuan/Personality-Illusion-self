"""
Shared API utilities for behavioral tasks.
Extracted from cells 3-5 in all behavioral task notebooks.
Extended to support LiteLLM.
"""

# import os
import os
# import anthropic
import openai
import backoff
# from together import Together



from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def initialize_clients():
    """
    Initialize all API clients including LiteLLM.
    Source: Cell 3 in all notebooks (Honesty.ipynb, IAT.ipynb, RiskTaking.ipynb, Sycophancy.ipynb)
    Extended: Added LiteLLM support
    """
    clients = {}

    # # Try Anthropic
    # try:
    #     api_key = os.environ.get('ANTHROPIC_API_KEY')
    #     if api_key:
    #         clients['anthropic'] = anthropic.Anthropic(api_key=api_key)
    #         print("✓ Anthropic client initialized")
    # except Exception as e:
    #     print(f"✗ Failed to initialize Anthropic client: {e}")

    # # Try OpenAI
    # try:
    #     api_key = os.environ.get('OPENAI_API_KEY')
    #     if api_key:
    #         clients['openai'] = openai.OpenAI(api_key=api_key)
    #         print("✓ OpenAI client initialized")
    # except Exception as e:
    #     print(f"✗ Failed to initialize OpenAI client: {e}")

    # # Try Together AI
    # try:
    #     api_key = os.environ.get('TOGETHER_API_KEY')
    #     if api_key:
    #         clients['together'] = Together(api_key=api_key)
    #         print("✓ Together AI client initialized")
    # except Exception as e:
    #     print(f"✗ Failed to initialize Together AI client: {e}")

    # # Try OpenRouter
    # try:
    #     api_key = os.environ.get('OPENROUTER_API_KEY')
    #     if api_key:
    #         clients['openrouter'] = openai.OpenAI(
    #             base_url="https://openrouter.ai/api/v1",
    #             api_key=api_key
    #         )
    #         print("✓ OpenRouter client initialized")
    # except Exception as e:
    #     print(f"✗ Failed to initialize OpenRouter client: {e}")

    # Try LiteLLM (OpenAI-compatible interface)
    try:
        api_key = os.environ.get('LITELLM_API_KEY')
        base_url = os.environ.get('LITELLM_BASE_URL', 'http://localhost:4000')
        if api_key:
            clients['litellm'] = openai.OpenAI(
                base_url=base_url,
                api_key=api_key
            )
            print(f"✓ LiteLLM client initialized (base_url: {base_url})")
    except Exception as e:
        print(f"✗ Failed to initialize LiteLLM client: {e}")

    return clients


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def call_anthropic_api(client, system_prompt, user_prompt, model_name, temperature=0.7, max_tokens=32, **kwargs):
    """
    Call Anthropic API with retry logic.
    Source: Cell 4 in all notebooks
    """
    messages = [{"role": "user", "content": user_prompt}]

    params = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages
    }

    if system_prompt.strip():
        params["system"] = system_prompt

    response = client.messages.create(**params)
    return response.content[0].text.strip()


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError), max_tries=3)
def call_openai_api(client, system_prompt, user_prompt, model_name, temperature=0.7, max_tokens=32, **kwargs):
    """
    Call OpenAI API with retry logic.
    Source: Cell 4 in all notebooks
    """
    messages = [{"role": "user", "content": user_prompt}]

    if system_prompt.strip():
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def call_together_api(client, system_prompt, user_prompt, model_name, temperature=0.7, max_tokens=32, **kwargs):
    """
    Call Together AI API with retry logic.
    Source: Cell 4 in all notebooks
    """
    messages = [{"role": "user", "content": user_prompt}]

    if system_prompt.strip():
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def call_openrouter_api(client, system_prompt, user_prompt, model_name, temperature=0.7, max_tokens=32, **kwargs):
    """
    Call OpenRouter API with retry logic.
    Source: Cell 4 in all notebooks
    """
    messages = [{"role": "user", "content": user_prompt}]

    if system_prompt.strip():
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://yoursite.com",
            "X-Title": "Self-Regulation Research"
        },
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError), max_tries=3)
def call_litellm_api(client, system_prompt, user_prompt, model_name, temperature=0.7, max_tokens=32, **kwargs):
    """
    Call LiteLLM API (OpenAI-compatible) with retry logic.
    Added: LiteLLM support for vLLM and other providers
    """
    messages = [{"role": "user", "content": user_prompt}]

    if system_prompt.strip():
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def route_api_call(clients, request):
    """
    Route API call to appropriate provider.
    Source: Cell 4 in all notebooks
    Extended: Added LiteLLM routing
    """
    provider = request['provider']

    if provider == 'anthropic':
        return call_anthropic_api(clients['anthropic'], **{k: v for k, v in request.items()
                                 if k not in ['provider']})
    elif provider == 'openai':
        return call_openai_api(clients['openai'], **{k: v for k, v in request.items()
                              if k not in ['provider']})
    elif provider == 'together':
        return call_together_api(clients['together'], **{k: v for k, v in request.items()
                                if k not in ['provider']})
    elif provider == 'openrouter':
        return call_openrouter_api(clients['openrouter'], **{k: v for k, v in request.items()
                                  if k not in ['provider']})
    elif provider == 'litellm':
        return call_litellm_api(clients['litellm'], **{k: v for k, v in request.items()
                               if k not in ['provider']})
    else:
        raise ValueError(f"Unknown provider: {provider}")


def process_batch_requests(clients, batch_requests, max_workers=8):
    """
    Process multiple API requests concurrently.
    Source: Cell 4 in all notebooks
    """
    results = [None] * len(batch_requests)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(route_api_call, clients, req): i
            for i, req in enumerate(batch_requests)
        }

        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
                completed += 1
                if completed % 10 == 0:
                    print(f"Completed {completed}/{len(batch_requests)} requests")
            except Exception as e:
                print(f"Request {index} failed: {e}")
                results[index] = "ERROR"

    return results