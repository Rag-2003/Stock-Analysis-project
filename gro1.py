import requests
import json

def call_groq_api(query, apikey):
    """
    Calls the Groq API with the specified query and API key.

    Parameters:
        query (str): The content of the query to be sent to the API.
        apikey (str): The API key for authorization.

    Returns:
        dict: The response from the API.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {apikey}"
    }
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Example usage:
response = call_groq_api("Explain the importance of fast language models", "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")
print(response)