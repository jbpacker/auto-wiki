import json
import os


def get_creds(cred_path: str = None):
    """
    Get credentials from the environment.
    """
    if cred_path is not None:
        f = open(cred_path, "r")
        keys = json.load(f)
        f.close()

        os.environ["OPENAI_API_KEY"] = keys["openai_api_key"]
        os.environ["PINECONE_API_KEY"] = keys["pinecone_api_key"]
        os.environ["GOOGLE_CSE_ID"] = keys["google_cse_id"]
        os.environ["GOOGLE_API_KEY"] = keys["google_api_key"]
    elif os.environ.get("OPENAI_API_KEY") is not None:
        keys = {
            "openai_api_key": os.environ.get("OPENAI_API_KEY"),
            "pinecone_api_key": os.environ.get("PINECONE_API_KEY"),
            "google_cse_id": os.environ.get("GOOGLE_CSE_ID"),
            "google_api_key": os.environ.get("GOOGLE_API_KEY"),
        }
    else:
        assert False, "Unable to find credentials."

    assert (
        keys["openai_api_key"] is not None
    ), "could not find openai_api_key in environment"
    assert (
        keys["pinecone_api_key"] is not None
    ), "could not find pinecone_api_key in environment"
    assert (
        keys["google_cse_id"] is not None
    ), "could not find google_cse_id in environment"
    assert (
        keys["google_api_key"] is not None
    ), "could not find google_api_key in environment"

    return keys
