import os

def check_env_vars():
    """Simple utility to check which environment variables are accessible."""
    env_vars = {
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY", "Not set"),
        "DEEPSEEK_API_URL": os.getenv("DEEPSEEK_API_URL", "Not set"),
        "DATABASE_URL": os.getenv("DATABASE_URL", "Not set"),
        "ENCRYPTION_KEY": os.getenv("ENCRYPTION_KEY", "Not set"),
        "CORS_ORIGINS": os.getenv("CORS_ORIGINS", "Not set"),
        "PORT": os.getenv("PORT", "Not set"),
        "USE_TOR_PROXY": os.getenv("USE_TOR_PROXY", "Not set"),
    }
    
    # Mask sensitive values
    for key in ["DEEPSEEK_API_KEY", "ENCRYPTION_KEY"]:
        if env_vars[key] != "Not set" and len(env_vars[key]) > 8:
            env_vars[key] = env_vars[key][:4] + "..." + env_vars[key][-4:]
    
    return env_vars 