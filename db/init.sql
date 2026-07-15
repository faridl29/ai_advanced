-- Auto-create databases for LiteLLM and Langfuse
-- Runs automatically on first Postgres startup via docker-entrypoint-initdb.d

CREATE DATABASE litellm;
CREATE DATABASE langfuse;

-- Grant full access to the default user
GRANT ALL PRIVILEGES ON DATABASE litellm TO ai;
GRANT ALL PRIVILEGES ON DATABASE langfuse TO ai;
