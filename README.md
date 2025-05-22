# PostgreSQL MCP Server

[![smithery badge](https://smithery.ai/badge/@gldc/mcp-postgres)](https://smithery.ai/server/@gldc/mcp-postgres)

<a href="https://glama.ai/mcp/servers/@gldc/mcp-postgres">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@gldc/mcp-postgres/badge" />
</a>

A PostgreSQL MCP server implementation using the [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) Python SDK- an open protocol that enables seamless integration between LLM applications and external data sources. This server allows AI agents to interact with PostgreSQL databases through a standardized interface.

## Features

- List database schemas
- List tables within schemas
- Describe table structures
- List table constraints and relationships
- Get foreign key information
- Execute SQL queries

## Quick Start

```bash
# Run the server without a DB connection (useful for Glama or inspection)
python postgres_server.py

# With a live database – pick one method:
export POSTGRES_CONNECTION_STRING="postgresql://user:pass@host:5432/db"
python postgres_server.py

# …or…
python postgres_server.py --conn "postgresql://user:pass@host:5432/db"

# Or using Docker (build once, then run):
# docker build -t mcp-postgres . && docker run -p 8000:8000 mcp-postgres
```

## Installation

### Installing via Smithery

To install PostgreSQL MCP Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@gldc/mcp-postgres):

```bash
npx -y @smithery/cli install @gldc/mcp-postgres --client claude
```

### Manual Installation
1. Clone this repository:
```bash
git clone <repository-url>
cd mcp-postgres
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the MCP server.

   ```bash
   # Without a connection string (server starts, DB‑backed tools will return a friendly error)
   python postgres_server.py

   # Or set the connection string via environment variable:
   export POSTGRES_CONNECTION_STRING="postgresql://username:password@host:port/database"
   python postgres_server.py

   # Or pass it using the --conn flag:
   python postgres_server.py --conn "postgresql://username:password@host:port/database"
   ```
2. The server provides the following tools:

- `query`: Execute SQL queries against the database
- `list_schemas`: List all available schemas
- `list_tables`: List all tables in a specific schema
- `describe_table`: Get detailed information about a table's structure
- `get_foreign_keys`: Get foreign key relationships for a table
- `find_relationships`: Discover both explicit and implied relationships for a table

### Running with Docker

Build the image:

```bash
docker build -t mcp-postgres .
```

Run the container without a database connection (the server stays inspectable):

```bash
docker run -p 8000:8000 mcp-postgres
```

Run with a live PostgreSQL database by supplying `POSTGRES_CONNECTION_STRING`:

```bash
docker run \
  -e POSTGRES_CONNECTION_STRING="postgresql://username:password@host:5432/database" \
  -p 8000:8000 \
  mcp-postgres
```

*If the environment variable is omitted, a server boots normally and all database‑backed tools return a friendly “connection string is not set” message until you provide it.*

## Reliability and Fault Tolerance Features

The PostgreSQL MCP Server incorporates several features to enhance its stability and resilience:

### Connection Pooling
The server utilizes connection pooling to manage database connections efficiently, reducing the overhead of establishing a new connection for each request.
- **Configuration (Environment Variables)**:
    - `POSTGRES_POOL_MIN_CONNS`: Minimum number of connections in the pool (default: `1`).
    - `POSTGRES_POOL_MAX_CONNS`: Maximum number of connections in the pool (default: `5`).

### Retry Mechanisms
Database operations, including establishing connections and executing queries, are automatically retried upon transient failures. This helps the server recover from temporary network issues or database load.
- **Configuration (Environment Variables)**:
    - `RETRY_ATTEMPTS`: Number of retry attempts for failing operations (default: `3`).
    - `RETRY_WAIT_MULTIPLIER`: Multiplier for exponential backoff between retries (default: `1` second).
    - `RETRY_WAIT_MAX`: Maximum wait time between retries (default: `10` seconds).

### Health Check Tool
A dedicated MCP tool named `health_check` is available to verify the server's database connectivity.
- **Purpose**: Allows clients or monitoring systems to confirm that the server can successfully connect to the PostgreSQL database and perform a basic query.
- **Usage**: Invoke the `health_check` tool via an MCP client.
- **Possible Return Messages**:
    - `"Health check OK: Database connection successful."`
    - `"Health check FAILED: Could not connect to database after retries (error details)."`
    - `"Health check FAILED: Database error (error details)."`
    - `"Health check FAILED: Query execution error (unexpected result)."`

### Graceful Shutdown
The server is designed to handle `SIGINT` (Ctrl+C) and `SIGTERM` signals to shut down gracefully. This process ensures that resources, particularly database connections in the pool, are properly released, preventing orphaned connections and ensuring a clean termination.

### Configuration with mcp.json

To integrate this server with MCP-compatible tools (like Cursor), add it to your `~/.cursor/mcp.json`:

```json
{
  "servers": {
    "postgres": {
      "command": "/path/to/venv/bin/python",
      "args": [
        "/path/to/postgres_server.py"
      ],
      "env": {
        "POSTGRES_CONNECTION_STRING": "postgresql://username:password@host:5432/database?ssl=true"
      }
    }
  }
}
```

*If `POSTGRES_CONNECTION_STRING` is omitted, the server still starts and is fully inspectable; database‑backed tools will simply return an informative error until the variable is provided.*

Replace:
- `/path/to/venv` with your virtual environment path
- `/path/to/postgres_server.py` with the absolute path to the server script

## Security

- Never expose sensitive database credentials in your code
- Use environment variables or secure configuration files for database connection strings
- Consider using connection pooling for better resource management
- Implement proper access controls and user authentication

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Projects

- [MCP Specification](https://github.com/modelcontextprotocol/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Servers](https://github.com/modelcontextprotocol/servers)

## License

MIT License

Copyright (c) 2025 gldc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.