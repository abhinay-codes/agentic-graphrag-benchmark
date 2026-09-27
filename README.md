# Agentic GraphRAG

This repository contains an Agentic GraphRAG system for TigerGraph.

## TigerGraph Cloud Setup

The graph loader and retriever are configured to connect securely to a TigerGraph Savanna Cloud instance.

**Important Security Notice:** NEVER commit your `TIGERGRAPH_SECRET`.

To connect to your live TigerGraph Cloud instance, create a `.env` file in the root of the project (or export these environment variables) containing:

```env
TIGERGRAPH_HOST=https://your-instance.i.tgcloud.io
TIGERGRAPH_GRAPH=AgenticGraphRAG
TIGERGRAPH_SECRET=your_gsql_secret
```

### Deployment

1. **Deploy the Schema:** See `reports/tigergraph_deployment_checklist.txt` for the GSQL commands to build the `Document`, `Chunk`, and `Entity` graph schema.
2. **Load the Graph:** Once configured, execute the loader:
   ```bash
   python -m src.ingestion.tigergraph_loader
   ```
3. **Verify:** Use the `GraphRetriever` API in Python to verify the loading process. Expected corpus counts are 19,032 total vertices and 16,081 edges.

*Note: If no connection is provided or variables are omitted, the scripts will run in a safe "mock" mode for development and testing.*
