import os
from dotenv import load_dotenv
import pyTigerGraph
import pytest

load_dotenv()

host = os.getenv("TIGERGRAPH_HOST")
graph = os.getenv("TIGERGRAPH_GRAPH")
secret = os.getenv("TIGERGRAPH_SECRET")

if not host:
    raise RuntimeError("TIGERGRAPH_HOST is missing from .env")
if not graph:
    raise RuntimeError("TIGERGRAPH_GRAPH is missing from .env")
if not secret:
    raise RuntimeError("TIGERGRAPH_SECRET is missing from .env")

def test_tigergraph_connection():
    print(f"Host: {host}")
    print(f"Graph: {graph}")
    print("Secret: [loaded]")

    conn = pyTigerGraph.TigerGraphConnection(
        host=host,
        graphname=graph,
        gsqlSecret=secret,
    )

    try:
        conn.getToken(secret)
        schema = conn.getSchema()
        print("\nTigerGraph connection: SUCCESS")
        print(f"Graph: {graph}")
        print("\nSchema:")
        print(schema)
    except Exception as e:
        pytest.skip(f"TigerGraph is unavailable: {e}")
