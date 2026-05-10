from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from .models import ChunkRecord, RelationEdge


class GraphStore:
    def __init__(self, uri: str, username: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self.driver.close()

    def ensure_constraints(self) -> None:
        queries = [
            "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        ]
        with self.driver.session() as session:
            for query in queries:
                session.run(query)

    def get_processed_chunk_ids(self) -> set[str]:
        """返回已完整处理的 chunk_id（有 Chunk 节点且有 MENTIONS 关系）。"""
        query = "MATCH (c:Chunk)-[:MENTIONS]->(:Entity) RETURN DISTINCT c.chunk_id AS chunk_id"
        with self.driver.session() as session:
            result = session.run(query)
            return {record["chunk_id"] for record in result}

    def upsert_chunk(self, chunk: ChunkRecord) -> None:
        query = """
        MERGE (d:Document {doc_id: $doc_id})
        ON CREATE SET d.title = $title, d.source_path = $source_path
        SET d.modality = $modality
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.text = $text,
            c.order = $order,
            c.page_number = $page_number,
            c.source_path = $source_path,
            c.title = $title
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        with self.driver.session() as session:
            session.run(
                query,
                doc_id=chunk.doc_id,
                title=chunk.title,
                source_path=chunk.source_path,
                modality=chunk.modality,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                order=chunk.order,
                page_number=chunk.page_number,
            )

    def upsert_relations(self, chunk: ChunkRecord, relations: list[RelationEdge]) -> int:
        if not relations:
            return 0
        batch = [
            {
                "source": rel.source,
                "target": rel.target,
                "relation": rel.relation,
                "evidence": rel.evidence,
            }
            for rel in relations
        ]
        query = """
        UNWIND $batch AS item
        MERGE (c:Chunk {chunk_id: $chunk_id})
        MERGE (s:Entity {name: item.source})
        MERGE (t:Entity {name: item.target})
        MERGE (c)-[:MENTIONS]->(s)
        MERGE (c)-[:MENTIONS]->(t)
        MERGE (s)-[r:RELATED_TO {type: item.relation, chunk_id: $chunk_id}]->(t)
        SET r.evidence = item.evidence,
            r.source_path = $source_path,
            r.title = $title
        """
        with self.driver.session() as session:
            session.run(
                query,
                chunk_id=chunk.chunk_id,
                source_path=chunk.source_path,
                title=chunk.title,
                batch=batch,
            )
        return len(relations)

    def query_entity_relations(
        self,
        entity_names: list[str],
        limit: int = 8,
        entity_limit: int = 5,
        max_hops: int = 1,
    ) -> list[dict[str, Any]]:
        entity_names = [name.strip() for name in entity_names if name.strip()]
        if not entity_names:
            return []
        max_hops = max(1, int(max_hops))

        rel_hop_pattern = f"*1..{max_hops}"
        query_template = """
        UNWIND $entity_names AS query_entity
        MATCH (e:Entity)
        WHERE e.name = query_entity OR e.name CONTAINS query_entity OR query_entity CONTAINS e.name
        WITH DISTINCT query_entity, e,
             CASE
                WHEN e.name = query_entity THEN 0
                WHEN e.name CONTAINS query_entity THEN 1
                ELSE 2
             END AS score
        ORDER BY score ASC, size(e.name) ASC
        WITH collect({query_entity: query_entity, matched_entity: e.name})[..$entity_limit] AS matched
        UNWIND matched AS item
        MATCH path=(e:Entity {name: item.matched_entity})-[:RELATED_TO__REL_HOP_PATTERN__]-(neighbor:Entity)
        WITH item, path, relationships(path) AS rels, length(path) AS hop
        UNWIND rels AS r
        RETURN DISTINCT item.query_entity AS query_entity,
               item.matched_entity AS matched_entity,
               startNode(r).name AS source,
               endNode(r).name AS target,
               r.type AS relation,
               r.evidence AS evidence,
               r.source_path AS source_path,
               r.title AS title,
               hop AS path_hops
        LIMIT $limit
        """
        query = query_template.replace("__REL_HOP_PATTERN__", rel_hop_pattern)
        with self.driver.session() as session:
            result = session.run(
                query,
                entity_names=entity_names,
                limit=limit,
                entity_limit=entity_limit,
            )
            return [record.data() for record in result]
