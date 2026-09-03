"""Initial multi-profile schema with pgvector corpus storage."""

from alembic import op


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def _execute_many(sql: str) -> None:
    for statement in sql.split(";"):
        if statement.strip():
            op.execute(statement)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    _execute_many(
        """
        CREATE TABLE workspaces (
            id varchar(120) PRIMARY KEY,
            name varchar(240) NOT NULL,
            created_at timestamptz NOT NULL
        );
        CREATE TABLE newsletter_profiles (
            workspace_id varchar(120) NOT NULL REFERENCES workspaces(id),
            profile_id varchar(120) NOT NULL,
            topic varchar(240) NOT NULL,
            enabled boolean NOT NULL,
            config_hash varchar(64) NOT NULL,
            config_snapshot jsonb NOT NULL,
            updated_at timestamptz NOT NULL,
            PRIMARY KEY (workspace_id, profile_id)
        );
        CREATE TABLE research_items (
            id varchar(36) PRIMARY KEY,
            fingerprint varchar(64) NOT NULL UNIQUE,
            canonical_url text NOT NULL,
            doi varchar(300),
            title text NOT NULL,
            snippet text NOT NULL,
            abstract text,
            authors text,
            venue text,
            published_at varchar(120),
            metadata jsonb NOT NULL,
            created_at timestamptz NOT NULL,
            updated_at timestamptz NOT NULL
        );
        CREATE INDEX ix_research_items_doi ON research_items (doi);
        CREATE TABLE profile_research_items (
            workspace_id varchar(120) NOT NULL,
            profile_id varchar(120) NOT NULL,
            research_item_id varchar(36) NOT NULL REFERENCES research_items(id),
            first_seen_at timestamptz NOT NULL,
            UNIQUE (workspace_id, profile_id, research_item_id),
            FOREIGN KEY (workspace_id, profile_id)
                REFERENCES newsletter_profiles(workspace_id, profile_id)
        );
        CREATE TABLE document_chunks (
            id varchar(36) PRIMARY KEY,
            research_item_id varchar(36) NOT NULL REFERENCES research_items(id),
            chunk_index integer NOT NULL,
            content text NOT NULL,
            content_hash varchar(64) NOT NULL,
            provenance jsonb NOT NULL,
            created_at timestamptz NOT NULL,
            UNIQUE (research_item_id, content_hash)
        );
        CREATE TABLE embeddings (
            id varchar(36) PRIMARY KEY,
            chunk_id varchar(36) NOT NULL REFERENCES document_chunks(id),
            provider varchar(120) NOT NULL,
            model varchar(240) NOT NULL,
            dimensions integer,
            embedding vector,
            status varchar(30) NOT NULL,
            error text,
            created_at timestamptz NOT NULL,
            updated_at timestamptz NOT NULL,
            UNIQUE (chunk_id, provider, model)
        );
        CREATE INDEX ix_embeddings_status ON embeddings (status);
        CREATE TABLE newsletter_issues (
            id varchar(36) PRIMARY KEY,
            workspace_id varchar(120) NOT NULL,
            profile_id varchar(120) NOT NULL,
            issue_key varchar(300) NOT NULL,
            cutoff timestamptz NOT NULL,
            state varchar(40) NOT NULL,
            subject text,
            summary text,
            draft_id varchar(300),
            lease_token varchar(36) NOT NULL,
            lease_expires_at timestamptz NOT NULL,
            error text,
            created_at timestamptz NOT NULL,
            updated_at timestamptz NOT NULL,
            UNIQUE (workspace_id, profile_id, issue_key),
            FOREIGN KEY (workspace_id, profile_id)
                REFERENCES newsletter_profiles(workspace_id, profile_id)
        );
        CREATE TABLE issue_items (
            issue_id varchar(36) NOT NULL REFERENCES newsletter_issues(id),
            research_item_id varchar(36) NOT NULL REFERENCES research_items(id),
            category varchar(240) NOT NULL,
            headline text NOT NULL,
            brief text NOT NULL,
            source_label text,
            position integer NOT NULL,
            UNIQUE (issue_id, research_item_id)
        );
        CREATE TABLE source_messages (
            issue_id varchar(36) NOT NULL REFERENCES newsletter_issues(id),
            message_id varchar(300) NOT NULL,
            thread_id varchar(300),
            subject text NOT NULL,
            internal_date timestamptz NOT NULL,
            label_id varchar(300) NOT NULL,
            acknowledged_at timestamptz,
            UNIQUE (issue_id, message_id)
        );
        """
    )


def downgrade() -> None:
    _execute_many(
        """
        DROP TABLE source_messages;
        DROP TABLE issue_items;
        DROP TABLE newsletter_issues;
        DROP TABLE embeddings;
        DROP TABLE document_chunks;
        DROP TABLE profile_research_items;
        DROP TABLE research_items;
        DROP TABLE newsletter_profiles;
        DROP TABLE workspaces;
        """
    )
