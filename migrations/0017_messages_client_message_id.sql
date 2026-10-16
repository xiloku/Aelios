-- Backfill-friendly ingest: let callers supply a stable client_message_id so a
-- re-ingested chat (replaying a week of old messages) does not insert duplicate rows.
--
-- Separate from client_message_hash: that column holds a bucketed content hash with a
-- GLOBAL unique index, used by the live chat path (saveUserMessages). The ingest dedup
-- key is caller-supplied and scoped to a single conversation, so it gets its own column
-- and a (namespace, conversation_id, client_message_id) partial unique index. Nullable:
-- messages without a client_message_id behave exactly as before (NULLs are not unique).
ALTER TABLE messages ADD COLUMN client_message_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_client_message_id_unique
ON messages(namespace, conversation_id, client_message_id)
WHERE client_message_id IS NOT NULL;
