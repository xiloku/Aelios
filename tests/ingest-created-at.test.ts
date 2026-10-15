import { strict as assert } from "node:assert";
import { test, beforeEach } from "node:test";
import { DatabaseSync } from "node:sqlite";
import { readFileSync, readdirSync } from "node:fs";
import { saveIngestMessages } from "../src/db/messages";
import { normalizeIngestCreatedAt } from "../src/utils/time";

// POST /v1/ingest/messages backfill: a message may carry an original `created_at`
// (ISO 8601 with offset/Z, not future, not older than 60 days) and a stable
// `client_message_id` that dedupes re-ingested rows per namespace + conversation.
// Messages carrying neither behave exactly as before (stamped now(), never deduped).

let sqlite: DatabaseSync;
let db: any;

beforeEach(() => {
  sqlite?.close();
  sqlite = new DatabaseSync(":memory:");
  for (const file of readdirSync("migrations").filter((f: string) => f.endsWith(".sql")).sort()) {
    try {
      sqlite.exec(readFileSync(`migrations/${file}`, "utf8"));
    } catch (error) {
      // The harness runs on plain SQLite without the fts5 module; skip those statements only.
      if (!String(error).includes("fts5")) throw error;
    }
  }
  db = {
    prepare(sql: string) {
      const statement = sqlite.prepare(sql);
      let args: any[] = [];
      const api = {
        bind(...values: any[]) {
          args = values;
          return api;
        },
        async first() {
          return statement.get(...args) || null;
        },
        async all() {
          return { results: statement.all(...args) };
        },
        async run() {
          const r = statement.run(...args);
          return { meta: { changes: r.changes } };
        }
      };
      return api;
    }
  };
});

function rows(): Array<{ role: string; content: string; created_at: string; seq: number; client_message_id: string | null }> {
  return sqlite
    .prepare("SELECT role, content, created_at, seq, client_message_id FROM messages ORDER BY seq ASC")
    .all() as any;
}

const CONV = "default:default";

async function ingest(messages: any[], extra: Record<string, unknown> = {}): Promise<string[]> {
  return saveIngestMessages(db, {
    conversationId: CONV,
    namespace: "default",
    source: "ingest-test",
    messages,
    ...extra
  } as any);
}

test("normalizeIngestCreatedAt: valid past instant with offset/Z is normalized to UTC ms-precision", () => {
  const now = new Date("2026-10-15T00:00:00.000Z");
  assert.equal(normalizeIngestCreatedAt("2026-10-10T14:30:00+08:00", now), "2026-10-10T06:30:00.000Z");
  assert.equal(normalizeIngestCreatedAt("2026-10-10T06:30:00Z", now), "2026-10-10T06:30:00.000Z");
  // whitespace tolerated
  assert.equal(normalizeIngestCreatedAt("  2026-10-10T06:30:00Z  ", now), "2026-10-10T06:30:00.000Z");
});

test("normalizeIngestCreatedAt: future / stale / zoneless / malformed all return null", () => {
  const now = new Date("2026-10-15T00:00:00.000Z");
  // more than 5 min in the future
  assert.equal(normalizeIngestCreatedAt("2026-10-15T00:10:00.000Z", now), null);
  // older than 60 days
  assert.equal(normalizeIngestCreatedAt("2026-07-01T00:00:00.000Z", now), null);
  // no zone => ambiguous, rejected
  assert.equal(normalizeIngestCreatedAt("2026-10-10T06:30:00", now), null);
  // date only
  assert.equal(normalizeIngestCreatedAt("2026-10-10", now), null);
  // garbage / wrong type
  assert.equal(normalizeIngestCreatedAt("not a date", now), null);
  assert.equal(normalizeIngestCreatedAt(1234567890 as any, now), null);
  assert.equal(normalizeIngestCreatedAt(undefined, now), null);
  // within 5-min skew is allowed
  assert.equal(normalizeIngestCreatedAt("2026-10-15T00:04:00.000Z", now), "2026-10-15T00:04:00.000Z");
});

test("valid past created_at is stored as the row's created_at", async () => {
  await ingest([{ role: "user", content: "backfilled line", created_at: "2026-10-01T09:00:00+08:00" }]);
  const [row] = rows();
  assert.equal(row.content, "backfilled line");
  assert.equal(row.created_at, "2026-10-01T01:00:00.000Z");
});

test("future or malformed created_at falls back to now()", async () => {
  const before = Date.now();
  await ingest([
    { role: "user", content: "future one", created_at: "2099-01-01T00:00:00Z" },
    { role: "assistant", content: "malformed one", created_at: "yesterday" }
  ]);
  const after = Date.now();
  for (const row of rows()) {
    const ms = new Date(row.created_at).getTime();
    assert.ok(
      ms >= before - 1000 && ms <= after + 1000,
      `expected ${row.created_at} to be ~now, window [${before}, ${after}]`
    );
  }
});

test("duplicate client_message_id in the same conversation is skipped", async () => {
  const first = await ingest([{ role: "user", content: "once", client_message_id: "cmid-1" }]);
  assert.equal(first.length, 1);

  // Re-ingest the same id (even with different content) — skipped, no new row.
  const second = await ingest([{ role: "user", content: "again, different text", client_message_id: "cmid-1" }]);
  assert.equal(second.length, 0);

  const all = rows();
  assert.equal(all.length, 1);
  assert.equal(all[0].content, "once");
  assert.equal(all[0].client_message_id, "cmid-1");
});

test("same client_message_id in a different conversation is NOT skipped", async () => {
  await ingest([{ role: "user", content: "conv A", client_message_id: "shared" }]);
  await saveIngestMessages(db, {
    conversationId: "default:other",
    namespace: "default",
    source: "ingest-test",
    messages: [{ role: "user", content: "conv B", client_message_id: "shared" }]
  });
  assert.equal(rows().length, 2);
});

test("no-fields path unchanged: stamped now(), no client_message_id, seq preserved", async () => {
  const before = Date.now();
  const ids = await ingest([
    { role: "user", content: "a" },
    { role: "assistant", content: "b" },
    { role: "user", content: "c" }
  ]);
  const after = Date.now();
  assert.equal(ids.length, 3);

  const all = rows();
  assert.deepEqual(all.map((r) => r.content), ["a", "b", "c"]);
  assert.deepEqual(all.map((r) => r.seq), [0, 1, 2]);
  for (const row of all) {
    assert.equal(row.client_message_id, null);
    const ms = new Date(row.created_at).getTime();
    assert.ok(ms >= before - 1000 && ms <= after + 1000);
  }
});

test("skipped duplicate does not consume a seq slot for later rows", async () => {
  await ingest([{ role: "user", content: "seed", client_message_id: "dup" }]);
  // Batch: first is a duplicate (skipped), the next two insert with seq 0,1.
  const ids = await ingest([
    { role: "user", content: "dup again", client_message_id: "dup" },
    { role: "user", content: "fresh-1" },
    { role: "assistant", content: "fresh-2" }
  ]);
  assert.equal(ids.length, 2);
  const inserted = rows().filter((r) => r.content.startsWith("fresh"));
  assert.deepEqual(inserted.map((r) => r.seq), [0, 1]);
});
