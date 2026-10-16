export function nowIso(): string {
  return new Date().toISOString();
}

// Ingest backfill bounds: a supplied created_at may run slightly ahead of the server
// clock (client skew) but not into the real future, and may not be older than 60 days.
const INGEST_CREATED_AT_SKEW_MS = 5 * 60 * 1000;
const INGEST_CREATED_AT_MAX_AGE_MS = 60 * 24 * 60 * 60 * 1000;

// Reject anything that is not an ISO 8601 instant carrying an explicit offset or Z.
// `new Date("2026-10-10")` and `new Date("2026-10-10T12:00:00")` both parse, but a
// backfill timestamp without a zone is ambiguous, so we require an offset / Z.
const ISO_WITH_ZONE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/;

/**
 * Normalize a caller-supplied `created_at` for message ingest.
 *
 * Returns the timestamp re-rendered in the exact format `nowIso()` produces
 * (`YYYY-MM-DDTHH:mm:ss.sssZ`, UTC) when it is a valid ISO 8601 instant with an
 * explicit zone, is not in the future beyond a 5-minute skew allowance, and is not
 * older than 60 days. Returns `null` for anything else, so callers fall back to now.
 *
 * @param now injectable clock for testing; defaults to the real time.
 */
export function normalizeIngestCreatedAt(value: unknown, now: Date = new Date()): string | null {
  if (typeof value !== "string" || !ISO_WITH_ZONE.test(value.trim())) return null;
  const parsed = new Date(value.trim());
  const ms = parsed.getTime();
  if (Number.isNaN(ms)) return null;
  const nowMs = now.getTime();
  if (ms > nowMs + INGEST_CREATED_AT_SKEW_MS) return null;
  if (ms < nowMs - INGEST_CREATED_AT_MAX_AGE_MS) return null;
  return parsed.toISOString();
}

/**
 * YYYY-MM-DD in a time zone, assembled from date parts.
 *
 * `Intl.DateTimeFormat(locale).format()` only renders an ISO-shaped string when the
 * runtime ships full ICU data for that locale. Node builds with small-icu (and any
 * runtime missing `en-CA`) fall back to `en`, which renders `M/D/YYYY`. Every date
 * *label* in this repo is compared and parsed as `YYYY-MM-DD`, so labels are built
 * from `formatToParts` instead of trusting the locale's rendering.
 */
export function formatDateLabel(date: Date, timeZone: string): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).formatToParts(date);
  const values = new Map(parts.map((part) => [part.type, part.value]));
  return `${values.get("year")}-${values.get("month")}-${values.get("day")}`;
}
