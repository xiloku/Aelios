/**
 * Shared parsing / normalization helpers used across the Worker.
 *
 * This module is the single source of truth for small utilities that were
 * previously duplicated in memory/*, api/* and utils/request.ts.
 */

/** Read a non-empty trimmed string, returning `null` for missing/blank values. */
export function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

/** Variant of `readString` that returns `undefined` instead of `null`. */
export function readStringOrUndefined(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

/** Normalize an array of values to a list of non-empty trimmed strings. */
export function readStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string").map((item) => item.trim()).filter(Boolean);
}

/**
 * Parse a string array from a variety of input shapes:
 * - an array of strings (trimmed, blanks removed)
 * - a JSON-encoded string array
 * - a plain string (treated as a single-element array)
 *
 * Differs from `readStringArray` which only accepts an actual array.
 */
export function parseStringArray(value: unknown): string[] {
  if (Array.isArray(value)) return value.filter((item): item is string => typeof item === "string").map((item) => item.trim()).filter(Boolean);
  if (typeof value !== "string" || !value.trim()) return [];

  try {
    const parsed = JSON.parse(value) as unknown;
    if (Array.isArray(parsed)) return parseStringArray(parsed);
  } catch {
    // Plain metadata strings are also valid single tags.
  }

  return [value.trim()];
}

/** Clamp a numeric value to the unit interval [0, 1]. */
export function clampScore(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? Math.min(Math.max(value, 0), 1) : fallback;
}

/**
 * Extract the outermost JSON object from a string.
 * First tries a direct parse; if that fails, scans for the first `{` and last `}`.
 */
export function extractJsonObject(text: string): unknown | null {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    // Some providers wrap JSON in prose; pull out the outermost object.
  }

  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) return null;

  try {
    return JSON.parse(text.slice(start, end + 1)) as unknown;
  } catch {
    return null;
  }
}

/**
 * Tolerant extraction of the FIRST balanced JSON object in a string.
 *
 * `extractJsonObject` slices from the first `{` to the LAST `}`. That breaks on two shapes
 * a chat model routinely produces:
 *  - prose or a second object after the diary JSON (the last `}` then belongs to the wrong
 *    object, so the slice is not valid JSON), and
 *  - a reply truncated by a token limit mid-object (there is no final `}`, so the last `}`
 *    is an inner one and the slice is unbalanced).
 *
 * This scanner walks the text tracking string state and escapes, so braces inside string
 * values never affect depth, and returns the first top-level `{...}` that balances. It also
 * strips a leading ```json / ``` fence. Returns the parsed value, or null when no balanced
 * object exists (e.g. a genuinely truncated reply) — the caller then retries.
 */
export function extractFirstJsonObject(text: string): unknown | null {
  // Fast path: a clean reply parses directly.
  try {
    return JSON.parse(text) as unknown;
  } catch {
    // fall through to scanning
  }

  // Drop a surrounding markdown code fence if present (```json ... ``` or ``` ... ```).
  const unfenced = text.replace(/```[a-zA-Z0-9]*\s*/g, "").replace(/```/g, "");

  const start = unfenced.indexOf("{");
  if (start === -1) return null;

  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < unfenced.length; i += 1) {
    const ch = unfenced[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === "\\") {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
    } else if (ch === "{") {
      depth += 1;
    } else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        try {
          return JSON.parse(unfenced.slice(start, i + 1)) as unknown;
        } catch {
          return null;
        }
      }
    }
  }

  // No balanced object closed (truncated reply).
  return null;
}
